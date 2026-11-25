"""
Diffusion evaluation — same held-out pages for every method
-----------------------------------------------------------
    python -m models.diffusion.evaluate            # writes models/diffusion/metrics.json

Test pages: data/diffusion/test (24 Kaggle resume PDFs, one per category, never used in training). Each page is
degraded with one named preset (noise · blur · shadow_lighting · low_contrast_jpeg · mixed_scan, cycled, fixed seed).
Methods on the identical degraded input: none (degraded) · Autoencoder · Diffusion · each "+ guard" (keep the degraded
input when local OCR reads the enhanced image less confidently — decided without ground truth). Clean page = ceiling.

OCR: RapidOCR (local, deterministic) so results are reproducible; ground truth = the PDF text layer.
Metrics: CER, WER, word recall · PSNR, SSIM vs the clean page · ink F1 (text/layout preservation, 1 px tolerance) ·
enhancement time and peak memory · failure cases (enhanced CER worse than the degraded input).
"""

import hashlib
import json
import os
import re
import sys
import threading
import time
import unicodedata
from datetime import datetime

import numpy as np
from PIL import Image, ImageFilter

from models.diffusion import data as dd
from models.diffusion.degradations import PRESETS, degrade_page
from models.diffusion.diffusion import MODEL_DIR, ResumeDiffusionEnhancer

OCR_CACHE = os.path.join(dd.DATA_DIR, "ocr_cache")


# ---------------- text metrics ----------------
def normalise(text):
    text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", text.lower()).strip()


def cer(reference, hypothesis):
    from rapidfuzz.distance import Levenshtein
    ref, hyp = normalise(reference), normalise(hypothesis)
    return Levenshtein.distance(ref, hyp) / max(1, len(ref))


def wer(reference, hypothesis):
    from rapidfuzz.distance import Levenshtein
    ref, hyp = normalise(reference).split(), normalise(hypothesis).split()
    return Levenshtein.distance(ref, hyp) / max(1, len(ref))


def word_recall(reference, hypothesis):
    ref, hyp = normalise(reference).split(), set(normalise(hypothesis).split())
    return float(np.mean([w in hyp for w in ref])) if ref else 0.0


# ---------------- image metrics ----------------
def _arr(img):
    return np.asarray(img.convert("L"), np.float64) / 255.0


def _match(ref, img):
    return img if img.size == ref.size else img.resize(ref.size, Image.LANCZOS)


def psnr(ref, img):
    err = float(np.mean((_arr(ref) - _arr(_match(ref, img))) ** 2))
    return float("inf") if err == 0 else 10 * np.log10(1.0 / err)


def ssim(ref, img, win=7):
    """Mean SSIM with a uniform 7x7 window."""
    from scipy.ndimage import uniform_filter

    a, b = _arr(ref), _arr(_match(ref, img))
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_a, mu_b = uniform_filter(a, win), uniform_filter(b, win)
    va = uniform_filter(a * a, win) - mu_a ** 2
    vb = uniform_filter(b * b, win) - mu_b ** 2
    cov = uniform_filter(a * b, win) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))
    return float(s.mean())


def _otsu(a):
    hist, edges = np.histogram(a, bins=256, range=(0, 1))
    p = hist / max(1, hist.sum())
    w0, mu = np.cumsum(p), np.cumsum(p * edges[:-1])
    between = (mu[-1] * w0 - mu) ** 2 / np.maximum(w0 * (1 - w0), 1e-12)
    return edges[int(np.argmax(between))]


def ink_f1(ref, img):
    """Text / layout preservation: F1 between the ink masks of the clean page and the output (1 px tolerance)."""
    a, b = _arr(ref), _arr(_match(ref, img))
    ink_a, ink_b = a < min(0.5, _otsu(a)), b < _otsu(b)
    dil = lambda m: np.asarray(Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(3))) > 0  # noqa
    precision = (ink_b & dil(ink_a)).sum() / max(1, ink_b.sum())
    recall = (ink_a & dil(ink_b)).sum() / max(1, ink_a.sum())
    return float(2 * precision * recall / max(1e-9, precision + recall))


# ---------------- resources ----------------
class PeakMemory:
    """Peak resident-memory increase (MB) of this process while the block runs."""

    def __enter__(self):
        import psutil
        self.proc = psutil.Process()
        self.base = self.peak = self.proc.memory_info().rss
        self.running = True
        self.thread = threading.Thread(target=self._watch, daemon=True)
        self.thread.start()
        return self

    def _watch(self):
        while self.running:
            self.peak = max(self.peak, self.proc.memory_info().rss)
            time.sleep(0.02)

    def __exit__(self, *exc):
        self.running = False
        self.thread.join()
        self.mb = (self.peak - self.base) / 1e6


# ---------------- OCR with cache ----------------
def cached_ocr(image):
    from core.resume_intelligence import local_ocr

    key = hashlib.sha256(image.convert("L").tobytes() + str(image.size).encode()).hexdigest()[:32]
    path = os.path.join(OCR_CACHE, key + ".json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    result = local_ocr(image)
    if result is None:
        raise RuntimeError("Local OCR (rapidocr_onnxruntime) is not installed — pip install rapidocr_onnxruntime")
    os.makedirs(OCR_CACHE, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f)
    return result


def score_text(truth, ocr):
    return {"cer": cer(truth, ocr["text"]), "wer": wer(truth, ocr["text"]), "word_recall": word_recall(truth, ocr["text"]),
            "ocr_score": ocr["score"], "ocr_confidence": ocr["confidence"]}


def _summary(rows, keys=("cer", "wer", "word_recall", "psnr", "ssim", "ink_f1", "seconds", "peak_mb")):
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None and np.isfinite(r[k])]
        if vals:
            out[k] = {"mean": float(np.mean(vals)), "median": float(np.median(vals)), "std": float(np.std(vals))}
    return out


def _bootstrap_ci(diffs, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(diffs)
    means = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def evaluate(steps=None, limit=None, log=print):
    from models.autoencoder import autoencoder as aem

    splits = dd.prepare_pages(log=None)
    pages = splits["test"][:limit] if limit else splits["test"]
    ae, diff = aem.ResumeDenoiser(version="v1"), ResumeDiffusionEnhancer()
    has_v2 = all(os.path.exists(p) for p in (aem.ENCODER_V2_PATH, aem.DECODER_V2_PATH))
    ae_v2 = aem.ResumeDenoiser(version="v2") if has_v2 else None
    enhancers = ["Autoencoder"] + (["Autoencoder v2"] if ae_v2 else []) + ["Diffusion"]
    presets = list(PRESETS)
    per_page = []
    for i, png in enumerate(pages):
        name, preset = os.path.basename(png)[:-4], presets[i % len(presets)]
        clean, truth = Image.open(png).convert("L"), dd.ground_truth(png)
        degraded = degrade_page(clean, preset, seed=1000 + i)
        outputs, timing = {"clean (ceiling)": clean, "degraded (no enhancement)": degraded}, {}
        with PeakMemory() as mem:
            t = time.time()
            outputs["Autoencoder"] = ae.denoise(degraded)[1]
            timing["Autoencoder"] = (time.time() - t, mem)
        if ae_v2:
            with PeakMemory() as mem:
                t = time.time()
                outputs["Autoencoder v2"] = ae_v2.denoise(degraded)[1]
                timing["Autoencoder v2"] = (time.time() - t, mem)
        with PeakMemory() as mem:
            t = time.time()
            _, outputs["Diffusion"], info = diff.enhance(degraded, steps=steps)
            timing["Diffusion"] = (time.time() - t, mem)
        row = {"page": name, "preset": preset, "truth_chars": len(normalise(truth)), "methods": {}}
        ocr = {m: cached_ocr(img) for m, img in outputs.items()}
        for m, img in outputs.items():
            r = score_text(truth, ocr[m])
            if not m.startswith("clean"):
                r.update(psnr=psnr(clean, img), ssim=ssim(clean, img), ink_f1=ink_f1(clean, img))
            if m in timing:
                r.update(seconds=timing[m][0], peak_mb=timing[m][1].mb)
            row["methods"][m] = r
        for m in enhancers:                              # guard: keep the degraded input if OCR reads it better
            use_original = ocr[m]["score"] < ocr["degraded (no enhancement)"]["score"]
            pick = dict(row["methods"]["degraded (no enhancement)" if use_original else m])
            pick["kept_original"] = use_original
            row["methods"][f"{m} + guard"] = pick
        per_page.append(row)
        if log:
            log(f"[{i + 1:2d}/{len(pages)}] {name:32s} {preset:18s} CER degraded "
                f"{row['methods']['degraded (no enhancement)']['cer']:.3f} · AE {row['methods']['Autoencoder']['cer']:.3f}"
                + (f" · AE v2 {row['methods']['Autoencoder v2']['cer']:.3f}" if ae_v2 else "") +
                f" · Diffusion {row['methods']['Diffusion']['cer']:.3f} · clean {row['methods']['clean (ceiling)']['cer']:.3f}"
                f" · {timing['Diffusion'][0]:.1f}s")

    methods = list(per_page[0]["methods"])
    overall = {m: _summary([p["methods"][m] for p in per_page]) for m in methods}
    by_preset = {pr: {m: _summary([p["methods"][m] for p in per_page if p["preset"] == pr], ("cer", "wer", "word_recall",
                                                                                            "psnr", "ssim", "ink_f1"))
                      for m in methods} for pr in presets}
    for m in enhancers:
        overall[f"{m} + guard"]["kept_original"] = int(sum(p["methods"][f"{m} + guard"]["kept_original"] for p in per_page))
    failures = {m: [{"page": p["page"], "preset": p["preset"], "cer_degraded": p["methods"]["degraded (no enhancement)"]["cer"],
                     "cer_enhanced": p["methods"][m]["cer"]}
                    for p in per_page if p["methods"][m]["cer"] > p["methods"]["degraded (no enhancement)"]["cer"] + 0.02]
                for m in enhancers + [f"{m} + guard" for m in enhancers]}

    def paired(a, b):
        d = [p["methods"][a]["cer"] - p["methods"][b]["cer"] for p in per_page]
        return {"mean_cer_difference": float(np.mean(d)), "ci95": _bootstrap_ci(d),
                "pages_first_better": int(sum(x < -0.005 for x in d)), "pages_second_better": int(sum(x > 0.005 for x in d)),
                "pages": len(d)}

    c = paired("Diffusion", "Autoencoder")
    comparison = {"mean_cer_difference_diffusion_minus_ae": c["mean_cer_difference"], "ci95": c["ci95"],
                  "pages_diffusion_better": c["pages_first_better"], "pages_ae_better": c["pages_second_better"],
                  "pages": c["pages"]}
    better = comparison["ci95"][1] < 0
    v2_report = None
    if ae_v2:
        v2_report = {"v2_minus_v1": paired("Autoencoder v2", "Autoencoder"),
                     "v2_minus_degraded": paired("Autoencoder v2", "degraded (no enhancement)")}
        v2_report["use_by_default"] = v2_report["v2_minus_v1"]["ci95"][1] < 0      # clearly better than v1
        cfg = {}
        if os.path.exists(aem.CONFIG_V2_PATH):
            with open(aem.CONFIG_V2_PATH) as f:
                cfg = json.load(f)
        cfg.update({"use_by_default": v2_report["use_by_default"], "evaluation": {
            "test_pages": len(per_page), "mean_cer": {m: overall[m]["cer"]["mean"] for m in
                                                     ("degraded (no enhancement)", "Autoencoder", "Autoencoder v2")},
            **v2_report, "evaluated_at": datetime.now().isoformat(timespec="seconds")}})
        with open(aem.CONFIG_V2_PATH, "w") as f:
            json.dump(cfg, f, indent=2)
    metrics = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "test_pages": len(per_page), "presets": PRESETS, "ocr_engine": "RapidOCR (rapidocr_onnxruntime, local)",
        "ground_truth": "PDF text layer of each Kaggle resume (first page)",
        "diffusion": {"steps": steps or diff.steps, "params": diff.num_params, "device": diff.device.type},
        "autoencoder_params": ae.num_params,
        "overall": overall, "by_preset": by_preset, "failures": failures,
        "diffusion_vs_autoencoder": comparison,
        "autoencoder_v2": v2_report,
        "verdict": ("Diffusion has significantly lower CER than the Autoencoder on this test set" if better else
                    "No evidence that Diffusion beats the Autoencoder on CER — keep the Autoencoder as default"),
        "per_page": per_page,
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    if log:
        log(f"metrics.json written · {metrics['verdict']}")
    return metrics


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    evaluate()
