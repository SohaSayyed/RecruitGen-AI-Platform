"""
Diffusion data — clean resume pages from the Kaggle Resume Dataset PDFs + paired degradations
--------------------------------------------------------------------------------------------
Clean images: first page of each PDF in data/data/<CATEGORY>/<ID>.pdf, rendered in grayscale at 150 dpi
(A4 -> 1240 x 1754 px, the scale the Autoencoder was trained on). Ground-truth text for OCR metrics: the PDF text layer.

Splits are made per category and per PDF, so no page is shared between train / validation / test:
    data/diffusion/<split>/<category>_<id>.png + .txt
"""

import glob
import io
import json
import os

import numpy as np
from PIL import Image

from models.diffusion.degradations import _field_params, degrade, random_spec
from models.gan.dataset import KAGGLE_DATASET, PROJECT_DIR, find_resume_csv

DATA_DIR = os.path.join(PROJECT_DIR, "data", "diffusion")
DPI = 150
SPLITS = {"train": 10, "val": 1, "test": 1}      # pages per category (24 categories)


def find_pdf_root(path=None):
    """Folder containing <CATEGORY>/<ID>.pdf (next to Resume/Resume.csv in the Kaggle download)."""
    csv = find_resume_csv(path)
    candidates = []
    if csv:
        base = os.path.dirname(os.path.dirname(csv))
        candidates += [os.path.join(base, "data", "data"), os.path.join(base, "data")]
    if path:
        candidates += [path, os.path.join(path, "data", "data")]
    for c in candidates:
        if os.path.isdir(c) and glob.glob(os.path.join(c, "*", "*.pdf")):
            return c
    return None


def render_page(pdf_path, dpi=DPI):
    """First page as a grayscale PIL image + its text layer."""
    import fitz

    with fitz.open(pdf_path) as doc:
        page = doc[0]
        pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
        return Image.open(io.BytesIO(pix.tobytes("png"))).convert("L"), page.get_text()


def prepare_pages(pdf_root=None, splits=SPLITS, seed=42, out_dir=DATA_DIR, log=print):
    """Render the split pages once (skips files that already exist). Returns {split: [png paths]}."""
    pdf_root = pdf_root or find_pdf_root()
    if pdf_root is None:
        raise FileNotFoundError(f"Kaggle resume PDFs not found — download {KAGGLE_DATASET} (see GAN_Training.ipynb).")
    rng = np.random.default_rng(seed)
    out = {s: [] for s in splits}
    for category in sorted(os.listdir(pdf_root)):
        pdfs = sorted(glob.glob(os.path.join(pdf_root, category, "*.pdf")))
        rng.shuffle(pdfs)
        start = 0
        for split, n in splits.items():
            for pdf in pdfs[start:start + n]:
                name = f"{category}_{os.path.splitext(os.path.basename(pdf))[0]}"
                png = os.path.join(out_dir, split, name + ".png")
                if not os.path.exists(png):
                    os.makedirs(os.path.dirname(png), exist_ok=True)
                    image, text = render_page(pdf)
                    image.save(png)
                    with open(png[:-4] + ".txt", "w", encoding="utf-8") as f:
                        f.write(text)
                out[split].append(png)
            start += n
    with open(os.path.join(out_dir, "splits.json"), "w") as f:
        json.dump({s: [os.path.basename(p) for p in v] for s, v in out.items()}, f, indent=1)
    if log:
        log("Pages: " + ", ".join(f"{s} {len(v)}" for s, v in out.items()) + f" -> {out_dir}")
    return out


def load_page(png):
    return np.asarray(Image.open(png).convert("L"), np.float32) / 255.0


def ground_truth(png):
    with open(png[:-4] + ".txt", encoding="utf-8") as f:
        return f.read()


class PatchSampler:
    """Random (degraded, clean) training patches. Crops are biased towards text; degradations are drawn per sample
    with page-consistent low-frequency fields."""

    def __init__(self, pages, patch=96, margin=8, text_bias=0.85, seed=0):
        self.pages = pages
        self.patch, self.margin, self.text_bias = patch, margin, text_bias
        self.rng = np.random.default_rng(seed)

    def sample(self, n):
        P, m = self.patch, self.margin
        size = P + 2 * m
        clean_b, deg_b = np.empty((n, 1, P, P), np.float32), np.empty((n, 1, P, P), np.float32)
        for i in range(n):
            page = self.pages[self.rng.integers(len(self.pages))]
            H, W = page.shape
            for _ in range(10):
                y, x = self.rng.integers(0, H - size), self.rng.integers(0, W - size)
                crop = page[y:y + size, x:x + size]
                if (crop < 0.5).mean() > 0.01 or self.rng.random() > self.text_bias:
                    break
            spec = random_spec(self.rng)
            params = _field_params(self.rng, spec, (H, W))
            deg = degrade(crop, spec, self.rng, origin=(y, x), page_shape=(H, W), page_params=params)
            clean_b[i, 0] = crop[m:m + P, m:m + P]
            deg_b[i, 0] = deg[m:m + P, m:m + P]
        return deg_b * 2 - 1, clean_b * 2 - 1          # model space [-1, 1]
