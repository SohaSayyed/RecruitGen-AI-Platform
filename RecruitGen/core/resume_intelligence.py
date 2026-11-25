"""
Resume reading (every common file format) and AI-generated resume insights.

Supported uploads: PDF (text or scanned), DOCX, DOC, TXT, MD, RTF, ODT, HTML,
PNG, JPG, JPEG, WEBP, BMP, TIFF.
"""

import io
import os
import re
import json
import time
import html
import base64
import zipfile

import requests
from PIL import Image

DOCUMENT_TYPES = ["pdf", "docx", "doc", "txt", "md", "rtf", "odt", "html", "htm"]
IMAGE_TYPES = ["png", "jpg", "jpeg", "webp", "bmp", "tiff", "tif"]
SUPPORTED_TYPES = DOCUMENT_TYPES + IMAGE_TYPES

MAX_PAGES = 3


# ============================================================
# Configuration (read from .streamlit/secrets.toml or env vars)
# ============================================================

def _settings():
    cfg = {}
    try:
        import streamlit as st
        if "ai" in st.secrets:
            cfg = dict(st.secrets["ai"])
    except Exception:
        pass
    keys = [cfg.get("api_key") or os.getenv("AI_API_KEY", ""),
            cfg.get("api_key_backup") or os.getenv("AI_API_KEY_BACKUP", "")]
    keys = [k for i, k in enumerate(keys) if k and k not in keys[:i]]
    return {
        "api_key": keys[0] if keys else "",
        "api_keys": keys,                      # primary first, backup used only if the primary fails
        "base_url": (cfg.get("base_url") or os.getenv("AI_BASE_URL", "")).rstrip("/"),
        "model": cfg.get("model") or os.getenv("AI_MODEL", ""),
        "vision_model": cfg.get("vision_model") or os.getenv("AI_VISION_MODEL", ""),
    }


def ai_available():
    s = _settings()
    return bool(s["api_keys"] and s["base_url"] and s["model"])


def _clean(text):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    return text.strip()


def _chat(messages, model=None, max_tokens=1800, temperature=0.3, json_mode=False, retries=3):
    """Send a chat request with the primary key, then the backup key if the primary fails.
    Returns the reply text, or None if the service is unavailable."""
    s = _settings()
    if not (s["api_keys"] and s["base_url"]):
        return None
    payload = {
        "model": model or s["model"],
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    for key in s["api_keys"]:
        for attempt in range(retries):
            try:
                r = requests.post(
                    f"{s['base_url']}/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json=payload,
                    timeout=90,
                )
                if r.status_code == 200:
                    return _clean(r.json()["choices"][0]["message"].get("content", ""))
                if r.status_code == 400 and "response_format" in payload:
                    payload.pop("response_format", None)
                    continue
                if r.status_code in (401, 403):   # key rejected -> try the next key straight away
                    break
                if r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(2 * (attempt + 1))
                    continue
                return None
            except requests.RequestException:
                time.sleep(1.5 * (attempt + 1))
    return None


def _parse_json(text):
    if not text:
        return None
    try:
        return json.loads(text)
    except ValueError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except ValueError:
                return None
    return None


# ============================================================
# AI output formatting — Markdown only, never raw HTML
# ============================================================
# Models sometimes answer with HTML tags (<br>, <b>, <p>, <li>…), most often inside Markdown
# tables. Streamlit shows those tags as literal text, so they are converted to Markdown here.

_BR = re.compile(r"\s*<\s*br\s*/?\s*>\s*", re.I)
_LI = re.compile(r"<\s*li\b[^<>]*>", re.I)
_ITEM_BREAK = re.compile(r"<\s*/?\s*(?:p|div|ul|ol)\b[^<>]*>|<\s*/\s*li\s*>", re.I)
_LINK = re.compile(r"""<\s*a\b[^>]*?href\s*=\s*["']([^"']+)["'][^>]*>(.*?)<\s*/\s*a\s*>""", re.I | re.S)
_PAIRS = [
    (re.compile(r"<\s*(b|strong)\b[^<>]*>(.*?)<\s*/\s*\1\s*>", re.I | re.S), "**"),
    (re.compile(r"<\s*(i|em)\b[^<>]*>(.*?)<\s*/\s*\1\s*>", re.I | re.S), "*"),
    (re.compile(r"<\s*(code)\b[^<>]*>(.*?)<\s*/\s*\1\s*>", re.I | re.S), "`"),
]
_HEADING_OPEN = re.compile(r"<\s*h([1-6])\b[^<>]*>", re.I)
_HEADING_CLOSE = re.compile(r"<\s*/\s*h[1-6]\s*>", re.I)
_HR = re.compile(r"<\s*hr\b[^<>]*>", re.I)
_BLOCK_END = re.compile(r"<\s*/\s*(?:p|div|ul|ol|blockquote|section|article|table)\s*>", re.I)
_LEFTOVER = re.compile(
    r"<\s*/?\s*(?:a|b|strong|i|em|code|p|div|span|u|ul|ol|li|font|small|mark|sup|sub|center|blockquote|section|"
    r"article|header|footer|table|thead|tbody|tfoot|tr|td|th|del|s|strike|ins|abbr|label|pre|h[1-6]|hr|br)\b[^<>]*>",
    re.I,
)
_BULLET = re.compile(r"^(?:[•·▪◦►]\s*|[*\-–—]\s+)")
_NUMBERED = re.compile(r"^\d+[.)]\s")
_CODE_SPAN = re.compile(r"`[^`]*`")
_TABLE_SPLIT = re.compile(r"(?<!\\)\|")


def _protect_code(text):
    spans = []

    def keep(m):
        spans.append(m.group(0))
        return f"\x00{len(spans) - 1}\x00"

    return _CODE_SPAN.sub(keep, text), spans


def _restore_code(text, spans):
    return re.sub(r"\x00(\d+)\x00", lambda m: spans[int(m.group(1))], text)


def _inline_markup(text):
    """HTML inline tags -> Markdown (links, bold, italic, code); any other known tag is dropped."""
    text = _LINK.sub(lambda m: f"[{m.group(2).strip()}]({m.group(1)})", text)
    for pattern, md in _PAIRS:
        text = pattern.sub(lambda m, md=md: f"{md}{m.group(2).strip()}{md}" if m.group(2).strip() else "", text)
    return _LEFTOVER.sub("", text)


def _one_line(text):
    """Flatten text that must stay on one line (a table cell or a short value):
    '<br>'-separated items become '• item • item' (or 'a; b' when they are not bullets)."""
    text = _LI.sub("<br>• ", text)
    text = _ITEM_BREAK.sub("<br>", text)
    items = [_inline_markup(i).strip() for i in _BR.split(text)]
    items = [re.sub(r"[ \t]{2,}", " ", i) for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if any(_NUMBERED.match(i) for i in items):
        return " ".join(items)
    if any(_BULLET.match(i) for i in items):
        return "• " + " • ".join(_BULLET.sub("", i) for i in items)
    joined = items[0]
    for item in items[1:]:
        joined += (" " if joined.endswith((".", ";", ":", "!", "?", ",")) else "; ") + item
    return joined


def _convert_line(line):
    if "<" not in line:
        return line
    line, spans = _protect_code(line)
    if line.strip().startswith("|") or len(_TABLE_SPLIT.findall(line)) >= 2:
        cells = _TABLE_SPLIT.split(line)
        line = "|".join(f" {_one_line(c)} " if "<" in c else c for c in cells)
    else:
        line = _BR.sub("  \n", line)
        line = _HR.sub("\n\n---\n\n", line)
        line = _HEADING_OPEN.sub(lambda m: "\n\n" + "#" * int(m.group(1)) + " ", line)
        line = _HEADING_CLOSE.sub("\n\n", line)
        line = _LI.sub("\n- ", line)
        line = _BLOCK_END.sub("\n\n", line)
        line = _inline_markup(line)
    return _restore_code(line, spans)


def format_ai_markdown(text):
    """Clean an AI chat reply so it renders as proper Markdown (no literal <br>, <b>, <p> …).
    Fenced code blocks and inline `code` are left untouched."""
    if not text or "<" not in text:
        return text
    out, in_fence = [], False
    for line in text.split("\n"):
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
            out.append(line)
        elif in_fence:
            out.append(line)
        else:
            out.append(_convert_line(line))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def format_ai_text(text):
    """Clean a short AI value (JSON field) to plain single-line-per-paragraph text."""
    if not text or ("<" not in text and "&" not in text):
        return text
    lines = []
    for line in text.split("\n"):
        line, spans = _protect_code(line)
        lines.append(_restore_code(_one_line(line), spans))
    return html.unescape("\n".join(lines)).strip()


def _clean_values(data):
    if isinstance(data, str):
        return format_ai_text(data)
    if isinstance(data, list):
        return [_clean_values(v) for v in data]
    if isinstance(data, dict):
        return {k: _clean_values(v) for k, v in data.items()}
    return data


# ============================================================
# File reading
# ============================================================

def _decode(data):
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def _read_pdf(data):
    import fitz  # PyMuPDF

    doc = fitz.open(stream=data, filetype="pdf")
    text = "\n".join(page.get_text() for page in doc)
    images = []
    if len(text.strip()) < 80:  # scanned PDF -> render pages as images
        for page in list(doc)[:MAX_PAGES]:
            pix = page.get_pixmap(dpi=150)
            images.append(Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB"))
    return text, images


def _read_docx(data):
    import docx

    document = docx.Document(io.BytesIO(data))
    parts = [p.text for p in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text for cell in row.cells))
    return "\n".join(parts)


def _read_doc(data):
    # Legacy Word binary: recover readable text runs
    raw = data.decode("latin-1", errors="ignore")
    runs = re.findall(r"[\x20-\x7E\r\n\t]{4,}", raw)
    return "\n".join(r.strip() for r in runs if len(r.strip()) > 3)


def _read_rtf(data):
    text = _decode(data)
    text = re.sub(r"\\par[d]?", "\n", text)
    text = re.sub(r"\\'[0-9a-f]{2}", "", text)
    text = re.sub(r"\\[a-z]+-?\d* ?", "", text)
    return re.sub(r"[{}]", "", text)


def _read_odt(data):
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        xml = z.read("content.xml").decode("utf-8", errors="ignore")
    xml = re.sub(r"</text:p>|<text:line-break/>", "\n", xml)
    return html.unescape(re.sub(r"<[^>]+>", " ", xml))


def _read_html(data):
    text = _decode(data)
    text = re.sub(r"(?is)<(script|style).*?</\1>", " ", text)
    text = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</li>|</h\d>", "\n", text)
    return html.unescape(re.sub(r"<[^>]+>", " ", text))


def read_resume(uploaded_file):
    """
    Read any supported resume file.
    Returns dict: {"text": str, "images": [PIL.Image], "kind": "document" | "image"}
    """
    name = uploaded_file.name
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    data = uploaded_file.getvalue()

    if ext in IMAGE_TYPES:
        image = Image.open(io.BytesIO(data))
        image.load()
        return {"text": "", "images": [image.convert("RGB")], "kind": "image"}

    images = []
    if ext == "pdf":
        text, images = _read_pdf(data)
    elif ext == "docx":
        text = _read_docx(data)
    elif ext == "doc":
        text = _read_doc(data)
    elif ext == "rtf":
        text = _read_rtf(data)
    elif ext == "odt":
        text = _read_odt(data)
    elif ext in ("html", "htm"):
        text = _read_html(data)
    else:
        text = _decode(data)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    return {"text": text, "images": images, "kind": "image" if images else "document"}


def image_to_text(image):
    """Read the text from a resume image (scanned page / photo)."""
    img = image.convert("RGB")
    img.thumbnail((1600, 1600))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    data_url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    s = _settings()
    reply = _chat(
        [{
            "role": "user",
            "content": [
                {"type": "text", "text": (
                    "Transcribe all text in this resume image exactly as written. "
                    "Keep the section order and line breaks. Return only the transcribed text."
                )},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }],
        model=s["vision_model"] or s["model"],
        max_tokens=2500,
        temperature=0.0,
    )
    if reply:
        return reply

    try:  # optional local OCR fallback
        import pytesseract
        return pytesseract.image_to_string(image)
    except Exception:
        pass
    result = local_ocr(image)  # offline fallback: RapidOCR (ONNX, CPU)
    return result["text"] if result else ""


# ============================================================
# Local OCR (RapidOCR) + image enhancement for OCR
# ============================================================

_LOCAL_OCR = None
_ENHANCERS = {}
_OCR_THREAD = __import__("threading").local()


def _ocr_engine(threads=None):
    """Shared engine (all cores) by default; with `threads`, one engine per worker thread limited to that many
    cores, so parallel bulk workers do not compete for the same cores."""
    global _LOCAL_OCR
    from rapidocr_onnxruntime import RapidOCR

    if not threads:
        if _LOCAL_OCR is None:
            _LOCAL_OCR = RapidOCR()
        return _LOCAL_OCR
    engines = getattr(_OCR_THREAD, "engines", None)
    if engines is None:
        engines = _OCR_THREAD.engines = {}
    if threads not in engines:
        engines[threads] = RapidOCR(intra_op_num_threads=int(threads), inter_op_num_threads=1)
    return engines[threads]


def local_ocr(image, threads=None):
    """Offline OCR. Returns {"text", "confidence", "score", "boxes"} or None if RapidOCR is not installed.
    score = sum(confidence x characters): a no-reference measure of how much text was read confidently."""
    try:
        import numpy as np
        result, _ = _ocr_engine(threads)(np.asarray(image.convert("RGB")))
    except Exception:
        return None
    boxes = [(min(p[1] for p in b), max(p[1] for p in b), min(p[0] for p in b), t, float(c))
             for b, t, c in (result or [])]
    lines = []
    for top, bottom, left, text, conf in sorted(boxes):        # group boxes into lines, then read left to right
        mid = (top + bottom) / 2
        if lines and abs(lines[-1]["mid"] - mid) < 0.5 * (bottom - top):
            lines[-1]["items"].append((left, text))
        else:
            lines.append({"mid": mid, "items": [(left, text)]})
    text = "\n".join(" ".join(t for _, t in sorted(line["items"])) for line in lines)
    confs = [b[4] for b in boxes]
    return {"text": text, "confidence": sum(confs) / len(confs) if confs else 0.0,
            "score": sum(c * len(t) for _, _, _, t, c in boxes), "boxes": len(boxes)}


def _enhancer(method):
    if method not in _ENHANCERS:
        if method == "autoencoder":
            from models.autoencoder.autoencoder import ResumeDenoiser
            _ENHANCERS[method] = ResumeDenoiser()
        elif method == "diffusion":
            from models.diffusion.diffusion import ResumeDiffusionEnhancer
            _ENHANCERS[method] = ResumeDiffusionEnhancer()
        else:
            raise ValueError(f"Unknown enhancement method: {method}")
    return _ENHANCERS[method]


DEGRADATIONS = ["noise", "blur", "shadows", "uneven lighting", "low contrast", "compression artifacts", "skew"]
SEVERITIES = ["none", "low", "medium", "high", "extreme"]


def _image_data_url(image, size=1024):
    img = image.convert("RGB")
    img.thumbnail((size, size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def local_image_diagnosis(image):
    """No-API fallback: simple image statistics -> the same structure as diagnose_resume_image()."""
    import numpy as np
    from PIL import ImageFilter

    gray = image.convert("L")
    gray.thumbnail((1400, 1400))
    a = np.asarray(gray, np.float32) / 255.0
    hp = a[1:, 1:] - 0.5 * (a[:-1, 1:] + a[1:, :-1])
    noise = float(np.median(np.abs(hp - np.median(hp))) * 1.4826 / 1.2247)
    lap = a[1:-1, 1:-1] * 4 - a[:-2, 1:-1] - a[2:, 1:-1] - a[1:-1, :-2] - a[1:-1, 2:]
    sharpness = float(lap.var())
    contrast = float(np.percentile(a, 99) - np.percentile(a, 1))
    background = np.asarray(gray.filter(ImageFilter.MaxFilter(15)).filter(ImageFilter.GaussianBlur(25)), np.float32) / 255
    lighting = float(np.percentile(background, 95) - np.percentile(background, 5))
    found = [d for d, hit in (("noise", noise > 0.02), ("blur", sharpness < 0.004), ("low contrast", contrast < 0.6),
                              ("uneven lighting", lighting > 0.15)) if hit]
    level = min(4, len(found) + (noise > 0.05) + (sharpness < 0.001))
    return {"degradations": found, "severity": SEVERITIES[level], "text_legible": level < 4,
            "recommended_method": "diffusion" if found else "none", "recommended_steps": 4 + 2 * (level >= 3),
            "notes": f"noise σ {noise:.3f} · sharpness {sharpness:.4f} · contrast {contrast:.2f} · lighting spread "
                     f"{lighting:.2f}", "source": "local analysis"}


def diagnose_resume_image(image):
    """Step 1 of the smart pipeline: the AI vision model inspects the image and recommends enhancement settings.
    It never edits the image. Returns a validated dict, or None if the AI service is unavailable / answers badly."""
    s = _settings()
    reply = _chat(
        [{"role": "user", "content": [
            {"type": "text", "text": (
                "You assess scanned or photographed resume images before local image enhancement and OCR. "
                "Return ONLY a JSON object with keys: "
                f'"degradations": list using only {DEGRADATIONS}; "severity": one of {SEVERITIES}; '
                '"text_legible": true/false; "recommended_method": "diffusion" (noise, blur, shadows, lighting, '
                'contrast problems), "autoencoder" (light noise only) or "none" (already clean); '
                '"recommended_steps": integer 1-10 (more for heavier degradation); "notes": one short sentence.')},
            {"type": "image_url", "image_url": {"url": _image_data_url(image)}},
        ]}],
        model=s["vision_model"] or s["model"], max_tokens=400, temperature=0.0, json_mode=True, retries=2,
    )
    data = _parse_json(reply)
    if not isinstance(data, dict):
        return None
    degradations = [d for d in (data.get("degradations") or []) if isinstance(d, str) and d.lower() in DEGRADATIONS]
    method = str(data.get("recommended_method", "")).lower()
    try:
        steps = int(data.get("recommended_steps", 4))
    except (TypeError, ValueError):
        steps = 4
    return {
        "degradations": [d.lower() for d in degradations],
        "severity": str(data.get("severity", "")).lower() if str(data.get("severity", "")).lower() in SEVERITIES
        else "medium",
        "text_legible": bool(data.get("text_legible", True)),
        "recommended_method": method if method in ("diffusion", "autoencoder", "none") else "diffusion",
        "recommended_steps": max(1, min(10, steps)),
        "notes": format_ai_text(str(data.get("notes", "")))[:300],
        "source": "AI analysis",
    }


def smart_enhance(image, diffusion=None, autoencoder=None, use_ai=True, guard=False, progress=None):
    """API first, then the local model: (1) AI image analysis (fallback: local analysis) picks the method and the
    diffusion steps, (2) the local Diffusion (or Autoencoder) model enhances the image. Returns enhance_resume_image's
    dict plus "diagnosis"."""
    diagnosis = diagnose_resume_image(image) if use_ai and ai_available() else None
    diagnosis = diagnosis or local_image_diagnosis(image)
    method = diagnosis["recommended_method"]
    if method == "diffusion" and diffusion is None:
        method = "autoencoder"
    if method == "none":                      # already clean: still return the untouched original
        out = enhance_resume_image(image, "none")
    else:
        out = enhance_resume_image(image, method, enhancer=diffusion if method == "diffusion" else autoencoder,
                                   guard=guard, progress=progress, steps=diagnosis["recommended_steps"])
    out["diagnosis"] = diagnosis
    return out


def enhance_resume_image(image, method="diffusion", enhancer=None, guard=False, progress=None, steps=None,
                         ocr_threads=None):
    """Enhance a resume image for OCR with the local Autoencoder or Diffusion model (never via an API).

    guard=True runs local OCR on the original and the enhanced image and keeps the ORIGINAL whenever the enhanced
    version is read less confidently. The original image is always returned untouched.
    Returns {"original", "enhanced", "image" (the one to OCR), "method", "used", "seconds", "ocr_original",
    "ocr_enhanced", "error"}."""
    import time
    from PIL import Image

    if not isinstance(image, Image.Image):
        raise TypeError("enhance_resume_image expects a PIL image")
    out = {"original": image, "enhanced": None, "image": image, "method": method, "used": "original",
           "seconds": 0.0, "ocr_original": None, "ocr_enhanced": None, "error": None}
    if method in (None, "none", "original"):
        return out
    start = time.time()
    try:
        model = enhancer or _enhancer(method)
        if method == "diffusion":
            _, enhanced, _ = model.enhance(image, steps=steps, progress=progress)
        else:
            _, enhanced = model.denoise(image, progress=progress)
    except Exception as exc:                      # enhancement failure never blocks reading the resume
        out["error"] = str(exc)
        return out
    out.update(enhanced=enhanced, image=enhanced, used="enhanced", seconds=time.time() - start)
    from models.autoencoder.autoencoder import CONTENT_MIN_RATIO, content_ratio
    out["content_ratio"] = content_ratio(image, enhanced)
    if out["content_ratio"] < CONTENT_MIN_RATIO:     # the model erased most of the text: never lose resume content
        out.update(image=image, used="original", error="enhancement removed most of the page content")
        return out
    if guard:
        out["ocr_original"] = local_ocr(image, ocr_threads)
        out["ocr_enhanced"] = local_ocr(enhanced, ocr_threads)
        if out["ocr_original"] and out["ocr_enhanced"] and out["ocr_enhanced"]["score"] < out["ocr_original"]["score"]:
            out.update(image=image, used="original")
    return out


# ============================================================
# AI insights
# ============================================================

SYSTEM_PROMPT = (
    "You are RecruitGen, an expert technical recruiter and career coach. "
    "You give precise, honest, constructive, evidence-based feedback on resumes."
)
NO_HTML = " Never use HTML tags (no <br>, <b>, <p>, <ul>, <li>)."
JSON_STYLE = " Use plain text in every JSON string value (no HTML, no Markdown)."
CHAT_STYLE = (
    " Format answers with Markdown only." + NO_HTML +
    " Inside Markdown tables keep every cell on a single line; separate several items in one cell with ' • '."
)


def resume_insights(resume_text, predicted_role=None, features=None):
    context = ""
    if predicted_role:
        context += f"\nScreening model predicted role: {predicted_role}."
    if features:
        context += f"\nExtracted features: {json.dumps(features)}."
    prompt = f"""Analyse this resume.{context}

Return ONLY a JSON object with these keys:
"candidate_name": string (or "Candidate" if not found),
"headline": one-line professional headline,
"summary": 3-4 sentence professional summary,
"overall_score": integer 0-100 (resume quality and hireability),
"ats_score": integer 0-100 (ATS friendliness),
"key_skills": list of up to 12 skills,
"strengths": list of 4-6 strengths,
"gaps": list of 3-5 weaknesses or missing items,
"suggestions": list of 4-6 concrete improvement suggestions,
"suitable_roles": list of 3 job titles that fit,
"interview_questions": list of 6 tailored interview questions.

RESUME:
{resume_text[:12000]}"""
    reply = _chat(
        [{"role": "system", "content": SYSTEM_PROMPT + JSON_STYLE + NO_HTML}, {"role": "user", "content": prompt}],
        max_tokens=3000, json_mode=True,
    )
    return _clean_values(_parse_json(reply))


def job_match(resume_text, job_description):
    prompt = f"""Compare the resume with the job description.

Return ONLY a JSON object with keys:
"match_score": integer 0-100,
"verdict": one of "Strong Match", "Good Match", "Partial Match", "Weak Match",
"matched_skills": list of skills present in both,
"missing_skills": list of required skills missing from the resume,
"summary": 3-4 sentence explanation for a recruiter,
"recommendations": list of 3-5 things the candidate should add or improve.

JOB DESCRIPTION:
{job_description[:6000]}

RESUME:
{resume_text[:10000]}"""
    reply = _chat(
        [{"role": "system", "content": SYSTEM_PROMPT + JSON_STYLE + NO_HTML}, {"role": "user", "content": prompt}],
        max_tokens=2500, json_mode=True,
    )
    return _clean_values(_parse_json(reply))


def interview_questions(resume_text, role, level="Mid", count=8):
    prompt = f"""Create {count} interview questions for a {level}-level {role} candidate, based on this resume.
Mix technical, project-based and behavioural questions.
Return ONLY a JSON object: {{"questions": [{{"question": str, "type": "Technical|Project|Behavioural", "what_to_look_for": str}}]}}

RESUME:
{resume_text[:10000]}"""
    reply = _chat(
        [{"role": "system", "content": SYSTEM_PROMPT + JSON_STYLE + NO_HTML}, {"role": "user", "content": prompt}],
        max_tokens=2500, json_mode=True,
    )
    data = _clean_values(_parse_json(reply))
    return data.get("questions") if isinstance(data, dict) else None


def synthetic_profile_summary(profile, role):
    """Short recruiter-style description of one GAN-generated profile. The profile is synthetic: the model
    must describe only the given numbers and never invent a name, employer or real person."""
    prompt = f"""This is a SYNTHETIC candidate profile generated by a GAN for dataset augmentation and testing.
It is not a real person. Job role (condition): {role}.
Feature values: {json.dumps(profile)}
Scales: experience in years; projects and certifications are counts; programming / data analytics / ML-AI /
cloud / communication are 0-100 keyword-evidence scores; education 1 Diploma, 2 Bachelor's, 3 Master's, 5 PhD.

Return ONLY a JSON object with keys:
"summary": 2-3 sentences describing this synthetic profile using only the numbers above,
"strengths": list of 2-3 short items,
"gaps": list of 2-3 short items,
"realism_note": one sentence on whether this combination looks plausible for the role.
Do not invent names, employers, schools or any facts that are not in the numbers."""
    reply = _chat(
        [{"role": "system", "content": SYSTEM_PROMPT + JSON_STYLE + NO_HTML}, {"role": "user", "content": prompt}],
        max_tokens=700, json_mode=True,
    )
    return _clean_values(_parse_json(reply))


def explain_gan_results(summary):
    """Plain-language explanation of GAN evaluation results that were computed in Python."""
    prompt = f"""Explain these evaluation results of a GAN used to generate synthetic candidate profiles
(9 numerical features + job role) for dataset augmentation. All numbers were computed in Python on a real
held-out test split. Use ONLY these numbers — never invent or change a value.

RESULTS (JSON):
{json.dumps(summary)[:9000]}

Write 2 short paragraphs and a bullet list for a recruiter / ML reviewer:
what the quality and diversity numbers mean compared with the real reference, whether augmentation helped the
downstream classifier, how the GAN compares with the CVAE baseline and random oversampling, and the main limitations."""
    reply = _chat(
        [{"role": "system", "content": SYSTEM_PROMPT + CHAT_STYLE}, {"role": "user", "content": prompt}],
        max_tokens=1200, temperature=0.2,
    )
    return format_ai_markdown(reply)


def enhancement_assistant(question, context):
    """Optional guidance on the image-enhancement settings / results. Never used to modify an image."""
    prompt = f"""You help a recruiter use RecruitGen's resume image enhancement (an Autoencoder baseline and an
optional local diffusion model; both run on the user's machine). Answer the question using the context below.
Only quote numbers that appear in the context; never invent results.

CONTEXT (JSON):
{json.dumps(context)[:7000]}

QUESTION: {question[:1500]}"""
    reply = _chat(
        [{"role": "system", "content": SYSTEM_PROMPT + CHAT_STYLE}, {"role": "user", "content": prompt}],
        max_tokens=900, temperature=0.2,
    )
    return format_ai_markdown(reply)


def assistant_reply(history, resume_text=""):
    system = SYSTEM_PROMPT + " Answer recruitment, resume and career questions clearly and concisely." + CHAT_STYLE
    if resume_text:
        system += f"\n\nThe user is currently reviewing this resume:\n{resume_text[:8000]}"
    reply = _chat([{"role": "system", "content": system}] + history[-12:], max_tokens=1500, temperature=0.5)
    return format_ai_markdown(reply)
