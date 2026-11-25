"""
Bulk resume processing — ZIP of up to 100 PDF / PNG / JPG / JPEG resumes → background job → CSV / Excel
-------------------------------------------------------------------------------------------------------
Reuses the existing pipeline in core.resume_intelligence:
    enhance_resume_image()  Diffusion / Autoencoder enhancement + content safeguard + optional OCR guard
    local_ocr()             RapidOCR (offline)            image_to_text()  AI vision OCR with its own fallbacks

Safety
    * ZIP entries are read in memory and stored under generated names (input/007.pdf) — archive paths are never used
      as file-system paths, so path traversal is impossible; unsafe names are still reported as Failed rows.
    * limits: files, per-file size, total uncompressed size, compression ratio (zip bombs), encrypted entries,
      magic-byte check of the declared type, PDF page cap.
    * every batch lives in data/bulk/<user>/<batch>/ and is only visible to its owner; batches expire after
      RETENTION_HOURS and can be deleted from the UI. Logs never contain resume text, file names or paths.

Output: exactly one row per uploaded resume (failed and unsupported files included), UTF-8, formula-injection safe.
"""

import csv
import io
import json
import os
import queue
import re
import shutil
import threading
import time
import unicodedata
import uuid
import zipfile
from datetime import datetime

LIMITS = {
    "max_files": 100,
    "max_file_mb": 20,
    "max_zip_mb": 200,
    "max_total_mb": 1024,          # total uncompressed
    "max_ratio": 200,              # uncompressed / compressed per entry
    "max_pdf_pages": 10,
    "pdf_dpi": 200,
    "min_text_chars": 150,         # below this the extraction is a Partial Success
    "min_confidence": 0.60,        # mean local-OCR confidence below this -> Partial Success
}
RETENTION_HOURS = 24
SUPPORTED = {"pdf": "pdf", "png": "image", "jpg": "image", "jpeg": "image"}
MAGIC = {"pdf": [b"%PDF"], "png": [b"\x89PNG\r\n\x1a\n"], "jpg": [b"\xff\xd8\xff"], "jpeg": [b"\xff\xd8\xff"]}
JUNK = ("__macosx/", ".ds_store", "thumbs.db", "desktop.ini")
TEXT_COLUMNS = ["resume_id", "filename", "status", "extracted_text", "error_message", "enhancement_method",
                "processing_time_seconds", "ocr_language", "pages_processed", "notes"]
STRUCTURED_COLUMNS = ["resume_id", "filename", "candidate_name", "email", "phone", "skills", "experience_years",
                      "education", "certifications", "previous_job_title", "previous_company"]

BULK_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "bulk")


# ============================================================
# ZIP inspection
# ============================================================

def _unsafe_name(name):
    n = name.replace("\\", "/")
    return n.startswith("/") or re.match(r"^[A-Za-z]:", n) is not None or any(p == ".." for p in n.split("/"))


def inspect_zip(data, limits=LIMITS):
    """Validate a ZIP and return (items, error). error is a message when the whole ZIP must be rejected.
    items: one dict per uploaded file (junk entries such as __MACOSX/ are skipped):
        {"index", "filename", "ext", "kind", "data" (bytes or None), "error" (None or message)}"""
    if len(data) > limits["max_zip_mb"] * 1024 * 1024:
        return [], f"The ZIP is larger than {limits['max_zip_mb']} MB."
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
        infos = zf.infolist()
    except (zipfile.BadZipFile, OSError, ValueError):
        return [], "This file is not a valid ZIP archive (corrupted or another format)."
    entries = [i for i in infos if not i.is_dir() and not any(j in i.filename.lower() for j in JUNK)
               and not os.path.basename(i.filename).startswith("._")]
    if not entries:
        return [], "The ZIP does not contain any files."
    resumes = [i for i in entries if os.path.splitext(i.filename)[1].lower().lstrip(".") in SUPPORTED]
    if len(resumes) > limits["max_files"]:
        return [], f"The ZIP contains {len(resumes)} resumes — the maximum is {limits['max_files']} per batch."
    if sum(i.file_size for i in entries) > limits["max_total_mb"] * 1024 * 1024:
        return [], f"The ZIP expands to more than {limits['max_total_mb']} MB."

    items, seen = [], {}
    for info in entries:
        raw_name = info.filename
        base = os.path.basename(raw_name.replace("\\", "/")) or raw_name
        seen[base.lower()] = seen.get(base.lower(), 0) + 1
        display = base if seen[base.lower()] == 1 else f"{base} (duplicate {seen[base.lower()]})"
        ext = os.path.splitext(base)[1].lower().lstrip(".")
        item = {"index": len(items) + 1, "filename": display, "ext": ext, "kind": SUPPORTED.get(ext), "data": None,
                "error": None}
        if seen[base.lower()] > 1:
            item["note"] = "duplicate filename in the ZIP — processed separately"
        if _unsafe_name(raw_name):
            item["error"] = "Rejected: unsafe path inside the ZIP"
        elif ext not in SUPPORTED:
            item["error"] = f"Unsupported format (.{ext or '?'}) — only PDF, PNG, JPG and JPEG are processed"
        elif info.flag_bits & 0x1:
            item["error"] = "Rejected: the file is encrypted"
        elif info.file_size > limits["max_file_mb"] * 1024 * 1024:
            item["error"] = f"Rejected: larger than {limits['max_file_mb']} MB"
        elif info.compress_size and info.file_size / max(info.compress_size, 1) > limits["max_ratio"]:
            item["error"] = "Rejected: suspicious compression ratio"
        else:
            try:
                with zf.open(info) as f:
                    content = f.read(limits["max_file_mb"] * 1024 * 1024 + 1)
                if len(content) > limits["max_file_mb"] * 1024 * 1024:
                    item["error"] = f"Rejected: larger than {limits['max_file_mb']} MB"
                elif not content:
                    item["error"] = "Corrupted or empty file"
                elif not any(content.startswith(m) for m in MAGIC[ext]):
                    item["error"] = f"Corrupted file or not a real .{ext} (content does not match the extension)"
                else:
                    item["data"] = content
            except (zipfile.BadZipFile, OSError, RuntimeError, NotImplementedError, EOFError):
                item["error"] = "Corrupted entry — could not be extracted"
        items.append(item)
    return items, None


# ============================================================
# Text analysis: language + structured fields
# ============================================================

_SCRIPTS = [("zh", [(0x4E00, 0x9FFF), (0x3400, 0x4DBF)]), ("ja", [(0x3040, 0x30FF)]), ("ko", [(0xAC00, 0xD7AF)]),
            ("hi", [(0x0900, 0x097F)]), ("ar", [(0x0600, 0x06FF)]), ("ru", [(0x0400, 0x04FF)]),
            ("ta", [(0x0B80, 0x0BFF)]), ("te", [(0x0C00, 0x0C7F)]), ("bn", [(0x0980, 0x09FF)]),
            ("el", [(0x0370, 0x03FF)]), ("he", [(0x0590, 0x05FF)]), ("th", [(0x0E00, 0x0E7F)])]
_STOPWORDS = {
    "en": {"the", "and", "of", "to", "in", "with", "for", "a", "on", "experience", "skills", "is", "as"},
    "es": {"de", "la", "el", "y", "en", "con", "para", "experiencia", "los", "las", "del"},
    "fr": {"de", "la", "le", "et", "en", "avec", "pour", "les", "des", "expérience", "du"},
    "de": {"und", "der", "die", "das", "mit", "für", "von", "erfahrung", "im", "zu", "den"},
    "pt": {"de", "e", "em", "com", "para", "experiência", "do", "da", "os", "as"},
    "it": {"di", "e", "il", "la", "con", "per", "esperienza", "del", "della", "in"},
}


def detect_language(text):
    """Script-based detection (+ stop words for Latin-script languages). Returns e.g. "en", "zh", "en+zh", "unknown"."""
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 20:
        return "unknown"
    counts = {}
    for c in letters:
        cp = ord(c)
        script = next((code for code, ranges in _SCRIPTS if any(lo <= cp <= hi for lo, hi in ranges)), "latin")
        counts[script] = counts.get(script, 0) + 1
    langs = [s for s, n in sorted(counts.items(), key=lambda kv: -kv[1]) if n / len(letters) >= 0.15]
    out = []
    for s in langs:
        if s == "latin":
            words = re.findall(r"[a-zà-ÿ]+", text.lower())
            scores = {lang: sum(w in sw for w in words) for lang, sw in _STOPWORDS.items()}
            best = max(scores, key=scores.get)
            out.append(best if scores[best] >= 3 else "latin")
        else:
            out.append(s)
    return "+".join(out) or "unknown"


EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?<![\w/])(\+?\(?\d[\d\s().-]{5,20}\d)(?![\w/])")
SECTION_ALIASES = {
    "summary": ("summary", "profile", "professional summary", "objective", "about me"),
    "skills": ("skills", "technical skills", "core skills", "key skills", "core competencies", "competencies"),
    "experience": ("experience", "work experience", "professional experience", "employment history", "work history"),
    "education": ("education", "academic background", "qualifications", "academic qualifications"),
    "certifications": ("certifications", "certificates", "certification", "licenses", "licenses & certifications"),
    "projects": ("projects", "key projects"),
    "other": ("languages", "awards", "achievements", "interests", "references", "publications", "volunteering"),
}
_HEADING = {alias: key for key, aliases in SECTION_ALIASES.items() for alias in aliases}
ROLE_RE = re.compile(r"^\s*(?P<title>[A-Za-z][^()\n|]{1,70}?)\s*(?:[-–—|]|\bat\b)\s*(?P<company>[A-Za-z0-9&][^()\n|]{1,70}?)"
                     r"\s*\(\s*(?P<dates>[^)]*\d{4}[^)]*)\)", re.I)
_BULLET = re.compile(r"^[\s\-–—•*·.●▪]+")
DEGREE_RE = re.compile(r"\b(ph\.?\s?d|doctorate|master(?:'s)?\s+of\s+[a-z ]{3,40}|m\.?\s?tech|m\.?\s?sc|mba|mca|"
                       r"bachelor(?:'s)?\s+of\s+[a-z ]{3,40}|b\.?\s?tech|b\.?\s?e\b|b\.?\s?sc|bca|bba|diploma\s+in\s+"
                       r"[a-z ]{3,40})", re.I)
TITLE_WORDS = {"resume", "curriculum", "vitae", "cv", "profile", "summary", "manager", "engineer", "developer",
               "analyst", "administrator", "specialist", "director", "consultant", "assistant", "intern", "officer",
               "executive", "associate", "designer", "accountant", "teacher", "chef", "technician", "coordinator",
               "representative", "supervisor", "advocate", "lead", "senior", "junior"}


class _Line(str):
    """A text line that remembers its position in the original text (to join only truly adjacent lines)."""

    def __new__(cls, value, index):
        obj = super().__new__(cls, value)
        obj.index = index
        return obj


def split_sections(text):
    """{section: [lines]} using the resume's own headings (SKILLS, EXPERIENCE, …); text before the first heading = 'head'."""
    sections, current = {"head": []}, "head"
    for i, line in enumerate(text.splitlines()):
        s = line.strip()
        key = _HEADING.get(s.rstrip(":").strip().lower()) if len(s) <= 40 else None
        if key:
            current = key
            sections.setdefault(key, [])
        elif s:
            sections.setdefault(current, []).append(_Line(s, i))
    return sections


def _join_wrapped(lines):
    """Join a line that a PDF wrapped mid-sentence to the line directly above it in the original text
    (next line starts lower-case, no bullet, previous line does not end a sentence)."""
    items, prev_index = [], None
    for line in lines:
        index = getattr(line, "index", None)
        if (items and index is not None and prev_index is not None and index == prev_index + 1
                and not _BULLET.match(line) and not items[-1].endswith((".", ":")) and line[:1].islower()):
            items[-1] += " " + line
        else:
            items.append(str(line))
        prev_index = index
    return items


def _phone(text, head_lines):
    """A phone number: in the contact block any 7-15 digit number, elsewhere only a full 10-15 digit number;
    filled only when exactly one distinct number is found."""
    found = []
    for scope, minimum in (("\n".join(head_lines[:6]), 7), (text, 10)):
        for m in PHONE_RE.finditer(scope):
            p = m.group(1).strip(" .-")
            digits = re.sub(r"\D", "", p)
            if minimum <= len(digits) <= 15 and not re.fullmatch(r"(19|20)\d{2}(19|20)\d{2}", digits):  # not "2019 2023"
                found.append((digits, p))
        if found:
            break
    return found[0][1] if found and len({d for d, _ in found}) == 1 else ""


def structured_fields(text):
    """Candidate fields, filled only when the resume states them clearly; everything else stays empty.
    Never invents values: every field is copied from the extracted text."""
    from models.transformer_v3 import feature_extractor as fx

    out = {k: "" for k in STRUCTURED_COLUMNS[2:]}
    sec = split_sections(text)
    head = sec.get("head", [])
    emails = sorted(set(EMAIL_RE.findall(text)))
    if len(emails) == 1:
        out["email"] = emails[0]
    out["phone"] = _phone(text, head)
    for line in head[:3]:
        words = line.split()
        if (2 <= len(words) <= 4 and all(re.fullmatch(r"[A-Z][a-zA-Z'.-]+", w) for w in words)
                and not any(w.lower().strip(".") in TITLE_WORDS for w in words) and not line.isupper()):
            out["candidate_name"] = line
            break

    # skills: the resume's own skills list, else known technical keywords found anywhere
    listed = []
    for item in _join_wrapped(sec.get("skills", [])):
        item = re.sub(r"^[A-Za-z ]{0,25}skills?\s*:\s*", "", _BULLET.sub("", item), flags=re.I)
        for s in re.split(r"[,;|•·]", item):
            s = s.strip(" .")
            if 1 <= len(s.split()) <= 5 and s.lower() not in [x.lower() for x in listed]:
                listed.append(s)
    if listed:
        out["skills"] = ", ".join(listed)
    else:
        low = text.lower()
        keywords = fx.PROGRAMMING_KEYWORDS + fx.DATA_ANALYTICS_KEYWORDS + fx.ML_AI_KEYWORDS + fx.CLOUD_KEYWORDS
        out["skills"] = ", ".join(sorted({k for k in keywords if len(k) > 3 and re.search(
            r"(?<![a-z])" + re.escape(k) + r"(?![a-z])", low)}))

    # education: the education section's lines, else degree patterns found anywhere
    edu = [_BULLET.sub("", l) for l in _join_wrapped(sec.get("education", []))][:2]
    if edu:
        out["education"] = "; ".join(edu)
    else:
        degrees = []
        for m in DEGREE_RE.finditer(text):
            d = re.sub(r"\s+", " ", m.group(0)).strip(" .")
            if d.lower() not in [x.lower() for x in degrees]:
                degrees.append(d)
        out["education"] = "; ".join(degrees[:3])

    certs = [_BULLET.sub("", l).strip() for l in _join_wrapped(sec.get("certifications", []))]
    out["certifications"] = "; ".join(c for c in certs if len(c) >= 2)

    # latest role: first "Title - Company (dates)" entry of the experience section
    for line in sec.get("experience", []):
        m = ROLE_RE.match(line)
        if m:
            out["previous_job_title"] = m.group("title").strip(" -–—|")
            out["previous_company"] = m.group("company").strip(" -–—|")
            break

    low = text.lower()
    m = re.search(r"(\d{1,2}(?:\.\d)?)\s*\+?\s*years?\s+(?:of\s+)?(?:professional\s+|work\s+|industry\s+|hands-on\s+)?"
                  r"(?:experience|work\b|in\b)", low)
    if m and float(m.group(1)) <= 50:
        out["experience_years"] = m.group(1)
    return out


# ============================================================
# Output: CSV / Excel (UTF-8, formula-injection safe)
# ============================================================

def clean_text(value):
    """Shared by the CSV and Excel exports so both contain identical text: removes control, invisible-format and
    private-use characters (e.g. PDF bullet glyphs), keeps every visible character, newlines and tabs."""
    s = value if isinstance(value, str) else str(value)
    return "".join(c for c in s if c in "\n\t" or unicodedata.category(c)[0] != "C")


def safe_cell(value):
    """Neutralise spreadsheet formulas (=, +, -, @, tab, CR at the start) and drop control characters."""
    if value is None:
        return ""
    s = clean_text(value)
    return "'" + s if s[:1] in ("=", "+", "-", "@", "\t", "\r") else s


def to_csv(rows, columns):
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns, quoting=csv.QUOTE_ALL, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for r in rows:
        writer.writerow({c: safe_cell(r.get(c, "")) for c in columns})
    return buf.getvalue().encode("utf-8-sig")          # BOM so Excel opens the UTF-8 file correctly


def to_excel(rows, structured, job=None):
    """Recruiter / management report: Dashboard · Candidate Profiles · Extracted Text · Errors & Warnings
    (see core/excel_report.py). Same rows and fields as the CSV exports."""
    from core.excel_report import build_report

    return build_report(rows, structured, job)


# ============================================================
# One resume
# ============================================================

class BatchCancelled(Exception):
    """Raised inside a worker when the user stops the batch."""


def _ocr(image, engine, threads=None):
    from core.resume_intelligence import image_to_text, local_ocr

    if engine == "ai_vision":
        text = image_to_text(image)          # existing engine: AI vision -> pytesseract -> local OCR
        return {"text": text or "", "confidence": None}
    r = local_ocr(image, threads)
    if r is None:
        raise RuntimeError("local OCR is not installed (pip install rapidocr_onnxruntime)")
    return {"text": r["text"], "confidence": r["confidence"], "score": r["score"]}


def _process_image(image, opts, get_model, notes, page_label, check=lambda: None):
    """Enhance (optional, with fallbacks) and OCR one page image. Returns (text, confidence, method_used).
    `check()` raises BatchCancelled when the user stops the batch (also polled between diffusion tile batches)."""
    from core.resume_intelligence import enhance_resume_image

    method = opts["enhancement"]
    threads = opts.get("ocr_threads")
    used = "None"
    target = image
    if method in ("diffusion", "autoencoder"):
        try:
            res = enhance_resume_image(image, method, enhancer=get_model(method), guard=opts["guard"],
                                       progress=lambda p: check(), ocr_threads=threads)
            check()                                  # stopped during enhancement -> do not continue with OCR
            if res["used"] == "enhanced":
                target, used = res["image"], "Diffusion" if method == "diffusion" else "Autoencoder"
            elif res["error"]:
                notes.append(f"{page_label}: enhancement not used ({res['error']}) — original image")
            else:
                notes.append(f"{page_label}: original kept (enhancement lowered OCR quality)")
            if opts["guard"] and opts["ocr_engine"] == "local" and res.get("ocr_enhanced") and res.get("ocr_original"):
                chosen = res["ocr_enhanced"] if res["used"] == "enhanced" else res["ocr_original"]
                return chosen["text"], chosen["confidence"], used          # already read by the guard
        except BatchCancelled:
            raise
        except Exception as exc:                                        # enhancement failure never stops the file
            notes.append(f"{page_label}: enhancement failed ({type(exc).__name__}) — original image")
    check()
    r = _ocr(target, opts["ocr_engine"], threads)
    return r["text"], r["confidence"], used


def process_resume(item, opts, get_model, stage=lambda s: None, cancelled=lambda: False):
    """Full pipeline for one uploaded file. Never raises (except BatchCancelled when the user stops the batch):
    errors become a Failed / Partial Success row."""
    def check():
        if cancelled():
            raise BatchCancelled()

    def stage_(s, _stage=stage):
        check()
        _stage(s)

    stage = stage_
    start = time.time()
    row = {"resume_id": item["resume_id"], "filename": item["filename"], "status": "Failed", "extracted_text": "",
           "error_message": "", "enhancement_method": "None", "processing_time_seconds": 0.0, "ocr_language": "",
           "pages_processed": "", "notes": item.get("note", "")}
    if item.get("error"):
        row["error_message"] = item["error"]
        return row
    notes, texts, confidences, methods, page_errors = [], [], [], set(), []
    try:
        from PIL import Image

        if item["kind"] == "pdf":
            import fitz

            stage("opening PDF")
            doc = fitz.open(stream=item["data"], filetype="pdf")
            if doc.needs_pass:
                raise ValueError("the PDF is password-protected")
            n_pages = len(doc)
            if n_pages == 0:
                raise ValueError("the PDF has no pages")
            limit = min(n_pages, opts["max_pages"])
            if n_pages > limit:
                notes.append(f"only the first {limit} of {n_pages} pages were processed")
            for p in range(limit):
                label = f"page {p + 1}"
                try:
                    page = doc[p]
                    text = page.get_text().strip()
                    if len(text) >= 30:                                   # real text layer: no OCR needed
                        texts.append(text)
                        continue
                    stage(f"{label}/{limit}: rendering scanned page")
                    pix = page.get_pixmap(dpi=LIMITS["pdf_dpi"], colorspace=fitz.csGRAY)
                    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")
                    stage(f"{label}/{limit}: enhancing + reading text")
                    t, conf, used = _process_image(img, opts, get_model, notes, label, check)
                    texts.append(t)
                    methods.add(used)
                    if conf is not None:
                        confidences.append(conf)
                except BatchCancelled:
                    raise
                except Exception as exc:
                    page_errors.append(f"{label}: {type(exc).__name__}: {str(exc)[:120]}")
            row["pages_processed"] = f"{limit - len(page_errors)}/{n_pages}"
        else:
            stage("opening image")
            img = Image.open(io.BytesIO(item["data"]))
            img.load()
            if img.width * img.height > 40_000_000:
                raise ValueError("the image is too large (over 40 megapixels)")
            img = img.convert("RGB")
            stage("enhancing + reading text")
            t, conf, used = _process_image(img, opts, get_model, notes, "image", check)
            texts.append(t)
            methods.add(used)
            if conf is not None:
                confidences.append(conf)
            row["pages_processed"] = "1/1"
    except BatchCancelled:
        raise
    except Exception as exc:
        row["error_message"] = f"Could not read the file: {type(exc).__name__}: {str(exc)[:160]}"
        row["processing_time_seconds"] = round(time.time() - start, 2)
        return row

    stage("finishing")
    text = "\n\n".join(t for t in texts if t and t.strip())
    row["extracted_text"] = text
    row["ocr_language"] = detect_language(text) if text else ""
    enhanced = [m for m in methods if m != "None"]
    row["enhancement_method"] = enhanced[0] if enhanced else "None"
    problems = list(page_errors)
    if not text.strip():
        row["status"] = "Failed"
        problems.insert(0, "no readable text was found")
    elif page_errors or len(text) < opts["min_text_chars"] or (
            confidences and sum(confidences) / len(confidences) < LIMITS["min_confidence"]):
        row["status"] = "Partial Success"
        if len(text) < opts["min_text_chars"]:
            problems.append(f"only {len(text)} characters extracted")
        if confidences and sum(confidences) / len(confidences) < LIMITS["min_confidence"]:
            problems.append(f"low OCR confidence ({sum(confidences) / len(confidences):.2f})")
    else:
        row["status"] = "Success"
    row["error_message"] = "; ".join(problems)
    row["notes"] = "; ".join(n for n in [row["notes"]] + notes if n)
    row["processing_time_seconds"] = round(time.time() - start, 2)
    return row


# ============================================================
# Background job queue
# ============================================================

def default_workers():
    """Parallel workers per batch: one per two logical CPUs (a worker's OCR / Diffusion step already uses
    several threads), between 1 and 4."""
    return max(1, min(4, (os.cpu_count() or 2) // 2))


class BulkManager:
    """One worker thread processes batches in submission order (CPU-bound models would only compete in parallel).
    State is written to disk after every file, so results survive a page refresh; a server restart marks
    unfinished files as Pending and they can be resumed with "Retry"."""

    def __init__(self, get_model, root=BULK_ROOT):
        self.get_model = get_model
        self.root = root
        self.jobs = {}
        self.lock = threading.RLock()
        self.queue = queue.Queue()
        os.makedirs(root, exist_ok=True)
        self.cleanup_expired()
        self._load_existing()
        self.worker = threading.Thread(target=self._run, name="recruitgen-bulk-worker", daemon=True)
        self.worker.start()

    # ---------- storage ----------
    def _dir(self, owner, batch_id):
        path = os.path.realpath(os.path.join(self.root, str(owner), batch_id))
        if not path.startswith(os.path.realpath(self.root) + os.sep):
            raise ValueError("invalid batch path")
        return path

    def _save(self, job):
        d = self._dir(job["owner"], job["id"])
        meta = {k: v for k, v in job.items() if k not in ("items",)}
        meta["items"] = [{k: v for k, v in it.items() if k != "data"} for it in job["items"]]
        tmp = os.path.join(d, "job.json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
        os.replace(tmp, os.path.join(d, "job.json"))

    def _load_existing(self):
        for owner in os.listdir(self.root):
            odir = os.path.join(self.root, owner)
            if not os.path.isdir(odir):
                continue
            for bid in os.listdir(odir):
                path = os.path.join(odir, bid, "job.json")
                if os.path.exists(path):
                    try:
                        with open(path, encoding="utf-8") as f:
                            job = json.load(f)
                    except (OSError, ValueError):
                        continue
                    for it in job["items"]:
                        if it["state"] == "processing":
                            it["state"] = "pending"
                    if job["state"] in ("queued", "running"):
                        job["state"] = "interrupted"
                    job.update(current=None, active={}, cancel=False)
                    self.jobs[job["id"]] = job

    def cleanup_expired(self):
        cutoff = time.time() - RETENTION_HOURS * 3600
        for owner in os.listdir(self.root) if os.path.isdir(self.root) else []:
            odir = os.path.join(self.root, owner)
            for bid in os.listdir(odir) if os.path.isdir(odir) else []:
                bdir = os.path.join(odir, bid)
                if os.path.isdir(bdir) and os.path.getmtime(bdir) < cutoff:
                    shutil.rmtree(bdir, ignore_errors=True)

    # ---------- public API (always scoped to the owner) ----------
    def submit(self, owner, zip_name, items, opts):
        batch_id = uuid.uuid4().hex
        d = self._dir(owner, batch_id)
        os.makedirs(os.path.join(d, "input"), exist_ok=True)
        job = {"id": batch_id, "owner": owner, "zip_name": zip_name, "created": time.time(), "started": None,
               "finished": None, "state": "queued", "opts": opts, "current": None, "items": [], "rows": {}}
        for it in items:
            rid = f"R{it['index']:03d}"
            entry = {"index": it["index"], "resume_id": rid, "filename": it["filename"], "ext": it["ext"],
                     "kind": it["kind"], "error": it["error"], "note": it.get("note", ""), "state": "pending"}
            if it["data"] is not None:          # stored under a generated name — never the archive path
                with open(os.path.join(d, "input", f"{it['index']:03d}.{it['ext']}"), "wb") as f:
                    f.write(it["data"])
            job["items"].append(entry)
        with self.lock:
            self.jobs[batch_id] = job
            self._save(job)
        self.queue.put(batch_id)
        return batch_id

    def get(self, owner, batch_id):
        job = self.jobs.get(batch_id)
        return job if job and str(job["owner"]) == str(owner) else None

    def list(self, owner):
        return sorted((j for j in self.jobs.values() if str(j["owner"]) == str(owner)), key=lambda j: -j["created"])

    def retry(self, owner, batch_id):
        """Re-queue failed and unfinished files of a finished / interrupted batch."""
        job = self.get(owner, batch_id)
        if not job or job["state"] in ("queued", "running"):
            return 0
        n = 0
        with self.lock:
            for it in job["items"]:
                row = job["rows"].get(it["resume_id"])
                if it["state"] == "pending" or (row and row["status"] == "Failed" and not it["error"]):
                    it["state"] = "pending"
                    it["retries"] = it.get("retries", 0) + 1
                    n += 1
            if n:
                job.update(state="queued", finished=None, cancel=False)
                self._save(job)
        if n:
            self.queue.put(batch_id)
        return n

    def delete(self, owner, batch_id):
        job = self.get(owner, batch_id)
        if not job or job["state"] == "running":
            return False
        with self.lock:
            self.jobs.pop(batch_id, None)
        shutil.rmtree(self._dir(owner, batch_id), ignore_errors=True)
        return True

    def results(self, owner, batch_id):
        """Rows in upload order (exactly one per uploaded file) + structured rows."""
        job = self.get(owner, batch_id)
        if not job:
            return [], []
        reason = ("Not processed — the batch was stopped by the user (use Resume to finish it)"
                  if job["state"] == "stopped" else "Not processed yet")
        rows = [job["rows"].get(it["resume_id"]) if it["state"] == "done" or it["resume_id"] in job["rows"] else
                {"resume_id": it["resume_id"], "filename": it["filename"], "status": "Failed", "extracted_text": "",
                 "error_message": reason, "enhancement_method": "None", "processing_time_seconds": 0.0,
                 "ocr_language": "", "pages_processed": "", "notes": ""}
                for it in job["items"]]
        structured = [{"resume_id": r["resume_id"], "filename": r["filename"],
                       **(structured_fields(r["extracted_text"]) if r.get("extracted_text") else {})} for r in rows]
        return rows, structured

    # ---------- worker ----------
    def _run(self):
        while True:
            batch_id = self.queue.get()
            job = self.jobs.get(batch_id)
            if not job or job["state"] not in ("queued",):
                continue
            try:
                self._run_job(job)
            except Exception as exc:          # a broken batch must never kill the worker
                job["state"] = "error"
                job["error"] = f"{type(exc).__name__}"
                self._save(job)

    def _run_job(self, job):
        """Process the pending files of one batch with a pool of parallel workers. The CPU is split between the
        workers (OCR engine threads per worker, PyTorch threads for the Diffusion model), so they do not fight
        over the same cores. Stops between files / pages / diffusion tiles when the user presses Stop."""
        from concurrent.futures import ThreadPoolExecutor

        d = self._dir(job["owner"], job["id"])
        pending = [it for it in job["items"] if it["state"] == "pending"]
        workers = max(1, min(int(job["opts"].get("workers") or default_workers()), len(pending) or 1))
        cpu = os.cpu_count() or 1
        opts = {"min_text_chars": LIMITS["min_text_chars"], **job["opts"], "ocr_threads": max(1, cpu // workers)}
        with self.lock:
            job.update(state="running", started=job["started"] or time.time(), active={}, current=None,
                       workers_used=workers)
            self._save(job)
        torch_threads = None
        if job["opts"].get("enhancement") in ("diffusion", "autoencoder"):
            import torch
            torch_threads = torch.get_num_threads()
            torch.set_num_threads(max(1, cpu // workers))

        cancelled = lambda: bool(job.get("cancel")) or self.jobs.get(job["id"]) is not job  # noqa: E731

        def work(it):
            if cancelled():
                return
            with self.lock:
                it["state"] = "processing"
                info = {"resume_id": it["resume_id"], "filename": it["filename"], "stage": "starting",
                        "since": time.time()}
                job["active"][it["resume_id"]] = info
                job["current"] = info
            data = None
            path = os.path.join(d, "input", f"{it['index']:03d}.{it['ext']}")
            if not it["error"] and os.path.exists(path):
                with open(path, "rb") as f:
                    data = f.read()
            item = {**it, "data": data, "error": it["error"] or (None if data is not None else "input file missing")}
            try:
                row = process_resume(item, opts, self.get_model, lambda s: info.__setitem__("stage", s), cancelled)
            except BatchCancelled:
                with self.lock:
                    it["state"] = "pending"                  # not processed: Resume finishes it later
                    job["active"].pop(it["resume_id"], None)
                return
            with self.lock:
                job["rows"][it["resume_id"]] = row
                it["state"] = "done"
                job["active"].pop(it["resume_id"], None)
                job["current"] = next(iter(job["active"].values()), None)
                self._save(job)

        try:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="recruitgen-bulk") as pool:
                list(pool.map(work, pending))
        finally:
            if torch_threads:
                import torch
                torch.set_num_threads(torch_threads)
        with self.lock:
            stopped = bool(job.get("cancel")) and any(it["state"] == "pending" for it in job["items"])
            job.update(state="stopped" if stopped else "finished", finished=time.time(), current=None, active={},
                       cancel=False)
            if self.jobs.get(job["id"]) is job:
                self._save(job)

    def stop(self, owner, batch_id):
        """Stop a queued or running batch. Finished files keep their results; the rest stay 'pending'."""
        job = self.get(owner, batch_id)
        if not job or job["state"] not in ("queued", "running"):
            return False
        with self.lock:
            job["cancel"] = True
            if job["state"] == "queued":                   # never started: nothing to wait for
                job.update(state="stopped", finished=time.time(), cancel=False)
            self._save(job)
        return True

    @staticmethod
    def counts(job):
        rows = list(job["rows"].values())
        done = sum(it["state"] == "done" for it in job["items"])
        return {"total": len(job["items"]), "completed": done,
                "success": sum(r["status"] == "Success" for r in rows),
                "partial": sum(r["status"] == "Partial Success" for r in rows),
                "failed": sum(r["status"] == "Failed" for r in rows)}
