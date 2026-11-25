"""Brand colours, fonts, text cleaning and paragraph styles shared by every RecruitGen PDF."""

import importlib.util
import os
import re
import unicodedata
from functools import lru_cache
from xml.sax.saxutils import escape

import reportlab
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ---------------- colours (print-friendly versions of the app palette) ----------------
PRIMARY = colors.HexColor("#7C5CFC")
PRIMARY_DARK = colors.HexColor("#5B3FD6")
PRIMARY_SOFT = colors.HexColor("#F1EDFF")
LAVENDER = colors.HexColor("#C4B5FD")
ACCENT = colors.HexColor("#22D3EE")
ACCENT_DARK = colors.HexColor("#0E7490")
ACCENT_SOFT = colors.HexColor("#ECFEFF")
SUCCESS = colors.HexColor("#059669")
SUCCESS_SOFT = colors.HexColor("#ECFDF5")
WARNING = colors.HexColor("#D97706")
WARNING_SOFT = colors.HexColor("#FFFBEB")
DANGER = colors.HexColor("#DC2626")
DANGER_SOFT = colors.HexColor("#FEF2F2")
INK = colors.HexColor("#0F172A")
TEXT = colors.HexColor("#334155")
MUTED = colors.HexColor("#64748B")
SUBTLE = colors.HexColor("#94A3B8")
LINE = colors.HexColor("#E2E8F0")
TRACK = colors.HexColor("#EEF2F7")
PANEL = colors.HexColor("#F8FAFC")
NIGHT = colors.HexColor("#0B1020")
NIGHT_2 = colors.HexColor("#2A1C6B")
COVER_TEXT = colors.HexColor("#C7D2FE")
WHITE = colors.white
GOLD = colors.HexColor("#F59E0B")
SILVER = colors.HexColor("#94A3B8")
BRONZE = colors.HexColor("#B45309")

SERIES = [PRIMARY, ACCENT_DARK, colors.HexColor("#A78BFA"), SUCCESS, GOLD, colors.HexColor("#EC4899"),
          colors.HexColor("#3B82F6"), colors.HexColor("#14B8A6"), colors.HexColor("#F97316"), SUBTLE]


def score_color(score):
    if score is None:
        return MUTED
    return SUCCESS if score >= 75 else WARNING if score >= 50 else DANGER


def hex_of(color):
    return "#" + color.hexval()[2:]


# ---------------- fonts ----------------
# The first complete family found on the machine is embedded. ReportLab's bundled Vera fonts are the
# guaranteed fallback, so PDFs render the same on Windows, Linux (Streamlit Cloud) and macOS.

def _font_sets():
    win = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
    rl = os.path.join(os.path.dirname(reportlab.__file__), "fonts")
    sets = [
        [os.path.join(win, f) for f in ("segoeui.ttf", "segoeuib.ttf", "segoeuii.ttf", "segoeuiz.ttf")],
        [os.path.join(win, f) for f in ("arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf")],
    ]
    for base in ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/dejavu", "/usr/share/fonts/TTF"):
        sets.append([os.path.join(base, f) for f in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf",
                                                     "DejaVuSans-Oblique.ttf", "DejaVuSans-BoldOblique.ttf")])
    sets.append([os.path.join("/usr/share/fonts/truetype/liberation", f) for f in (
        "LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf", "LiberationSans-Italic.ttf",
        "LiberationSans-BoldItalic.ttf")])
    sets.append([os.path.join("/System/Library/Fonts/Supplemental", f) for f in (
        "Arial.ttf", "Arial Bold.ttf", "Arial Italic.ttf", "Arial Bold Italic.ttf")])
    spec = importlib.util.find_spec("matplotlib")
    if spec and spec.origin:
        mpl = os.path.join(os.path.dirname(spec.origin), "mpl-data", "fonts", "ttf")
        sets.append([os.path.join(mpl, f) for f in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf",
                                                    "DejaVuSans-Oblique.ttf", "DejaVuSans-BoldOblique.ttf")])
    sets.append([os.path.join(rl, f) for f in ("Vera.ttf", "VeraBd.ttf", "VeraIt.ttf", "VeraBI.ttf")])
    return sets


@lru_cache(maxsize=None)
def fonts():
    names = ("RG", "RG-Bold", "RG-Italic", "RG-BoldItalic")
    for paths in _font_sets():
        if not all(os.path.isfile(p) for p in paths):
            continue
        try:
            for name, path in zip(names, paths):
                pdfmetrics.registerFont(TTFont(name, path))
            pdfmetrics.registerFontFamily("RG", normal=names[0], bold=names[1], italic=names[2], boldItalic=names[3])
            return {"regular": names[0], "bold": names[1], "italic": names[2], "bold_italic": names[3], "ttf": True}
        except Exception:
            continue
    return {"regular": "Helvetica", "bold": "Helvetica-Bold", "italic": "Helvetica-Oblique",
            "bold_italic": "Helvetica-BoldOblique", "ttf": False}


@lru_cache(maxsize=None)
def _coverage():
    f = fonts()
    if not f["ttf"]:
        return None
    return frozenset(pdfmetrics.getFont(f["regular"]).face.charToGlyph)


# ---------------- text cleaning ----------------

_SUBSTITUTES = {
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2212": "-",
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2022": "*", "\u2026": "...",
    "\u2192": "->", "\u2190": "<-", "\u2713": "v", "\u2714": "v", "\u2605": "*", "\u20b9": "Rs.",
}


def clean(text):
    """Make any text safe for the embedded font: unsupported symbols (emoji, rare dashes…) are
    replaced with a close equivalent or removed, so nothing renders as an empty box."""
    if text is None:
        return ""
    covered = _coverage()
    out, removed = [], False
    for ch in str(text):
        if ch in "\n\t":
            out.append(ch)
            continue
        cat = unicodedata.category(ch)
        if cat[0] == "C" or cat == "Me" or 0xFE00 <= ord(ch) <= 0xFE0F or 0x1F3FB <= ord(ch) <= 0x1F3FF:
            removed = True  # control/format chars, keycap & emoji modifiers, variation selectors
            continue
        ok = ord(ch) in covered if covered is not None else bool(ch.encode("cp1252", "ignore"))
        if ok:
            out.append(ch)
            continue
        sub = _SUBSTITUTES.get(ch)
        if sub is None:
            sub = unicodedata.normalize("NFKD", ch).encode("ascii", "ignore").decode()
        removed = removed or not sub
        out.append(sub)
    result = "".join(out)
    if removed:  # tidy the gap a dropped emoji leaves behind ("delivery 🚀." -> "delivery.")
        result = re.sub(r"(?<=\S)[ \t]+(?=[.,;:!?)])", "", re.sub(r"(?<=\S)[ \t]{2,}(?=\S)", " ", result)).strip(" \t")
    return result


def esc(text):
    """Clean + escape text for ReportLab paragraph markup."""
    return escape(clean(text))


def md_inline(text):
    """Escape text and convert inline Markdown (**bold**, *italic*, `code`, [link](url)) to markup."""
    s = esc(text)
    s = re.sub(r"`([^`\n]+)`", lambda m: f'<font color="{hex_of(PRIMARY_DARK)}">{m.group(1)}</font>', s)
    s = re.sub(r"\*\*(?=\S)(.+?)(?<=\S)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"__(?=\S)(.+?)(?<=\S)__", r"<b>\1</b>", s)
    s = re.sub(r"(?<![\w*])\*(?=[^\s*])([^*\n]+?)(?<=[^\s*])\*(?![\w*])", r"<i>\1</i>", s)
    s = re.sub(r"\[([^\]\n]+)\]\((https?://[^)\s]+)\)",
               lambda m: f'<link href="{m.group(2)}"><font color="{hex_of(PRIMARY_DARK)}"><u>{m.group(1)}</u></font></link>',
               s)
    return s


def ellipsis():
    return "…" if (_coverage() is None or ord("…") in _coverage()) else "..."


# ---------------- paragraph styles ----------------

@lru_cache(maxsize=None)
def styles():
    f = fonts()
    r, b, i = f["regular"], f["bold"], f["italic"]
    body = dict(fontName=r, fontSize=9.4, leading=14, textColor=TEXT)
    cell = dict(fontName=r, fontSize=8, leading=10.6, textColor=TEXT)
    return {
        "body": ParagraphStyle("rg_body", spaceAfter=5, **body),
        "lead": ParagraphStyle("rg_lead", fontName=r, fontSize=10.4, leading=15.6, textColor=INK, spaceAfter=6),
        "small": ParagraphStyle("rg_small", fontName=r, fontSize=8, leading=11.2, textColor=MUTED),
        "h2": ParagraphStyle("rg_h2", fontName=b, fontSize=11.5, leading=15, textColor=INK, spaceBefore=8,
                             spaceAfter=4, keepWithNext=1),
        "h3": ParagraphStyle("rg_h3", fontName=b, fontSize=9.6, leading=13, textColor=PRIMARY_DARK, spaceBefore=6,
                             spaceAfter=5, keepWithNext=1),
        "bullet": ParagraphStyle("rg_bullet", leftIndent=14, bulletIndent=3, bulletFontName=b, bulletFontSize=9.4,
                                 spaceAfter=3.5, **body),
        "quote": ParagraphStyle("rg_quote", fontName=i, fontSize=9.4, leading=14, textColor=MUTED, leftIndent=10,
                                spaceAfter=5),
        "cell": ParagraphStyle("rg_cell", **cell),
        "cell_right": ParagraphStyle("rg_cell_r", alignment=TA_RIGHT, **cell),
        "cell_center": ParagraphStyle("rg_cell_c", alignment=TA_CENTER, **cell),
        "cell_bold": ParagraphStyle("rg_cell_b", fontName=b, fontSize=8, leading=10.6, textColor=INK),
        "cell_head": ParagraphStyle("rg_cell_h", fontName=b, fontSize=7.8, leading=10.2, textColor=WHITE),
        "cell_head_right": ParagraphStyle("rg_cell_hr", fontName=b, fontSize=7.8, leading=10.2, textColor=WHITE,
                                          alignment=TA_RIGHT),
        "cell_head_center": ParagraphStyle("rg_cell_hc", fontName=b, fontSize=7.8, leading=10.2, textColor=WHITE,
                                           alignment=TA_CENTER),
        "caption": ParagraphStyle("rg_caption", fontName=r, fontSize=7.8, leading=10.4, textColor=MUTED,
                                  alignment=TA_CENTER),
        "code": ParagraphStyle("rg_code", fontName="Courier", fontSize=7.6, leading=9.8, textColor=INK),
        "text": ParagraphStyle("rg_text", fontName=r, fontSize=8.2, leading=11.4, textColor=TEXT),
        "card_title": ParagraphStyle("rg_card_title", fontName=b, fontSize=9.8, leading=13.4, textColor=INK,
                                     spaceAfter=3),
        "card_text": ParagraphStyle("rg_card_text", fontName=r, fontSize=8.4, leading=12, textColor=MUTED),
    }
