"""Page layout shared by every RecruitGen PDF: branded cover header on page 1, running header and
footer on the following pages, and "Page X of Y" numbering."""

import io
import re
from datetime import datetime

from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.platypus import BaseDocTemplate, Frame, NextPageTemplate, PageTemplate

from . import theme as t

MARGIN = 16 * mm
COVER_H = 52 * mm
TOP = 21 * mm
BOTTOM = 19 * mm


def page_size(wide=False):
    return landscape(A4) if wide else A4


def content_width(wide=False):
    return page_size(wide)[0] - 2 * MARGIN


def body_height(wide=False):
    """Usable frame height on pages after the cover."""
    return page_size(wide)[1] - TOP - BOTTOM


def pdf_filename(kind, name=""):
    """recruitgen_<kind>_<name>_<date>.pdf with a filesystem-safe name."""
    slug = re.sub(r"[^a-z0-9]+", "_", f"{kind} {name}".lower()).strip("_")[:60] or "report"
    return f"recruitgen_{slug}_{datetime.now():%Y%m%d}.pdf"


def _truncate(text, font, size, width):
    text = t.clean(text)
    if stringWidth(text, font, size) <= width:
        return text
    dots = t.ellipsis()
    while text and stringWidth(text + dots, font, size) > width:
        text = text[:-1]
    return text.rstrip() + dots


def _fit(text, font, size, width, min_size):
    text = t.clean(text)
    while size > min_size and stringWidth(text, font, size) > width:
        size -= 0.5
    return _truncate(text, font, size, width), size


def _gradient_rect(c, x, y, w, h, colors, radius=0, horizontal=True):
    c.saveState()
    path = c.beginPath()
    if radius:
        path.roundRect(x, y, w, h, radius)
    else:
        path.rect(x, y, w, h)
    c.clipPath(path, stroke=0, fill=0)
    if horizontal:
        c.linearGradient(x, y, x + w, y, colors, extend=False)
    else:
        c.linearGradient(x, y, x + w, y + h, colors, extend=False)
    c.restoreState()


def draw_logo(c, x, y, size):
    """Rounded gradient square with the RG monogram."""
    _gradient_rect(c, x, y, size, size, (t.PRIMARY, t.ACCENT), radius=size * 0.28, horizontal=False)
    c.setFillColor(t.WHITE)
    fs = size * 0.42
    c.setFont(t.fonts()["bold"], fs)
    c.drawCentredString(x + size / 2, y + size / 2 - fs * 0.36, "RG")


class _NumberedCanvas(rl_canvas.Canvas):
    """Canvas that knows the total page count, so every footer can say "Page 2 of 5"."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._page_states = []

    def showPage(self):
        self._page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._page_states)
        for state in self._page_states:
            self.__dict__.update(state)
            w, _ = self._pagesize
            self.setFont(t.fonts()["bold"], 7.4)
            self.setFillColor(t.MUTED)
            self.drawRightString(w - MARGIN, 9.2 * mm, f"Page {self._pageNumber} of {total}")
            super().showPage()
        super().save()


def _draw_footer(c, info):
    w, _ = c._pagesize
    f = t.fonts()
    c.setStrokeColor(t.LINE)
    c.setLineWidth(0.6)
    c.line(MARGIN, 13.5 * mm, w - MARGIN, 13.5 * mm)
    c.setFont(f["regular"], 7.2)
    c.setFillColor(t.SUBTLE)
    text = f"RecruitGen · {info['kind']} · Generated {info['generated']} · Confidential"
    c.drawString(MARGIN, 9.2 * mm, _truncate(text, f["regular"], 7.2, w - 2 * MARGIN - 70))


def _draw_cover(c, doc, info):
    w, h = c._pagesize
    f = t.fonts()
    y0 = h - COVER_H

    # Gradient band with soft glow circles.
    c.saveState()
    band = c.beginPath()
    band.rect(0, y0, w, COVER_H)
    c.clipPath(band, stroke=0, fill=0)
    c.linearGradient(0, y0, w, h, (t.NIGHT, t.NIGHT_2), extend=False)
    c.setFillColor(t.PRIMARY)
    c.setFillAlpha(0.28)
    c.circle(w - 36 * mm, h + 2 * mm, 36 * mm, stroke=0, fill=1)
    c.setFillColor(t.ACCENT)
    c.setFillAlpha(0.14)
    c.circle(w - 6 * mm, y0 + 2 * mm, 24 * mm, stroke=0, fill=1)
    c.restoreState()
    _gradient_rect(c, 0, y0 - 2, w, 2, (t.PRIMARY, t.ACCENT))

    # Brand row.
    logo = 11 * mm
    lx, ly = MARGIN, h - 9 * mm - logo
    draw_logo(c, lx, ly, logo)
    c.setFillColor(t.WHITE)
    c.setFont(f["bold"], 13.5)
    c.drawString(lx + logo + 3.2 * mm, ly + 5.6 * mm, "RecruitGen")
    c.setFillColor(t.COVER_TEXT)
    c.setFont(f["regular"], 7.6)
    c.drawString(lx + logo + 3.2 * mm, ly + 1.5 * mm, "AI Recruitment & Candidate Screening")

    # Meta block (top right).
    meta = [("Generated", info["generated"])] + [(k, v) for k, v in info["meta"].items() if v]
    my = h - 12.5 * mm
    for key, value in meta[:4]:
        c.setFont(f["regular"], 7.2)
        c.setFillColor(t.COVER_TEXT)
        value = _truncate(value, f["bold"], 7.4, 64 * mm)
        vw = stringWidth(value, f["bold"], 7.4)
        c.drawRightString(w - MARGIN - vw - 4, my, t.clean(key).upper())
        c.setFont(f["bold"], 7.4)
        c.setFillColor(t.WHITE)
        c.drawRightString(w - MARGIN, my, value)
        my -= 4.2 * mm

    # Eyebrow, title, subtitle.
    c.saveState()  # character spacing is graphics state: keep it from leaking into later text
    to = c.beginText(MARGIN, y0 + 24.5 * mm)
    to.setFont(f["bold"], 7.8)
    to.setCharSpace(1.6)
    to.setFillColor(t.ACCENT)
    to.textOut(_truncate(info["eyebrow"].upper(), f["bold"], 7.8, w - 2 * MARGIN - 60))
    to.setCharSpace(0)
    c.drawText(to)
    c.restoreState()
    title, size = _fit(info["title"], f["bold"], 22, w - 2 * MARGIN, 14)
    c.setFillColor(t.WHITE)
    c.setFont(f["bold"], size)
    c.drawString(MARGIN, y0 + 14.5 * mm, title)
    if info["subtitle"]:
        c.setFillColor(t.COVER_TEXT)
        c.setFont(f["regular"], 9.2)
        c.drawString(MARGIN, y0 + 7.5 * mm, _truncate(info["subtitle"], f["regular"], 9.2, w - 2 * MARGIN))
    _draw_footer(c, info)


def _draw_running(c, doc, info):
    w, h = c._pagesize
    f = t.fonts()
    draw_logo(c, MARGIN, h - 14 * mm, 6 * mm)
    c.setFont(f["bold"], 9)
    c.setFillColor(t.INK)
    c.drawString(MARGIN + 8 * mm, h - 12 * mm, "RecruitGen")
    c.setFont(f["regular"], 8)
    c.setFillColor(t.MUTED)
    c.drawRightString(w - MARGIN, h - 12 * mm,
                      _truncate(f"{info['kind']} · {info['title']}", f["regular"], 8, w - 2 * MARGIN - 40 * mm))
    c.setStrokeColor(t.LINE)
    c.setLineWidth(0.6)
    c.line(MARGIN, h - 16.5 * mm, w - MARGIN, h - 16.5 * mm)
    _draw_footer(c, info)


def build_pdf(story, *, title, kind, eyebrow, subtitle="", meta=None, wide=False):
    """Lay out `story` (a list of flowables) on branded pages and return the PDF bytes."""
    buf = io.BytesIO()
    size = page_size(wide)
    w, h = size
    info = {"title": t.clean(title) or kind, "kind": t.clean(kind), "eyebrow": t.clean(eyebrow),
            "subtitle": t.clean(subtitle), "meta": {k: t.clean(v) for k, v in (meta or {}).items()},
            "generated": datetime.now().strftime("%d %b %Y, %I:%M %p")}
    doc = BaseDocTemplate(
        buf, pagesize=size, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=TOP, bottomMargin=BOTTOM,
        title=f"{info['kind']} - {info['title']}", author="RecruitGen", subject=info["subtitle"] or info["kind"],
        creator="RecruitGen AI Recruitment & Screening",
    )
    pad = dict(leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    first = Frame(MARGIN, BOTTOM, w - 2 * MARGIN, h - COVER_H - 8 * mm - BOTTOM, id="cover", **pad)
    later = Frame(MARGIN, BOTTOM, w - 2 * MARGIN, h - TOP - BOTTOM, id="body", **pad)
    doc.addPageTemplates([
        PageTemplate(id="cover", frames=[first], onPage=lambda c, d: _draw_cover(c, d, info)),
        PageTemplate(id="body", frames=[later], onPage=lambda c, d: _draw_running(c, d, info)),
    ])
    doc.build([NextPageTemplate("body")] + list(story), canvasmaker=_NumberedCanvas)
    return buf.getvalue()
