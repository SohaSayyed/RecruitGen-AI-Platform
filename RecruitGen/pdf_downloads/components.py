"""Reusable PDF building blocks: section titles, KPI cards, score rings, bar charts, donut chart,
chips, styled tables, image grids, callouts, interview rating rows and a small Markdown renderer."""

import io
import re

from PIL import Image as PILImage
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import Flowable, Image, Paragraph, Preformatted, Spacer, Table, TableStyle
from reportlab.platypus.flowables import HRFlowable

from . import theme as t
from .theme import clean, esc, md_inline


# ============================================================
# Text helpers
# ============================================================

def fit_text(text, font, size, max_width, min_size=None):
    """Shrink (down to min_size) and then truncate with an ellipsis so the text fits max_width."""
    text = clean(text)
    min_size = min_size or size
    while size > min_size and stringWidth(text, font, size) > max_width:
        size -= 0.5
    if stringWidth(text, font, size) <= max_width:
        return text, size
    dots = t.ellipsis()
    while text and stringWidth(text + dots, font, size) > max_width:
        text = text[:-1]
    return text.rstrip() + dots, size


def _spaced(c, x, y, text, font, size, spacing=0.9, color=None):
    """Letter-spaced label (used for small uppercase captions). Character spacing is part of the PDF
    graphics state, so it is wrapped in save/restore to keep it from leaking into later text."""
    c.saveState()
    to = c.beginText(x, y)
    to.setFont(font, size)
    to.setCharSpace(spacing)
    if color is not None:
        to.setFillColor(color)
    to.textOut(text)
    to.setCharSpace(0)
    c.drawText(to)
    c.restoreState()


def _gradient_bar(c, x, y, w, h, colors):
    c.saveState()
    path = c.beginPath()
    path.roundRect(x, y, w, h, h / 2)
    c.clipPath(path, stroke=0, fill=0)
    c.linearGradient(x, y, x + w, y, colors, extend=False)
    c.restoreState()


def para(text, style="body", markdown=True):
    return Paragraph(md_inline(text) if markdown else esc(text), t.styles()[style])


def spacer(height=8):
    return Spacer(1, height)


def divider(space=8):
    return HRFlowable(width="100%", thickness=0.7, color=t.LINE, spaceBefore=space, spaceAfter=space)


# ============================================================
# Headings & cards
# ============================================================

class SectionTitle(Flowable):
    """Section heading with a violet accent bar and an optional muted subtitle."""

    def __init__(self, title, subtitle=None, color=None, space_before=12):
        super().__init__()
        self.title = clean(title)
        self.subtitle = clean(subtitle) if subtitle else ""
        self.color = color or t.PRIMARY
        self.space_before = space_before
        self.keepWithNext = 1

    def wrap(self, aw, ah):
        self.width = aw
        self.height = self.space_before + (32 if self.subtitle else 22)
        return aw, self.height

    def draw(self):
        c, f = self.canv, t.fonts()
        top = self.height - self.space_before
        c.setFillColor(self.color)
        c.roundRect(0, top - 15, 3.4, 14.5, 1.7, stroke=0, fill=1)
        c.setFillColor(t.INK)
        title, size = fit_text(self.title, f["bold"], 12.6, self.width - 12, 10)
        c.setFont(f["bold"], size)
        c.drawString(10, top - 12, title)
        if self.subtitle:
            sub, _ = fit_text(self.subtitle, f["regular"], 8.2, self.width - 12)
            c.setFillColor(t.MUTED)
            c.setFont(f["regular"], 8.2)
            c.drawString(10, top - 25, sub)


class MetricCards(Flowable):
    """A row of KPI cards. items: [(label, value, sub, color)]."""

    def __init__(self, items, height=58, gap=8):
        super().__init__()
        self.items = [i for i in items if i]
        self.h, self.gap = height, gap

    def wrap(self, aw, ah):
        self.width = aw
        return aw, self.h

    def draw(self):
        c, f = self.canv, t.fonts()
        n = max(1, len(self.items))
        cw = (self.width - self.gap * (n - 1)) / n
        for i, (label, value, sub, color) in enumerate(self.items):
            x = i * (cw + self.gap)
            color = color or t.PRIMARY
            c.setFillColor(t.PANEL)
            c.setStrokeColor(t.LINE)
            c.setLineWidth(0.7)
            c.roundRect(x, 0, cw, self.h, 7, stroke=1, fill=1)
            c.setFillColor(color)
            c.roundRect(x + 0.8, 8, 2.6, self.h - 16, 1.3, stroke=0, fill=1)
            pad = 11
            lab, _ = fit_text(str(label).upper(), f["bold"], 6.6, cw - 2 * pad - 8)
            _spaced(c, x + pad, self.h - 15, lab, f["bold"], 6.6, 0.8, t.MUTED)
            val, size = fit_text(value, f["bold"], 15.5, cw - 2 * pad, 9)
            c.setFillColor(color)
            c.setFont(f["bold"], size)
            c.drawString(x + pad, self.h - 35, val)
            if sub:
                s, _ = fit_text(sub, f["regular"], 7.2, cw - 2 * pad)
                c.setFillColor(t.MUTED)
                c.setFont(f["regular"], 7.2)
                c.drawString(x + pad, 10, s)


class ScoreRings(Flowable):
    """Circular progress rings. items: [(value 0-100, label, centre_text, color)]."""

    def __init__(self, items, size=64, align="center"):
        super().__init__()
        self.items, self.size, self.align = items, size, align

    def wrap(self, aw, ah):
        self.width = aw
        self.height = self.size + 18
        return aw, self.height

    def draw(self):
        c, f = self.canv, t.fonts()
        n = len(self.items)
        slot = self.width / n if self.align == "center" else self.size + 26
        for i, (value, label, text, color) in enumerate(self.items):
            value = max(0.0, min(100.0, float(value or 0)))
            cx = slot * i + slot / 2
            cy = 16 + self.size / 2
            r = self.size / 2 - 4
            c.setLineWidth(6.5)
            c.setStrokeColor(t.TRACK)
            c.circle(cx, cy, r, stroke=1, fill=0)
            if value > 0:
                c.setStrokeColor(color or t.PRIMARY)
                c.setLineCap(1)
                sweep = 3.6 * value
                c.arc(cx - r, cy - r, cx + r, cy + r, startAng=90 - sweep, extent=sweep)
                c.setLineCap(0)
            txt, size = fit_text(text, f["bold"], 14, 2 * r - 12, 8)
            c.setFillColor(t.INK)
            c.setFont(f["bold"], size)
            c.drawCentredString(cx, cy - size * 0.35, txt)
            lab, _ = fit_text(label, f["regular"], 7.8, slot - 4)
            c.setFillColor(t.MUTED)
            c.setFont(f["regular"], 7.8)
            c.drawCentredString(cx, 2, lab)


class RankCards(Flowable):
    """Top-N role cards with medal badges. items: [(title, score 0-100, detail)]."""

    MEDALS = [t.GOLD, t.SILVER, t.BRONZE]

    def __init__(self, items, height=74, gap=9):
        super().__init__()
        self.items, self.h, self.gap = items, height, gap

    def wrap(self, aw, ah):
        self.width = aw
        return aw, self.h

    def draw(self):
        c, f = self.canv, t.fonts()
        n = max(1, len(self.items))
        cw = (self.width - self.gap * (n - 1)) / n
        for i, (title, score, detail) in enumerate(self.items):
            x = i * (cw + self.gap)
            first = i == 0
            c.setFillColor(t.WARNING_SOFT if first else t.PANEL)
            c.setStrokeColor(t.GOLD if first else t.LINE)
            c.setLineWidth(1.1 if first else 0.7)
            c.roundRect(x, 0, cw, self.h, 8, stroke=1, fill=1)
            medal = self.MEDALS[i] if i < 3 else t.PRIMARY
            c.setFillColor(medal)
            c.circle(x + 17, self.h - 17, 8.5, stroke=0, fill=1)
            c.setFillColor(t.WHITE)
            c.setFont(f["bold"], 9)
            c.drawCentredString(x + 17, self.h - 20.2, str(i + 1))
            _spaced(c, x + 31, self.h - 20, f"RANK {i + 1}", f["bold"], 6.6, 1, t.MUTED)
            name, size = fit_text(title, f["bold"], 11, cw - 22, 8.5)
            c.setFillColor(t.INK)
            c.setFont(f["bold"], size)
            c.drawString(x + 11, self.h - 40, name)
            bar_w = cw - 22 - 44
            c.setFillColor(t.TRACK)
            c.roundRect(x + 11, 22, bar_w, 6, 3, stroke=0, fill=1)
            frac = max(0.0, min(1.0, float(score) / 100))
            if frac > 0:
                _gradient_bar(c, x + 11, 22, max(6, bar_w * frac), 6, (t.PRIMARY, t.ACCENT))
            c.setFillColor(t.score_color(score))
            c.setFont(f["bold"], 11.5)
            c.drawRightString(x + cw - 11, 20.5, f"{score:.1f}%")
            if detail:
                d, _ = fit_text(detail, f["regular"], 7, cw - 22)
                c.setFillColor(t.MUTED)
                c.setFont(f["regular"], 7)
                c.drawString(x + 11, 8, d)


# ============================================================
# Charts
# ============================================================

class BarChart(Flowable):
    """Horizontal bar chart. items: [(label, value)] in display order; the first bar is highlighted."""

    def __init__(self, items, max_value=100, fmt="{:.1f}%", color=None, highlight=True, label_ratio=0.3, row_h=17):
        super().__init__()
        self.items = [(clean(k), float(v or 0)) for k, v in items]
        self.max_value = max_value or max([v for _, v in self.items] + [1])
        self.fmt, self.color, self.highlight = fmt, color, highlight
        self.label_ratio, self.row_h = label_ratio, row_h

    def wrap(self, aw, ah):
        self.width = aw
        self.height = len(self.items) * self.row_h + 2
        return aw, self.height

    def draw(self):
        c, f = self.canv, t.fonts()
        lw = self.width * self.label_ratio
        vw = 46
        bw = self.width - lw - vw - 6
        for i, (label, value) in enumerate(self.items):
            y = self.height - (i + 1) * self.row_h
            top = self.highlight and i == 0
            lab, _ = fit_text(label, f["bold"] if top else f["regular"], 8.4, lw - 8)
            c.setFillColor(t.INK if top else t.TEXT)
            c.setFont(f["bold"] if top else f["regular"], 8.4)
            c.drawString(0, y + 5, lab)
            c.setFillColor(t.TRACK)
            c.roundRect(lw, y + 4, bw, 8.5, 4.25, stroke=0, fill=1)
            frac = max(0.0, min(1.0, value / self.max_value))
            if frac > 0:
                width = max(8.5, bw * frac)
                if self.color is not None:
                    c.setFillColor(self.color)
                    c.roundRect(lw, y + 4, width, 8.5, 4.25, stroke=0, fill=1)
                elif top:
                    _gradient_bar(c, lw, y + 4, width, 8.5, (t.PRIMARY, t.ACCENT))
                else:
                    c.setFillColor(t.LAVENDER)
                    c.roundRect(lw, y + 4, width, 8.5, 4.25, stroke=0, fill=1)
            c.setFillColor(t.INK)
            c.setFont(f["bold"], 8.2)
            c.drawRightString(self.width, y + 5, self.fmt.format(value))


class GroupedBars(Flowable):
    """Several thin bars per row. rows: [(label, [v1, v2, ...])], series: [(name, color)]."""

    def __init__(self, rows, series, max_value=100, label_ratio=0.26):
        super().__init__()
        self.rows = [(clean(k), [float(v or 0) for v in vals]) for k, vals in rows]
        self.series, self.max_value, self.label_ratio = series, max_value, label_ratio
        self.bar_h, self.bar_gap, self.row_gap, self.legend_h = 5.5, 2.2, 9, 16

    def _row_h(self):
        n = len(self.series)
        return n * self.bar_h + (n - 1) * self.bar_gap + self.row_gap

    def wrap(self, aw, ah):
        self.width = aw
        self.height = self.legend_h + len(self.rows) * self._row_h()
        return aw, self.height

    def draw(self):
        c, f = self.canv, t.fonts()
        x = 0
        for name, color in self.series:
            c.setFillColor(color)
            c.roundRect(x, self.height - 8, 8, 6, 2, stroke=0, fill=1)
            c.setFillColor(t.MUTED)
            c.setFont(f["regular"], 7.6)
            c.drawString(x + 11, self.height - 7.6, clean(name))
            x += 22 + stringWidth(clean(name), f["regular"], 7.6)
        lw = self.width * self.label_ratio
        vw = 40
        bw = self.width - lw - vw - 6
        y = self.height - self.legend_h
        for label, values in self.rows:
            rh = self._row_h()
            block = rh - self.row_gap
            lab, _ = fit_text(label, f["regular"], 8.3, lw - 8)
            c.setFillColor(t.TEXT)
            c.setFont(f["regular"], 8.3)
            c.drawString(0, y - block / 2 - 3, lab)
            by = y
            for (name, color), value in zip(self.series, values):
                by -= self.bar_h
                c.setFillColor(t.TRACK)
                c.roundRect(lw, by, bw, self.bar_h, self.bar_h / 2, stroke=0, fill=1)
                frac = max(0.0, min(1.0, value / self.max_value))
                if frac > 0:
                    c.setFillColor(color)
                    c.roundRect(lw, by, max(self.bar_h, bw * frac), self.bar_h, self.bar_h / 2, stroke=0, fill=1)
                c.setFillColor(t.MUTED)
                c.setFont(f["regular"], 6.6)
                c.drawRightString(self.width, by + 0.6, f"{value:.1f}%")
                by -= self.bar_gap
            y -= rh


class Donut(Flowable):
    """Donut chart with a legend. data: [(label, value)]."""

    def __init__(self, data, size=104, center_label="Total", fmt="{:.0f}"):
        super().__init__()
        self.data = [(clean(k), float(v)) for k, v in data if v]
        self.size, self.center_label, self.fmt = size, center_label, fmt

    def wrap(self, aw, ah):
        self.width = aw
        self.height = max(self.size, len(self.data) * 15) + 4
        return aw, self.height

    def draw(self):
        c, f = self.canv, t.fonts()
        total = sum(v for _, v in self.data) or 1
        x0, y0 = 2, (self.height - self.size) / 2
        cx, cy = x0 + self.size / 2, y0 + self.size / 2
        start = 90.0
        c.setStrokeColor(t.WHITE)
        c.setLineWidth(1.6)
        for i, (_, v) in enumerate(self.data):
            extent = 360.0 * v / total
            c.setFillColor(t.SERIES[i % len(t.SERIES)])
            if extent >= 359.9:
                c.circle(cx, cy, self.size / 2, stroke=0, fill=1)
            else:
                c.wedge(x0, y0, x0 + self.size, y0 + self.size, start - extent, extent, stroke=1, fill=1)
            start -= extent
        c.setFillColor(t.WHITE)
        c.circle(cx, cy, self.size * 0.31, stroke=0, fill=1)
        c.setFillColor(t.INK)
        c.setFont(f["bold"], 15)
        c.drawCentredString(cx, cy - 2, self.fmt.format(total))
        c.setFillColor(t.MUTED)
        c.setFont(f["regular"], 6.8)
        c.drawCentredString(cx, cy - 11, clean(self.center_label))
        lx = x0 + self.size + 16
        lw = self.width - lx
        ly = self.height / 2 + len(self.data) * 15 / 2 - 11
        for i, (label, v) in enumerate(self.data):
            c.setFillColor(t.SERIES[i % len(t.SERIES)])
            c.roundRect(lx, ly + 1, 8, 8, 2, stroke=0, fill=1)
            value = f"{self.fmt.format(v)}  ({100 * v / total:.0f}%)"
            vw = stringWidth(value, f["bold"], 7.8)
            lab, _ = fit_text(label, f["regular"], 7.8, lw - vw - 22)
            c.setFillColor(t.TEXT)
            c.setFont(f["regular"], 7.8)
            c.drawString(lx + 13, ly + 2, lab)
            c.setFillColor(t.INK)
            c.setFont(f["bold"], 7.8)
            c.drawRightString(self.width, ly + 2, value)
            ly -= 15


# ============================================================
# Chips, lists, callouts
# ============================================================

class Chips(Flowable):
    """Wrapping rounded tags. kind: '' (violet), 'good', 'bad', 'info'."""

    KINDS = {
        "": (t.PRIMARY_SOFT, t.PRIMARY_DARK, t.LAVENDER),
        "good": (t.SUCCESS_SOFT, t.SUCCESS, "#A7F3D0"),
        "bad": (t.DANGER_SOFT, t.DANGER, "#FECACA"),
        "info": (t.ACCENT_SOFT, t.ACCENT_DARK, "#A5F3FC"),
    }

    def __init__(self, items, kind=""):
        super().__init__()
        self.items = [clean(i).strip() for i in items if str(i).strip()]
        self.kind = kind
        self.chip_h, self.pad, self.gap = 16, 8, 5

    def wrap(self, aw, ah):
        f = t.fonts()
        self.width = aw
        self.layout, x, row = [], 0.0, 0
        for text in self.items:
            label, _ = fit_text(text, f["regular"], 7.9, aw - 2 * self.pad)
            w = stringWidth(label, f["regular"], 7.9) + 2 * self.pad
            if x > 0 and x + w > aw:
                x, row = 0.0, row + 1
            self.layout.append((row, x, w, label))
            x += w + self.gap
        rows = row + 1 if self.items else 0
        self.height = rows * self.chip_h + max(0, rows - 1) * self.gap
        return aw, self.height

    def draw(self):
        from reportlab.lib import colors

        c, f = self.canv, t.fonts()
        bg, fg, border = (col if not isinstance(col, str) else colors.HexColor(col) for col in self.KINDS[self.kind])
        for row, x, w, label in self.layout:
            y = self.height - (row + 1) * self.chip_h - row * self.gap
            c.setFillColor(bg)
            c.setStrokeColor(border)
            c.setLineWidth(0.6)
            c.roundRect(x, y, w, self.chip_h, self.chip_h / 2, stroke=1, fill=1)
            c.setFillColor(fg)
            c.setFont(f["regular"], 7.9)
            c.drawString(x + self.pad, y + 5.1, label)


def bullets(items, color=None, numbered=False):
    """Bulleted (or numbered) list as paragraphs with a coloured bullet and hanging indent."""
    style = t.styles()["bullet"]
    color = t.hex_of(color or t.PRIMARY)
    out = []
    n = 0
    for item in items:
        if not str(item).strip():
            continue
        n += 1
        mark = f"{n}." if numbered else "&#8226;"
        indent = 18 if numbered else 14
        st = style if not numbered else ParagraphStyle("rg_num", parent=style, leftIndent=indent, bulletIndent=2)
        out.append(Paragraph(f'<bullet bulletColor="{color}">{mark}</bullet>{md_inline(item)}', st))
    return out


def callout(text, width, kind="info", title=None):
    """Soft panel with a coloured left border for notes, summaries and verdicts."""
    palette = {
        "info": (t.PRIMARY_SOFT, t.PRIMARY), "tip": (t.ACCENT_SOFT, t.ACCENT_DARK),
        "good": (t.SUCCESS_SOFT, t.SUCCESS), "warn": (t.WARNING_SOFT, t.WARNING), "bad": (t.DANGER_SOFT, t.DANGER),
    }
    bg, bar = palette.get(kind, palette["info"])
    content = []
    if title:
        content.append(Paragraph(f'<font color="{t.hex_of(bar)}"><b>{esc(title)}</b></font>', t.styles()["body"]))
    content.append(Paragraph(md_inline(text), t.styles()["body"]))
    table = Table([[content]], colWidths=[width])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("LINEBEFORE", (0, 0), (0, -1), 3, bar),
        ("LEFTPADDING", (0, 0), (-1, -1), 11), ("RIGHTPADDING", (0, 0), (-1, -1), 11),
        ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def columns(cells, width, ratios=None, gap=14):
    """Side-by-side blocks. cells: list of flowable lists (each must fit on one page)."""
    n = len(cells)
    ratios = ratios or [1 / n] * n
    usable = width - gap * (n - 1)
    data, widths = [], []
    for i, (cell, ratio) in enumerate(zip(cells, ratios)):
        if i:
            data.append("")
            widths.append(gap)
        data.append(cell)
        widths.append(usable * ratio)
    table = Table([data], colWidths=widths)
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return table


# ============================================================
# Tables
# ============================================================

def data_table(header, rows, widths, align=None, header_color=None, font_size=None, zebra=True):
    """Styled table with wrapping cells. rows hold plain strings (escaped) or ready-made flowables.
    align: per-column 'LEFT' | 'RIGHT' | 'CENTER'."""
    s = t.styles()
    align = align or ["LEFT"] * len(header)
    body_style = {"LEFT": s["cell"], "RIGHT": s["cell_right"], "CENTER": s["cell_center"]}
    head_style = {"LEFT": s["cell_head"], "RIGHT": s["cell_head_right"], "CENTER": s["cell_head_center"]}
    if font_size:
        body_style = {k: ParagraphStyle(f"{v.name}_{font_size}", parent=v, fontSize=font_size,
                                        leading=font_size * 1.32) for k, v in body_style.items()}
    data = [[Paragraph(esc(h), head_style[a]) for h, a in zip(header, align)]]
    for row in rows:
        data.append([v if isinstance(v, Flowable) else Paragraph(esc("" if v is None else v), body_style[a])
                     for v, a in zip(row, align)])
    table = Table(data, colWidths=widths, repeatRows=1)
    pad = 3.2 if font_size and font_size < 7.8 else 4.5  # dense tables (admin exports) fit more rows per page
    cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), header_color or t.PRIMARY_DARK),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), pad), ("BOTTOMPADDING", (0, 0), (-1, -1), pad),
        ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 1), (-1, -1), 0.4, t.LINE),
        ("BOX", (0, 0), (-1, -1), 0.6, t.LINE),
    ]
    if zebra and len(data) > 2:
        cmds.append(("ROWBACKGROUNDS", (0, 1), (-1, -1), [t.WHITE, t.PANEL]))
    table.setStyle(TableStyle(cmds))
    return table


def kv_table(pairs, width, key_ratio=0.3):
    """Two-column key/value list."""
    s = t.styles()
    data = [[Paragraph(f"<b>{esc(k)}</b>", s["cell"]), Paragraph(esc(v), s["cell"])] for k, v in pairs if v]
    if not data:
        return Spacer(1, 1)
    table = Table(data, colWidths=[width * key_ratio, width * (1 - key_ratio)])
    table.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, t.LINE),
        ("TEXTCOLOR", (0, 0), (0, -1), t.MUTED),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 2), ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return table


def status_cell(text):
    """Colour a status / event label: green for online/success, red for failures, amber for idle."""
    raw = clean(text).strip()
    low = raw.lower()
    color = t.TEXT
    if any(k in low for k in ("online", "login", "admin", "sign up", "ready")) and "fail" not in low:
        color = t.SUCCESS
    if any(k in low for k in ("logged out", "logout")):
        color = t.MUTED
    if any(k in low for k in ("fail", "denied", "error")):
        color = t.DANGER
    if "idle" in low:
        color = t.WARNING
    return Paragraph(f'<font color="{t.hex_of(color)}"><b>{esc(raw)}</b></font>', t.styles()["cell"])


# ============================================================
# Images
# ============================================================

def pil_image(img, max_w, max_h, max_px=1500, lossless=False):
    """PIL image -> ReportLab Image scaled to fit max_w x max_h (keeps aspect ratio).
    Previews are embedded as high-quality JPEG (passed straight through, ~10x faster); lossless=True keeps
    PNG for pages where every pixel of text matters."""
    if img.mode not in ("L", "RGB"):
        img = img.convert("RGB")
    if max(img.size) > max_px:
        img = img.copy()
        img.thumbnail((max_px, max_px), PILImage.LANCZOS)
    buf = io.BytesIO()
    if lossless:
        img.save(buf, format="PNG", compress_level=6)
    else:
        img.save(buf, format="JPEG", quality=92, subsampling=0)
    buf.seek(0)
    w, h = img.size
    scale = min(max_w / w, max_h / h)
    return Image(buf, width=w * scale, height=h * scale)


def image_grid(items, width, max_h=250, gap=10):
    """Images side by side, each framed with a caption. items: [(PIL image, caption)]."""
    s = t.styles()
    n = len(items)
    cw = (width - gap * (n - 1)) / n
    data, widths, cmds = [], [], []
    for i, (img, caption) in enumerate(items):
        if i:
            data.append("")
            widths.append(gap)
        col = len(data)
        data.append([pil_image(img, cw - 12, max_h, max_px=1100), Spacer(1, 5),
                     Paragraph(esc(caption), s["caption"])])
        widths.append(cw)
        cmds += [("BOX", (col, 0), (col, 0), 0.6, t.LINE), ("BACKGROUND", (col, 0), (col, 0), t.PANEL)]
    table = Table([data], colWidths=widths)
    table.setStyle(TableStyle(cmds + [
        ("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table


# ============================================================
# Interview kit pieces
# ============================================================

class RatingRow(Flowable):
    """'Rating ①②③④⑤  Notes ______' row for printed interview sheets."""

    def __init__(self, label="Rating", boxes=5):
        super().__init__()
        self.label, self.boxes = label, boxes

    def wrap(self, aw, ah):
        self.width = aw
        return aw, 16

    def draw(self):
        c, f = self.canv, t.fonts()
        c.setFillColor(t.MUTED)
        c.setFont(f["bold"], 7.4)
        c.drawString(0, 4, clean(self.label).upper())
        x = stringWidth(clean(self.label).upper(), f["bold"], 7.4) + 8
        c.setLineWidth(0.7)
        for i in range(1, self.boxes + 1):
            c.setStrokeColor(t.LAVENDER)
            c.setFillColor(t.WHITE)
            c.roundRect(x, 0.5, 14, 14, 3, stroke=1, fill=1)
            c.setFillColor(t.MUTED)
            c.setFont(f["regular"], 7.4)
            c.drawCentredString(x + 7, 4.6, str(i))
            x += 18
        x += 10
        c.setFillColor(t.MUTED)
        c.setFont(f["bold"], 7.4)
        c.drawString(x, 4, "NOTES")
        x += stringWidth("NOTES", f["bold"], 7.4) + 6
        c.setStrokeColor(t.LINE)
        c.setDash(1.5, 2)
        c.line(x, 3, self.width, 3)
        c.setDash()


class CheckOptions(Flowable):
    """Row of empty check boxes with labels (e.g. the final hiring recommendation)."""

    def __init__(self, options):
        super().__init__()
        self.options = [clean(o) for o in options]

    def wrap(self, aw, ah):
        self.width = aw
        return aw, 18

    def draw(self):
        c, f = self.canv, t.fonts()
        slot = self.width / len(self.options)
        for i, option in enumerate(self.options):
            x = i * slot
            c.setStrokeColor(t.PRIMARY)
            c.setFillColor(t.WHITE)
            c.setLineWidth(0.9)
            c.roundRect(x, 2, 12, 12, 2.5, stroke=1, fill=1)
            c.setFillColor(t.INK)
            c.setFont(f["regular"], 8.6)
            c.drawString(x + 17, 4.6, option)


def card(flowables, width, accent=None, background=None):
    """Bordered card with a coloured left edge (content must fit on one page)."""
    table = Table([[flowables]], colWidths=[width])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), background or t.WHITE),
        ("BOX", (0, 0), (-1, -1), 0.6, t.LINE),
        ("LINEBEFORE", (0, 0), (0, -1), 3, accent or t.PRIMARY),
        ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 9), ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))
    return table


class Tag(Flowable):
    """Small inline label pill, e.g. 'Q3 · TECHNICAL' or a chat speaker name."""

    def __init__(self, text, color, filled=True, line=False):
        super().__init__()
        self.text, self.color, self.filled, self.line = clean(text), color, filled, line
        self.keepWithNext = 1

    def wrap(self, aw, ah):
        self.width = aw
        return aw, 16

    def draw(self):
        c, f = self.canv, t.fonts()
        w = stringWidth(self.text, f["bold"], 7.4) + 16
        c.setFillColor(self.color if self.filled else t.WHITE)
        c.setStrokeColor(self.color)
        c.setLineWidth(0.8)
        c.roundRect(0, 1, w, 14, 7, stroke=1, fill=1)
        c.setFillColor(t.WHITE if self.filled else self.color)
        c.setFont(f["bold"], 7.4)
        c.drawString(8, 5.2, self.text)
        if self.line:
            c.setStrokeColor(t.LINE)
            c.setLineWidth(0.6)
            c.line(w + 8, 8, self.width, 8)


# ============================================================
# Long text & Markdown
# ============================================================

def text_block(text, max_chars=None, style="text"):
    """Long plain text (e.g. the extracted resume) as wrapped paragraphs that split across pages."""
    text = (text or "").replace("\r", "")
    truncated = False
    if max_chars and len(text) > max_chars:
        text, truncated = text[:max_chars], True
    out, blank = [], False
    for line in text.split("\n"):
        if not line.strip():
            if not blank and out:
                out.append(Spacer(1, 4))
            blank = True
            continue
        blank = False
        out.append(Paragraph(esc(line.rstrip()), t.styles()[style]))
    if truncated:
        out.append(Paragraph(f"<i>{esc('… truncated for the report. Download the full text from the app.')}</i>",
                             t.styles()["small"]))
    return out


def _code_block(code, width):
    s = t.styles()["code"]
    chars = max(20, int(width / (s.fontSize * 0.6)) - 4)
    lines = code.encode("cp1252", "replace").decode("cp1252").split("\n")
    blocks = []
    for start in range(0, len(lines), 55):
        chunk = "\n".join(lines[start:start + 55])
        pre = Preformatted(chunk, s, maxLineLength=chars, newLineChars="")
        table = Table([[pre]], colWidths=[width])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), t.PANEL), ("BOX", (0, 0), (-1, -1), 0.6, t.LINE),
            ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        blocks += [table, Spacer(1, 6)]
    return blocks


_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


def _md_cells(line):
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|") and not line.endswith("\\|"):
        line = line[:-1]
    return [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", line)]


def _column_widths(header, rows, width):
    """Share the width by content length, but never narrower than the longest single word of a column
    (so short columns like 'Week' are not broken letter by letter)."""
    f = t.fonts()
    n = len(header)
    weights, minimum = [], []
    for j in range(n):
        texts = [clean(header[j]).replace("**", "")] + [clean(r[j]).replace("**", "") for r in rows]
        weights.append(min(max(max(len(x) for x in texts), 4), 70))
        words = [w for x in texts for w in x.split()] or [""]
        longest = max(stringWidth(w, f["bold"], 8) for w in words) + 14
        minimum.append(min(longest, width * 0.3))
    widths = [width * w / sum(weights) for w in weights]
    fixed = [j for j in range(n) if widths[j] < minimum[j]]
    if fixed:
        spare = width - sum(minimum[j] for j in fixed)
        flex = [j for j in range(n) if j not in fixed]
        flex_total = sum(weights[j] for j in flex) or 1
        widths = [minimum[j] if j in fixed else max(minimum[j] * 0.6, spare * weights[j] / flex_total)
                  for j in range(n)]
        scale = width / sum(widths)
        widths = [w * scale for w in widths]
    return widths


def _md_table(header, rows, width):
    n = len(header)
    rows = [(r + [""] * n)[:n] for r in rows]
    widths = _column_widths(header, rows, width)
    s = t.styles()
    body = [[Paragraph(md_inline(v), s["cell"]) for v in r] for r in rows]
    table = data_table([clean(h).replace("**", "") for h in header], body, widths)
    return table


def markdown_flowables(text, width):
    """Render an AI answer written in Markdown: headings, paragraphs, bullet / numbered lists, tables,
    block quotes, horizontal rules and fenced code."""
    s = t.styles()
    lines = (text or "").replace("\r", "").split("\n")
    out, para_lines = [], []

    def flush():
        if para_lines:
            parts = []
            for raw in para_lines:
                hard = raw.endswith("  ") or raw.endswith("\\")
                parts.append(md_inline(raw.strip().rstrip("\\")) + ("<br/>" if hard else " "))
            out.append(Paragraph("".join(parts).strip(), s["body"]))
            para_lines.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            flush()
            fence, code = stripped[:3], []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith(fence):
                code.append(lines[i])
                i += 1
            i += 1
            out += _code_block("\n".join(code), width)
            continue
        if stripped.startswith("|") and i + 1 < len(lines) and _TABLE_SEP.match(lines[i + 1]):
            flush()
            header = _md_cells(stripped)
            i += 2
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(_md_cells(lines[i]))
                i += 1
            out += [_md_table(header, rows, width), Spacer(1, 7)]
            continue
        if not stripped:
            flush()
            i += 1
            continue
        heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading:
            flush()
            style = s["h2"] if len(heading.group(1)) <= 2 else s["h3"]
            out.append(Paragraph(md_inline(heading.group(2).strip().strip("#").strip()), style))
            i += 1
            continue
        if re.match(r"^(-{3,}|\*{3,}|_{3,})$", stripped):
            flush()
            out.append(divider(6))
            i += 1
            continue
        item = re.match(r"^(\s*)([-*+•])\s+(.*)$", line)
        number = re.match(r"^(\s*)(\d+)[.)]\s+(.*)$", line)
        if item or number:
            flush()
            m = item or number
            level = min(3, len(m.group(1).expandtabs(4)) // 2)
            style = ParagraphStyle(f"rg_md_list{level}{'n' if number else ''}", parent=s["bullet"],
                                   leftIndent=(18 if number else 14) + 14 * level, bulletIndent=2 + 14 * level)
            mark = f"{m.group(2)}." if number else "&#8226;"
            out.append(Paragraph(f'<bullet bulletColor="{t.hex_of(t.PRIMARY)}">{mark}</bullet>'
                                 f"{md_inline(m.group(3))}", style))
            i += 1
            continue
        if stripped.startswith(">"):
            flush()
            out.append(Paragraph(md_inline(stripped.lstrip("> ").strip()), s["quote"]))
            i += 1
            continue
        para_lines.append(line)
        i += 1
    flush()
    return out
