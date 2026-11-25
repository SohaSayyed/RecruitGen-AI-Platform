"""
Bulk Resume Processing — Excel report (openpyxl)
------------------------------------------------
    Dashboard            title, batch details, KPI cards, status doughnut + bar chart, processing-time and enhancement
                         statistics. Every number is an Excel formula over the Candidate Profiles sheet, so the
                         dashboard updates when the data changes.
    Candidate Profiles   one row per uploaded resume (failed ones included) — Excel table, banded rows, filters,
                         frozen header, status colours. Unavailable fields read "Not Found"; nothing is invented.
    Extracted Text       complete extracted text per resume (texts over Excel's 32,767-character cell limit are split
                         across "Part" rows, never truncated), wrapped, with page count, language and status.
    Errors & Warnings    failed and partially extracted resumes with the failing stage, details, retry status and a
                         recommended action — or a single "No errors found" message.

Theme: navy / teal / white / subtle grey. Text is written as plain strings (never evaluated as formulas).
"""

import io
import re
import statistics
from datetime import datetime

from openpyxl import Workbook
from openpyxl.chart import BarChart, DoughnutChart, Reference
from openpyxl.chart.label import DataLabel, DataLabelList
from openpyxl.chart.series import DataPoint
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo

# ---------------- theme ----------------
NAVY, NAVY_DARK, TEAL, TEAL_LIGHT = "1F3A5F", "16304F", "0F9D9A", "E3F4F3"
GRAY_BG, GRAY_LINE, GRAY_TEXT, INK = "F4F6F9", "D9DEE5", "6B7280", "1F2937"
STATUS_STYLE = {"Success": ("0B7A6F", "DDF3EF"), "Partial Success": ("9A6B00", "FDF2D6"), "Failed": ("A4262C", "FBE3E4")}
FONT = "Calibri"
NOT_FOUND = "Not Found"
CELL_LIMIT = 32_000                      # below Excel's 32,767 hard limit

PROFILE_COLUMNS = [  # (header, width, wrap, number format)
    ("Resume ID", 11, False, None), ("Filename", 30, False, None), ("Candidate Name", 22, False, None),
    ("Email", 30, False, None), ("Phone", 18, False, None), ("Skills", 42, True, None),
    ("Experience (Years)", 12, False, "General"), ("Education", 36, True, None), ("Certifications", 34, True, None),
    ("Previous Job Title", 24, False, None), ("Previous Company", 24, False, None), ("OCR Language", 11, False, None),
    ("Status", 15, False, None), ("Enhancement", 13, False, None), ("Processing Time (s)", 13, False, "0.00"),
    ("Pages", 9, False, None),
]
COL = {h: get_column_letter(i) for i, (h, *_rest) in enumerate(PROFILE_COLUMNS, 1)}
PROFILES = "'Candidate Profiles'"

def _clean(value):
    """Same cleaning as the CSV export (core.bulk.clean_text), so both files hold identical text."""
    from core.bulk import clean_text

    return clean_text(value) if isinstance(value, str) else value


def _text(cell, value):
    """Write text as a plain string — a resume line starting with '=' must never become a formula."""
    cell.value = _clean(value)
    if isinstance(cell.value, str):
        cell.data_type = "s"


def _font(size=10, bold=False, color=INK, italic=False):
    return Font(name=FONT, size=size, bold=bold, color=color, italic=italic)


def _fill(color):
    return PatternFill("solid", start_color=color, end_color=color)


THIN = Side(style="thin", color=GRAY_LINE)
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
HEADER_FONT = _font(10, True, "FFFFFF")
HEADER_FILL = _fill(NAVY)


def _header_row(ws, headers, row=1):
    for i, h in enumerate(headers, 1):
        c = ws.cell(row=row, column=i, value=h)
        c.font, c.fill = HEADER_FONT, HEADER_FILL
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = Border(bottom=Side(style="medium", color=TEAL))
    ws.row_dimensions[row].height = 30


def _print_setup(ws, landscape=True, title_rows=None):
    ws.page_setup.orientation = "landscape" if landscape else "portrait"
    ws.page_setup.paperSize = ws.PAPERSIZE_A4
    ws.page_setup.fitToWidth, ws.page_setup.fitToHeight = 1, 0
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.page_margins.left = ws.page_margins.right = 0.4
    ws.page_margins.top = ws.page_margins.bottom = 0.5
    ws.print_options.horizontalCentered = True
    if title_rows:
        ws.print_title_rows = title_rows
    ws.oddFooter.center.text = "RecruitGen · Bulk Resume Processing Report · Page &P of &N"
    ws.oddFooter.center.size = 8


def _add_table(ws, name, ncols, nrows):
    ref = f"A1:{get_column_letter(ncols)}{max(nrows, 1) + 1}"
    table = Table(displayName=name, ref=ref)
    table.tableStyleInfo = TableStyleInfo(name="TableStyleLight1", showRowStripes=False, showColumnStripes=False)
    ws.add_table(table)


def _band(ws, row, ncols):
    if row % 2 == 1:
        for c in range(1, ncols + 1):
            ws.cell(row=row, column=c).fill = _fill(GRAY_BG)


def _status_rule(ws, col_letter, first, last):
    for status, (font_color, fill_color) in STATUS_STYLE.items():
        ws.conditional_formatting.add(
            f"{col_letter}{first}:{col_letter}{last}",
            CellIsRule(operator="equal", formula=[f'"{status}"'], font=Font(name=FONT, bold=True, color=font_color),
                       fill=_fill(fill_color)))


# ============================================================
# data preparation
# ============================================================

def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _profile_values(row, fields):
    def found(key):
        v = (fields.get(key) or "").strip()
        return v if v else NOT_FOUND

    pages = row.get("pages_processed") or ""
    exp = _num(fields.get("experience_years"))
    return [row["resume_id"], row["filename"], found("candidate_name"), found("email"), found("phone"),
            found("skills"), exp if exp is not None else NOT_FOUND, found("education"), found("certifications"),
            found("previous_job_title"), found("previous_company"), row.get("ocr_language") or NOT_FOUND,
            row.get("status") or "Failed", row.get("enhancement_method") or "None",
            _num(row.get("processing_time_seconds")) or 0.0, pages or "—"]


def _stage_and_action(row, item):
    """Map an error message to the failing stage, retry status and a recommended action."""
    msg = (row.get("error_message") or "").lower()
    retries = (item or {}).get("retries", 0)
    validation = bool((item or {}).get("error"))
    if row["status"] == "Partial Success":
        stage = "Text extraction (partial)"
        if "page" in msg and ":" in msg:
            stage = "Page processing"
        action = ("Upload a clearer scan or the original digital PDF; review the extracted text before using it."
                  if "confidence" in msg or "characters" in msg else
                  "Check the pages listed in the details; re-scan or re-upload them if they matter.")
        retry = "Not required — text was extracted"
    elif msg.startswith("not processed"):
        stage, retry = "Not processed (batch stopped)", "Resume available"
        action = "Open the batch and click Resume / retry to process the remaining files."
    elif "unsupported format" in msg:
        stage, retry, action = "Upload validation", "Not retryable", "Convert the file to PDF, PNG or JPG and upload it again."
    elif "unsafe path" in msg or "encrypted" in msg or "larger than" in msg or "compression" in msg:
        stage, retry, action = "Upload validation", "Not retryable", "Re-create the ZIP with a normal, unencrypted copy of this file."
    elif "corrupted" in msg or "empty" in msg or "content does not match" in msg:
        stage, retry, action = "Upload validation", "Not retryable", "The file is damaged — export or download it again and re-upload."
    elif "password" in msg:
        stage, retry, action = "File reading", "Retry available" if not retries else f"Retried {retries}× — still failing", \
            "Remove the PDF password and upload the file again."
    elif "could not read the file" in msg:
        stage, action = "File reading", "Open the file to check it; re-save as PDF / PNG and upload again."
        retry = "Not retryable" if validation else ("Retry available" if not retries else f"Retried {retries}× — still failing")
    else:
        stage, action = "Text extraction (OCR)", ("The page has no readable text — upload a sharper scan or the "
                                                  "original document.")
        retry = "Retry available" if not retries else f"Retried {retries}× — still failing"
    return stage, retry, action


def _chunks(text, size=CELL_LIMIT):
    """Split long text on whitespace into pieces that fit an Excel cell — nothing is dropped."""
    if len(text) <= size:
        return [text]
    parts, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            cut = text.rfind(" ", start + size // 2, end)
            end = cut + 1 if cut > start else end
        parts.append(text[start:end])
        start = end
    return parts


# ============================================================
# sheets
# ============================================================

def _profiles_sheet(wb, rows, structured):
    ws = wb.create_sheet("Candidate Profiles")
    ws.sheet_properties.tabColor = TEAL
    headers = [h for h, *_ in PROFILE_COLUMNS]
    _header_row(ws, headers)
    by_id = {s["resume_id"]: s for s in structured}
    for r_i, row in enumerate(rows, 2):
        values = _profile_values(row, by_id.get(row["resume_id"], {}))
        for c_i, ((_, _, wrap, fmt), value) in enumerate(zip(PROFILE_COLUMNS, values), 1):
            cell = ws.cell(row=r_i, column=c_i)
            _text(cell, value) if isinstance(value, str) else setattr(cell, "value", value)
            cell.font = _font(10, color=GRAY_TEXT, italic=True) if value == NOT_FOUND else _font(10)
            cell.alignment = Alignment(vertical="top", wrap_text=wrap,
                                       horizontal="center" if c_i in (1, 7, 12, 13, 14, 15, 16) else "left")
            if fmt and not isinstance(value, str):
                cell.number_format = fmt
        _band(ws, r_i, len(headers))
        longest = max(len(str(values[i])) / max(PROFILE_COLUMNS[i][1] - 2, 1) for i in (5, 7, 8))
        ws.row_dimensions[r_i].height = min(90, max(18, 15 * (int(longest) + 1)))
    for i, (_, width, _, _) in enumerate(PROFILE_COLUMNS, 1):
        ws.column_dimensions[get_column_letter(i)].width = width
    _add_table(ws, "CandidateProfiles", len(headers), len(rows))
    _status_rule(ws, COL["Status"], 2, max(2, len(rows) + 1))
    ws.freeze_panes = "C2"                                    # header row + resume id / filename stay visible
    _print_setup(ws, title_rows="1:1")
    return ws


def _text_sheet(wb, rows):
    ws = wb.create_sheet("Extracted Text")
    ws.sheet_properties.tabColor = "8A94A6"
    headers = ["Resume ID", "Filename", "Status", "Language", "Pages", "Characters", "Part", "Extracted Text",
               "Processing Notes"]
    widths = [11, 30, 15, 11, 9, 12, 8, 110, 36]
    _header_row(ws, headers)
    r_i = 2
    for row in rows:
        text = row.get("extracted_text") or ""
        parts = _chunks(text) if text else [""]
        for p_i, part in enumerate(parts, 1):
            values = [row["resume_id"], row["filename"], row.get("status") or "Failed", row.get("ocr_language") or "",
                      row.get("pages_processed") or "", len(text), f"{p_i}/{len(parts)}", part,
                      row.get("notes") or ""]
            for c_i, value in enumerate(values, 1):
                cell = ws.cell(row=r_i, column=c_i)
                _text(cell, value) if isinstance(value, str) else setattr(cell, "value", value)
                cell.font = _font(10)
                cell.alignment = Alignment(vertical="top", wrap_text=c_i in (8, 9),
                                           horizontal="center" if c_i in (1, 3, 4, 5, 6, 7) else "left")
            ws.cell(row=r_i, column=6).number_format = "#,##0"
            lines = sum(max(1, len(l) // 120 + 1) for l in part.splitlines()) if part else 1
            ws.row_dimensions[r_i].height = min(409, max(18, 13.5 * lines))     # 409 pt = Excel's maximum
            _band(ws, r_i, len(headers))
            r_i += 1
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    _add_table(ws, "ExtractedText", len(headers), r_i - 2)
    _status_rule(ws, "C", 2, max(2, r_i - 1))
    ws.freeze_panes = "C2"
    _print_setup(ws, title_rows="1:1")
    return ws


def _errors_sheet(wb, rows, items):
    ws = wb.create_sheet("Errors & Warnings")
    headers = ["Resume ID", "Filename", "Severity", "Failed Stage", "Error Details", "Retry Status",
               "Recommended Action"]
    widths = [11, 30, 12, 24, 50, 30, 54]
    _header_row(ws, headers)
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    item_by_id = {it["resume_id"]: it for it in items or []}
    problems = [r for r in rows if r.get("status") in ("Failed", "Partial Success")]
    if not problems:
        ws.sheet_properties.tabColor = TEAL
        ws.merge_cells("A2:G2")
        c = ws["A2"]
        c.value = f"✓  No errors found — all {len(rows)} resumes were processed without errors or warnings."
        c.font, c.fill = _font(12, True, "0B7A6F"), _fill(TEAL_LIGHT)
        c.alignment = Alignment(horizontal="center", vertical="center")
        ws.row_dimensions[2].height = 34
        ws.auto_filter.ref = "A1:G1"
    else:
        ws.sheet_properties.tabColor = "A4262C" if any(r["status"] == "Failed" for r in problems) else "C99A2E"
        for r_i, row in enumerate(problems, 2):
            stage, retry, action = _stage_and_action(row, item_by_id.get(row["resume_id"]))
            severity = "Error" if row["status"] == "Failed" else "Warning"
            values = [row["resume_id"], row["filename"], severity, stage, row.get("error_message") or "",
                      retry, action]
            font_color, fill_color = STATUS_STYLE["Failed" if severity == "Error" else "Partial Success"]
            for c_i, value in enumerate(values, 1):
                cell = ws.cell(row=r_i, column=c_i)
                _text(cell, value)
                cell.font = _font(10, bold=c_i == 3, color=font_color if c_i == 3 else INK)
                cell.fill = _fill(fill_color)
                cell.border = Border(bottom=THIN)
                cell.alignment = Alignment(vertical="top", wrap_text=c_i in (5, 6, 7),
                                           horizontal="center" if c_i in (1, 3) else "left")
            ws.row_dimensions[r_i].height = max(30, 15 * (max(len(values[4]) // 50, len(values[6]) // 54) + 1))
        _add_table(ws, "ErrorsWarnings", len(headers), len(problems))
    ws.freeze_panes = "A2"
    _print_setup(ws, title_rows="1:1")
    return ws


def _card(ws, col, row, label, formula, number_format, accent, caption):
    """KPI card: 2 merged columns × 4 rows (label · big value · caption), white with a coloured top bar."""
    c1, c2 = get_column_letter(col), get_column_letter(col + 1)
    for r in range(row, row + 4):
        for c in (col, col + 1):
            cell = ws.cell(row=r, column=c)
            cell.fill = _fill("FFFFFF")
            cell.border = Border(left=THIN if c == col else None, right=THIN if c == col + 1 else None,
                                 top=Side(style="thick", color=accent) if r == row else None,
                                 bottom=THIN if r == row + 3 else None)
    ws.merge_cells(f"{c1}{row}:{c2}{row}")
    ws.merge_cells(f"{c1}{row + 1}:{c2}{row + 2}")
    ws.merge_cells(f"{c1}{row + 3}:{c2}{row + 3}")
    lab = ws[f"{c1}{row}"]
    lab.value, lab.font = label.upper(), _font(9, True, GRAY_TEXT)
    lab.alignment = Alignment(horizontal="center", vertical="bottom")
    val = ws[f"{c1}{row + 1}"]
    val.value, val.number_format = formula, number_format
    val.font = _font(28, True, accent)
    val.alignment = Alignment(horizontal="center", vertical="center")
    cap = ws[f"{c1}{row + 3}"]
    cap.value, cap.font = caption, _font(9, color=GRAY_TEXT, italic=True)
    cap.alignment = Alignment(horizontal="center", vertical="top")


def _section(ws, cell_range, title):
    ws.merge_cells(cell_range)
    c = ws[cell_range.split(":")[0]]
    c.value, c.font = title, _font(12, True, NAVY)
    c.alignment = Alignment(vertical="bottom")
    first, last = cell_range.split(":")
    row = int(re.sub(r"\D", "", first))
    for col in range(ws[first].column, ws[last].column + 1):
        ws.cell(row=row, column=col).border = Border(bottom=Side(style="medium", color=TEAL))


def _kv_table(ws, top, left, title, pairs, label_span=3, value_span=2):
    """Statistics block: merged label columns | merged value columns (formula)."""
    last = left + label_span + value_span - 1
    ws.cell(row=top, column=left, value=title).font = _font(10, True, "FFFFFF")
    for c in range(left, last + 1):
        ws.cell(row=top, column=c).fill = _fill(NAVY)
    ws.merge_cells(start_row=top, start_column=left, end_row=top, end_column=last)
    ws.cell(row=top, column=left).alignment = Alignment(horizontal="left", vertical="center", indent=1)
    ws.row_dimensions[top].height = 21
    for i, (label, formula, fmt) in enumerate(pairs, 1):
        r = top + i
        ws.merge_cells(start_row=r, start_column=left, end_row=r, end_column=left + label_span - 1)
        lc = ws.cell(row=r, column=left, value=label)
        lc.font, lc.alignment = _font(10, color=INK), Alignment(indent=1, vertical="center")
        ws.merge_cells(start_row=r, start_column=left + label_span, end_row=r, end_column=last)
        vc = ws.cell(row=r, column=left + label_span, value=formula)
        vc.font, vc.number_format = _font(10, True, NAVY), fmt
        vc.alignment = Alignment(horizontal="right", vertical="center", indent=1)
        for c in range(left, last + 1):
            cell = ws.cell(row=r, column=c)
            cell.fill = _fill(GRAY_BG if i % 2 else "FFFFFF")
            cell.border = Border(bottom=THIN)
        ws.row_dimensions[r].height = 19


def _dashboard(wb, job, rows):
    ws = wb.active
    ws.title = "Dashboard"
    ws.sheet_view.showGridLines = False
    ws.sheet_view.zoomScale = 90
    ws.sheet_properties.tabColor = NAVY
    widths = {"A": 2.5}
    card_cols = [2, 5, 8, 11, 14]                          # B, E, H, K, N  (each card = 2 columns)
    for c in range(2, 17):
        widths[get_column_letter(c)] = 2.5 if c in (4, 7, 10, 13) else 12.5
    widths["P"] = 2.5
    for k, v in widths.items():
        ws.column_dimensions[k].width = v

    # ---- title band ----
    for r in (2, 3):
        for c in range(2, 16):
            ws.cell(row=r, column=c).fill = _fill(NAVY)
    ws.merge_cells("B2:O2")
    ws["B2"].value = "RecruitGen — Bulk Resume Processing Report"
    ws["B2"].font = _font(22, True, "FFFFFF")
    ws["B2"].alignment = Alignment(vertical="center", indent=1)
    ws.row_dimensions[2].height = 42
    ws.merge_cells("B3:O3")
    ws["B3"].value = "Resume enhancement · OCR text extraction · candidate profile summary"
    ws["B3"].font = _font(11, color="BFD3E6")
    ws["B3"].alignment = Alignment(vertical="top", indent=1)
    ws.row_dimensions[3].height = 24
    for c in range(2, 16):
        ws.cell(row=4, column=c).fill = _fill(TEAL)
    ws.row_dimensions[4].height = 4

    # ---- batch details ----
    opts = (job or {}).get("opts", {})
    state = (job or {}).get("state", "finished")
    started, finished = (job or {}).get("started"), (job or {}).get("finished")
    wall = f"{(finished - started) / 60:.1f} min" if started and finished else "—"
    details = [
        ("Batch ID", (job or {}).get("id", "—")),
        ("Source file", (job or {}).get("zip_name", "—")),
        ("Report generated", datetime.now().strftime("%d %b %Y, %H:%M")),
        ("Processing status", {"finished": "Completed", "running": "In progress", "queued": "Queued",
                               "interrupted": "Interrupted", "stopped": "Stopped by user",
                               "error": "Stopped with an error"}.get(state, state)),
        ("Enhancement", {"diffusion": "Diffusion model", "autoencoder": "Autoencoder", "none": "None"}.get(
            opts.get("enhancement", ""), "—") + (" · quality guard on" if opts.get("guard") else "")),
        ("OCR engine", {"local": "Local OCR (RapidOCR)", "ai_vision": "AI vision OCR"}.get(opts.get("ocr_engine"), "—")),
        ("Total run time", wall),
    ]
    for i, (label, value) in enumerate(details):
        r, left = 6 + i % 4, 2 if i < 4 else 9
        lc = ws.cell(row=r, column=left, value=label)
        lc.font = _font(10, True, GRAY_TEXT)
        ws.merge_cells(start_row=r, start_column=left, end_row=r, end_column=left + 1)
        ws.merge_cells(start_row=r, start_column=left + 2, end_row=r, end_column=left + 5)
        vc = ws.cell(row=r, column=left + 2)
        _text(vc, str(value))
        vc.font = _font(10, True, NAVY if label != "Processing status" else "0B7A6F" if state == "finished" else "9A6B00")
    for r in range(6, 10):
        ws.row_dimensions[r].height = 18

    # ---- KPI cards ----
    _section(ws, "B11:O11", "Key metrics")
    ws.row_dimensions[11].height = 24
    m = PROFILES
    total = f"COUNTA({m}!$A:$A)-1"
    cards = [
        ("Total resumes", f"={total}", "#,##0", NAVY, "uploaded in this batch"),
        ("Successful", f'=COUNTIF({m}!${COL["Status"]}:${COL["Status"]},"Success")', "#,##0", "0B7A6F",
         "full text extracted"),
        ("Partial success", f'=COUNTIF({m}!${COL["Status"]}:${COL["Status"]},"Partial Success")', "#,##0", "9A6B00",
         "some text extracted"),
        ("Failed", f'=COUNTIF({m}!${COL["Status"]}:${COL["Status"]},"Failed")', "#,##0", "A4262C", "no usable text"),
        ("Success rate", "=IFERROR(E13/B13,0)", "0.0%", TEAL, "successful ÷ total"),
    ]
    for col, card in zip(card_cols, cards):
        _card(ws, col, 12, *card)
    ws.row_dimensions[12].height = 20
    ws.row_dimensions[13].height = 26
    ws.row_dimensions[14].height = 22
    ws.row_dimensions[15].height = 18

    # ---- status distribution: table + charts ----
    _section(ws, "B17:O17", "Processing status distribution")
    ws.row_dimensions[17].height = 24
    for text, c in [("Status", 2), ("Resumes", 3), ("Share", 4)]:
        cell = ws.cell(row=19, column=c, value=text)
        cell.font, cell.fill = HEADER_FONT, HEADER_FILL
        cell.alignment = Alignment(horizontal="center")
    ws.cell(row=19, column=5).fill = HEADER_FILL
    ws.merge_cells("D19:E19")
    for i, status in enumerate(["Success", "Partial Success", "Failed"]):
        r = 20 + i
        font_color, fill_color = STATUS_STYLE[status]
        sc = ws.cell(row=r, column=2, value=status)
        sc.font, sc.fill = _font(10, True, font_color), _fill(fill_color)
        cc = ws.cell(row=r, column=3, value=f'=COUNTIF({m}!${COL["Status"]}:${COL["Status"]},B{r})')
        cc.font, cc.alignment, cc.number_format = _font(10, True), Alignment(horizontal="center"), "#,##0"
        ws.merge_cells(start_row=r, start_column=4, end_row=r, end_column=5)
        pc = ws.cell(row=r, column=4, value=f"=IFERROR(C{r}/SUM($C$20:$C$22),0)")
        pc.font, pc.alignment, pc.number_format = _font(10), Alignment(horizontal="center"), "0.0%"
        for c in range(2, 6):
            ws.cell(row=r, column=c).border = Border(bottom=THIN)
    tot = ws.cell(row=23, column=2, value="Total")
    tot.font = _font(10, True, NAVY)
    t2 = ws.cell(row=23, column=3, value="=SUM(C20:C22)")
    t2.font, t2.alignment, t2.number_format = _font(10, True, NAVY), Alignment(horizontal="center"), "#,##0"
    for c in range(2, 6):
        ws.cell(row=23, column=c).border = Border(top=Side(style="medium", color=NAVY))

    labels = Reference(ws, min_col=2, min_row=20, max_row=22)
    values = Reference(ws, min_col=3, min_row=19, max_row=22)
    colors = ["0B7A6F", "D9A21B", "C0392B"]
    dn = DoughnutChart(holeSize=58)
    dn.title = "Status distribution"
    dn.add_data(values, titles_from_data=True)
    dn.set_categories(labels)
    for i, color in enumerate(colors):
        pt = DataPoint(idx=i)
        pt.graphicalProperties.solidFill = color
        pt.graphicalProperties.line.solidFill = "FFFFFF"
        dn.series[0].dPt.append(pt)
    dn.dataLabels = DataLabelList(showPercent=True, showVal=False, showCatName=False, showSerName=False,
                                  showLegendKey=False, showLeaderLines=False)
    counts = [sum(r.get("status") == s for r in rows) for s in ("Success", "Partial Success", "Failed")]
    dn.dataLabels.dLbl = [DataLabel(idx=i, showPercent=False, showVal=False, showCatName=False, showSerName=False,
                                    showLegendKey=False) for i, n in enumerate(counts) if n == 0]   # no "0%" labels
    dn.legend.position = "b"
    dn.height, dn.width = 7.4, 8.0
    ws.add_chart(dn, "F18")

    bar = BarChart()
    bar.type, bar.style = "col", 10
    bar.title = "Resumes by status"
    bar.add_data(values, titles_from_data=True)
    bar.set_categories(labels)
    bar.legend = None
    bar.y_axis.majorGridlines = None
    bar.y_axis.number_format = "0"
    bar.y_axis.delete = False
    bar.x_axis.delete = False
    bar.gapWidth = 70
    for i, color in enumerate(colors):
        pt = DataPoint(idx=i)
        pt.graphicalProperties.solidFill = color
        pt.graphicalProperties.line.solidFill = color
        bar.series[0].dPt.append(pt)
    bar.dataLabels = DataLabelList(showVal=True, showCatName=False, showSerName=False, showLegendKey=False,
                                   showPercent=False)
    bar.height, bar.width = 7.4, 9.8
    ws.add_chart(bar, "K18")

    # ---- processing statistics ----
    _section(ws, "B34:O34", "Processing time & enhancement")
    ws.row_dimensions[34].height = 24
    t = f"{m}!${COL['Processing Time (s)']}:${COL['Processing Time (s)']}"
    e = f"{m}!${COL['Enhancement']}:${COL['Enhancement']}"
    _kv_table(ws, 36, 2, "Processing time (seconds)", [
        ("Total processing time", f"=SUM({t})", '#,##0.0" s"'),
        ("Total (hh:mm:ss)", f"=SUM({t})/86400", "[h]:mm:ss"),
        ("Average per resume", f"=IFERROR(AVERAGE({t}),0)", '0.00" s"'),
        ("Median per resume", f"=IFERROR(MEDIAN({t}),0)", '0.00" s"'),
        ("Longest resume", f"=IFERROR(MAX({t}),0)", '0.00" s"'),
        ("Fastest resume", f"=IFERROR(MIN({t}),0)", '0.00" s"'),
    ], label_span=3, value_span=2)                      # B:D label · E:F value
    _kv_table(ws, 36, 8, "Image enhancement applied", [
        ("Diffusion", f'=COUNTIF({e},"Diffusion")', "#,##0"),
        ("Autoencoder", f'=COUNTIF({e},"Autoencoder")', "#,##0"),
        ("None (text PDF or original kept)", f'=COUNTIF({e},"None")', "#,##0"),
        ("Share enhanced", f'=IFERROR((COUNTIF({e},"Diffusion")+COUNTIF({e},"Autoencoder"))/({total}),0)', "0.0%"),
    ], label_span=4, value_span=4)                      # H:K label · L:O value

    # ---- navigation + note ----
    _section(ws, "B44:O44", "Workbook contents")
    links = [("Candidate Profiles", "one row per resume · contact, skills, education, experience, status"),
             ("Extracted Text", "complete OCR / PDF text of every resume"),
             ("Errors & Warnings", "failed and partially extracted resumes with recommended actions")]
    for i, (sheet, desc) in enumerate(links):
        r = 45 + i
        lc = ws.cell(row=r, column=2, value=f"→  {sheet}")
        lc.hyperlink = f"#'{sheet}'!A1"
        lc.font = Font(name=FONT, size=10, bold=True, color=TEAL, underline="single")
        ws.merge_cells(start_row=r, start_column=2, end_row=r, end_column=4)
        dc = ws.cell(row=r, column=5, value=desc)
        dc.font = _font(10, color=GRAY_TEXT)
        ws.merge_cells(start_row=r, start_column=5, end_row=r, end_column=15)
    note = ws.cell(row=49, column=2, value="All dashboard figures are live Excel formulas over the Candidate Profiles "
                                            "sheet — they update automatically if the data is edited. Unavailable "
                                            "candidate details are shown as “Not Found”; nothing is estimated.")
    note.font = _font(9, color=GRAY_TEXT, italic=True)
    note.alignment = Alignment(wrap_text=True, vertical="top")
    ws.merge_cells("B49:O50")

    ws.print_area = "B2:O50"
    _print_setup(ws)
    ws.page_setup.fitToHeight = 1
    return ws


# ============================================================
# public
# ============================================================

def build_report(rows, structured, job=None):
    """Return the .xlsx bytes. rows / structured: one entry per uploaded resume, in upload order."""
    wb = Workbook()
    wb.properties.title = "RecruitGen — Bulk Resume Processing Report"
    wb.properties.creator = "RecruitGen"
    wb.properties.subject = f"Batch {(job or {}).get('id', '')}"
    _dashboard(wb, job, rows)
    _profiles_sheet(wb, rows, structured)
    _text_sheet(wb, rows)
    _errors_sheet(wb, rows, (job or {}).get("items"))
    wb.active = 0
    wb.calculation.fullCalcOnLoad = True           # Excel computes every formula when the file opens
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def summary(rows):
    """Python-side values of the dashboard metrics (used to validate the workbook formulas)."""
    times = [_num(r.get("processing_time_seconds")) or 0.0 for r in rows]
    n = len(rows)
    count = {s: sum(r.get("status") == s for r in rows) for s in ("Success", "Partial Success", "Failed")}
    return {"total": n, **count, "success_rate": count["Success"] / n if n else 0,
            "time_total": sum(times), "time_avg": statistics.mean(times) if times else 0,
            "time_median": statistics.median(times) if times else 0, "time_max": max(times, default=0),
            "time_min": min(times, default=0)}
