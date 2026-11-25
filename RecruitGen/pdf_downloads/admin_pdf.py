"""History and Admin Panel pages — screening history, activity log, login sessions, users and the full
admin report. Landscape pages so wide tables stay readable."""

from datetime import datetime

import pandas as pd
from reportlab.platypus import CondPageBreak, Spacer

from . import theme as t
from .components import BarChart, Donut, MetricCards, SectionTitle, columns, data_table, para, status_cell
from .document import build_pdf, content_width

MAX_ROWS = 1000


def _fmt(value):
    if isinstance(value, (pd.Timestamp, datetime)):
        return "—" if pd.isna(value) else value.strftime("%d %b %Y, %I:%M %p")
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return "—"
    if isinstance(value, float):
        return f"{value:,.2f}"
    return str(value)


def _rows(df, status_columns=(), max_len=220):
    rows = []
    for record in df.head(MAX_ROWS).itertuples(index=False):
        row = []
        for col, value in zip(df.columns, record):
            text = _fmt(value)
            if len(text) > max_len:
                text = text[:max_len - 1] + "…"
            row.append(status_cell(text) if col in status_columns else text)
        rows.append(row)
    return rows


def _limit_note(df):
    if len(df) > MAX_ROWS:
        return [Spacer(1, 4), para(f"Showing the latest {MAX_ROWS:,} of {len(df):,} rows. Export CSV for everything.",
                                   "small")]
    return []


def _counts(series, top=8):
    counts = series.dropna().astype(str).str.strip().value_counts()
    counts = counts[counts.index != ""]
    if len(counts) > top:
        counts = pd.concat([counts.head(top), pd.Series({"Other": counts.iloc[top:].sum()})])
    return list(counts.items())


def _filters_line(filters):
    return f"Filters: {filters}" if filters else "All records"


def build_history_pdf(view, prepared_by="", filters=""):
    """view columns: Date, Screened By, Candidate, File, Predicted Role, Confidence (%), Best Fit Role,
    Fit Score (%), Resume Score."""
    width = content_width(wide=True)
    conf = pd.to_numeric(view["Confidence (%)"], errors="coerce")
    fit = pd.to_numeric(view["Fit Score (%)"], errors="coerce")
    per_recruiter = _counts(view["Screened By"], top=10)
    story = [
        SectionTitle("Overview", _filters_line(filters), space_before=0),
        MetricCards([
            ("Resumes screened", f"{len(view):,}", "In this report", t.PRIMARY),
            ("Recruiters", str(view["Screened By"].nunique()), "Users who screened resumes", t.ACCENT_DARK),
            ("Avg. confidence", f"{conf.mean():.1f}%" if conf.notna().any() else "—", "Transformer V3",
             t.score_color(conf.mean()) if conf.notna().any() else t.MUTED),
            ("Avg. fit score", f"{fit.mean():.1f}%" if fit.notna().any() else "—", "CVAE role fit",
             t.score_color(fit.mean()) if fit.notna().any() else t.MUTED),
        ]),
        Spacer(1, 12),
        columns([
            [para("Predicted roles", "h3"), Donut(_counts(view["Predicted Role"]), center_label="screenings")],
            [para("Screenings per recruiter", "h3"),
             BarChart(per_recruiter, max_value=max([v for _, v in per_recruiter] + [1]), fmt="{:.0f}",
                      label_ratio=0.42)],
        ], width),
        CondPageBreak(120),
        SectionTitle("All screenings"),
        data_table(list(view.columns), _rows(view),
                   [width * w for w in (0.12, 0.15, 0.12, 0.13, 0.12, 0.08, 0.12, 0.08, 0.08)],
                   align=["LEFT", "LEFT", "LEFT", "LEFT", "LEFT", "RIGHT", "LEFT", "RIGHT", "RIGHT"], font_size=7.4),
        *_limit_note(view),
    ]
    return build_pdf(story, title="Screening History", kind="Screening History", eyebrow="Admin · All recruiters",
                     subtitle=f"{len(view):,} screened candidates", meta={"Prepared by": prepared_by}, wide=True)


def _activity_story(table, width, title="Activity log", subtitle=""):
    events = table["Event"].astype(str)
    users = table["User"].nunique()
    logins = events.str.contains("Login", case=False) & ~events.str.contains("fail", case=False)
    failed = events.str.contains("fail", case=False)
    categories = _counts(table["Category"])
    top_users = _counts(table["User"], top=8)
    return [
        SectionTitle(title, subtitle, space_before=0),
        MetricCards([
            ("Events", f"{len(table):,}", "In this report", t.PRIMARY),
            ("Users", str(users), "With recorded activity", t.ACCENT_DARK),
            ("Logins", f"{int(logins.sum()):,}", "Successful sign-ins", t.SUCCESS),
            ("Failed logins", f"{int(failed.sum()):,}", "Wrong password or wrong tab",
             t.DANGER if failed.any() else t.SUCCESS),
        ]),
        Spacer(1, 12),
        columns([
            [para("Events by category", "h3"), Donut(categories, center_label="events")],
            [para("Most active users", "h3"),
             BarChart(top_users, max_value=max([v for _, v in top_users] + [1]), fmt="{:.0f}", label_ratio=0.42)],
        ], width),
        CondPageBreak(120),
        SectionTitle("Events", "Newest first"),
        data_table(list(table.columns), _rows(table, status_columns=("Event",)),
                   [width * w for w in (0.13, 0.1, 0.11, 0.12, 0.09, 0.27, 0.12, 0.06)], font_size=7.3),
        *_limit_note(table),
    ]


def build_activity_pdf(table, prepared_by="", filters=""):
    """table columns: Time, User, Name, Event, Category, Details, Device / IP, Session."""
    story = _activity_story(table, content_width(wide=True), subtitle=_filters_line(filters))
    return build_pdf(story, title="Activity Log", kind="Activity Log", eyebrow="Admin · Activity monitoring",
                     subtitle=f"{len(table):,} events", meta={"Prepared by": prepared_by}, wide=True)


def _sessions_story(table, width, subtitle=""):
    status = table["Status"].astype(str)
    return [
        SectionTitle("Login sessions", subtitle, space_before=0),
        MetricCards([
            ("Sessions", f"{len(table):,}", "Logins recorded", t.PRIMARY),
            ("Online now", str(int(status.str.contains("Online").sum())), "Active in the last 10 min", t.SUCCESS),
            ("Logged out", str(int(status.str.contains("Logged out").sum())), "Pressed Logout", t.INK),
            ("Idle / closed", str(int(status.str.contains("Idle").sum())), "Closed the tab without logout",
             t.WARNING),
        ]),
        Spacer(1, 10),
        data_table(list(table.columns), _rows(table, status_columns=("Status",)),
                   [width * w for w in (0.1, 0.12, 0.11, 0.13, 0.13, 0.13, 0.08, 0.14, 0.06)], font_size=7.4),
        *_limit_note(table),
    ]


def build_sessions_pdf(table, prepared_by="", filters=""):
    """table columns: User, Name, Status, Logged in, Logged out, Last activity, Duration, Device / IP, Session."""
    story = _sessions_story(table, content_width(wide=True), _filters_line(filters))
    return build_pdf(story, title="Login & Logout Sessions", kind="Session Report", eyebrow="Admin · Sessions",
                     subtitle=f"{len(table):,} sessions", meta={"Prepared by": prepared_by}, wide=True)


def _users_story(table, width):
    logins = pd.to_numeric(table["Logins"], errors="coerce").fillna(0)
    per_user = sorted(zip(table["Username"], logins), key=lambda kv: kv[1], reverse=True)[:10]
    return [
        SectionTitle("Users", "Every account with its activity totals", space_before=0),
        MetricCards([
            ("Users", str(len(table)), "Registered accounts", t.PRIMARY),
            ("Admins", str(int(table["Role"].astype(str).str.contains("Admin").sum())), "Administrator accounts",
             t.GOLD),
            ("Online now", str(int(table["Status"].astype(str).str.contains("Online").sum())), "Active users",
             t.SUCCESS),
            ("Total logins", f"{int(logins.sum()):,}", "All sessions", t.ACCENT_DARK),
        ]),
        Spacer(1, 10),
        data_table(list(table.columns), _rows(table, status_columns=("Status", "Role")),
                   [width * w for w in (0.12, 0.11, 0.17, 0.08, 0.08, 0.1, 0.14, 0.06, 0.07, 0.07)],
                   align=["LEFT"] * 7 + ["RIGHT"] * 3, font_size=7.4),
        Spacer(1, 12),
        para("Logins per user", "h3"),
        BarChart(per_user, max_value=max([v for _, v in per_user] + [1]), fmt="{:.0f}", label_ratio=0.25),
    ]


def build_users_pdf(table, prepared_by=""):
    """table columns: Name, Username, Email, Role, Status, Joined, Last login, Logins, Screenings, Events."""
    story = _users_story(table, content_width(wide=True))
    return build_pdf(story, title="User Accounts", kind="User Report", eyebrow="Admin · Users",
                     subtitle=f"{len(table)} accounts", meta={"Prepared by": prepared_by}, wide=True)


def build_admin_report_pdf(users, sessions, activity, prepared_by=""):
    """Complete admin report: users, the latest sessions and the latest activity."""
    width = content_width(wide=True)
    story = _users_story(users, width)
    story += [CondPageBreak(400)] + _sessions_story(sessions.head(150), width, "Latest 150 sessions")
    story += [CondPageBreak(400)] + _activity_story(activity.head(400), width, "Activity", "Latest 400 events")
    return build_pdf(story, title="Platform Activity Report", kind="Admin Report", eyebrow="Admin · Full report",
                     subtitle=f"{len(users)} users · {len(sessions):,} sessions · {len(activity):,} events",
                     meta={"Prepared by": prepared_by}, wide=True)
