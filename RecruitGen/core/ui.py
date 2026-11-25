"""Shared styling and UI components."""

import html

import streamlit as st

PRIMARY = "#7C5CFC"
ACCENT = "#22D3EE"
SUCCESS = "#34D399"
WARNING = "#FBBF24"
DANGER = "#F87171"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea { font-family: 'Inter', sans-serif; }
.stApp {
    background:
        radial-gradient(1200px 600px at 10% -10%, rgba(124,92,252,0.18), transparent 60%),
        radial-gradient(900px 500px at 110% 10%, rgba(34,211,238,0.10), transparent 60%),
        #080D16;
}
/* Start content below Streamlit's fixed top bar (~2.9rem) so the first card is never cut off. */
.block-container, [data-testid="stAppViewBlockContainer"], [data-testid="stMainBlockContainer"] {
    padding-top: 4.25rem; padding-bottom: 3rem; max-width: 1300px; }
/* Top bar blends into the page (frosted glass) instead of a hard black band. */
header[data-testid="stHeader"] { background: rgba(8,13,22,0.55); backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px); }
#MainMenu, footer { visibility: hidden; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0E1424 0%, #0A0F1C 100%);
    border-right: 1px solid rgba(124,92,252,0.18);
}
section[data-testid="stSidebar"] .block-container, section[data-testid="stSidebar"] > div { padding-top: 0.6rem; }

/* ---------- sidebar brand + profile ---------- */
.rg-side-brand { display: flex; align-items: center; gap: 12px; padding: 8px 2px 16px; }
.rg-side-brand .mark { width: 42px; height: 42px; border-radius: 13px; display: grid; place-items: center;
    font-size: 1.35rem; background: linear-gradient(135deg, #7C5CFC, #22D3EE);
    box-shadow: 0 6px 20px rgba(124,92,252,0.35); }
.rg-side-brand .name { font-size: 1.3rem; font-weight: 800; letter-spacing: -0.3px; line-height: 1.1;
    background: linear-gradient(90deg, #FFFFFF, #C4B5FD); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.rg-side-brand .tag { color: #7C8BA8; font-size: 0.74rem; letter-spacing: 0.4px; }

.rg-profile { display: flex; align-items: center; gap: 12px; padding: 14px; border-radius: 16px;
    background: linear-gradient(135deg, rgba(124,92,252,0.16), rgba(34,211,238,0.06));
    border: 1px solid rgba(124,92,252,0.30); margin-bottom: 6px; }
.rg-avatar { flex: 0 0 44px; width: 44px; height: 44px; border-radius: 50%; display: grid; place-items: center;
    font-weight: 800; font-size: 1rem; color: #fff; background: linear-gradient(135deg, #7C5CFC, #22D3EE);
    box-shadow: 0 0 0 3px rgba(124,92,252,0.25); }
.rg-avatar.admin { background: linear-gradient(135deg, #F59E0B, #EF4444); box-shadow: 0 0 0 3px rgba(245,158,11,0.28); }
.rg-profile .who { min-width: 0; }
.rg-profile .who b { display: block; font-size: 0.98rem; color: #F1F5F9; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.rg-profile .who span { display: block; font-size: 0.76rem; color: #94A3B8; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.rg-badge { display: inline-block; margin-top: 5px; padding: 2px 9px; border-radius: 999px; font-size: 0.66rem;
    font-weight: 700; letter-spacing: 1px; text-transform: uppercase;
    background: rgba(124,92,252,0.18); color: #C4B5FD; border: 1px solid rgba(124,92,252,0.4); }
.rg-badge.admin { background: rgba(245,158,11,0.15); color: #FCD34D; border-color: rgba(245,158,11,0.45); }
.rg-badge.good { background: rgba(52,211,153,0.12); color: #6EE7B7; border-color: rgba(52,211,153,0.4); }
.rg-badge.bad { background: rgba(248,113,113,0.12); color: #FCA5A5; border-color: rgba(248,113,113,0.4); }
.rg-badge.muted { background: rgba(148,163,184,0.10); color: #94A3B8; border-color: rgba(148,163,184,0.3); }
.rg-side-label { color: #64748B; font-size: 0.68rem; font-weight: 700; letter-spacing: 2px;
    text-transform: uppercase; margin: 14px 4px 6px; }

/* ---------- sidebar navigation (radio styled as nav items) ---------- */
section[data-testid="stSidebar"] div[role="radiogroup"] { gap: 3px; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label {
    width: 100%; margin: 0; padding: 10px 14px; border-radius: 12px; border: 1px solid transparent;
    transition: background .15s ease, border-color .15s ease; cursor: pointer; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child { display: none; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label p { font-size: 0.95rem; font-weight: 500; color: #CBD5E1; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover { background: rgba(124,92,252,0.10); }
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
    background: linear-gradient(90deg, rgba(124,92,252,0.32), rgba(124,92,252,0.06));
    border-color: rgba(124,92,252,0.45); box-shadow: inset 3px 0 0 #7C5CFC; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p { color: #FFFFFF; font-weight: 700; }
section[data-testid="stSidebar"] hr { margin: 12px 0; border-color: rgba(148,163,184,0.12); }

.rg-hero {
    padding: 28px 32px; border-radius: 22px; margin-bottom: 22px;
    background: linear-gradient(135deg, rgba(124,92,252,0.28), rgba(34,211,238,0.10));
    border: 1px solid rgba(124,92,252,0.35);
    box-shadow: 0 10px 40px rgba(124,92,252,0.15);
}
.rg-hero h1 { margin: 0; font-size: 2.1rem; font-weight: 800; letter-spacing: -0.5px;
    background: linear-gradient(90deg, #FFFFFF, #C4B5FD 60%, #67E8F9);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.rg-hero p { margin: 6px 0 0; color: #A5B4CF; font-size: 1.02rem; }
.rg-eyebrow { font-size: 0.72rem; letter-spacing: 3px; text-transform: uppercase; color: #A78BFA; font-weight: 700; }

.rg-card {
    background: rgba(21,27,40,0.78); border: 1px solid rgba(148,163,184,0.14);
    border-radius: 18px; padding: 20px 22px; height: 100%;
    transition: transform .15s ease, border-color .15s ease;
}
.rg-card:hover { transform: translateY(-2px); border-color: rgba(124,92,252,0.5); }
.rg-card h3 { margin: 0 0 6px; font-size: 1.08rem; }
.rg-card p { margin: 0; color: #94A3B8; font-size: 0.92rem; line-height: 1.5; }
.rg-icon { font-size: 1.7rem; margin-bottom: 8px; }

.rg-metric { background: rgba(21,27,40,0.85); border: 1px solid rgba(148,163,184,0.14);
    border-radius: 16px; padding: 16px 18px; }
.rg-metric .label { color: #94A3B8; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1px; }
.rg-metric .value { font-size: 1.6rem; font-weight: 800; margin-top: 2px; }
.rg-metric .sub { color: #64748B; font-size: 0.8rem; }

.rg-chip { display: inline-block; padding: 5px 12px; margin: 3px 4px 3px 0; border-radius: 999px;
    background: rgba(124,92,252,0.16); border: 1px solid rgba(124,92,252,0.35); color: #DDD6FE; font-size: 0.85rem; }
.rg-chip.good { background: rgba(52,211,153,0.12); border-color: rgba(52,211,153,0.4); color: #A7F3D0; }
.rg-chip.bad { background: rgba(248,113,113,0.12); border-color: rgba(248,113,113,0.4); color: #FECACA; }

.rg-role { border-radius: 16px; padding: 16px 18px; border: 1px solid rgba(148,163,184,0.16);
    background: rgba(21,27,40,0.85); }
.rg-role.rank-1 { border-color: rgba(251,191,36,0.6); box-shadow: 0 0 24px rgba(251,191,36,0.12); }
.rg-role .rank { font-size: 0.75rem; color: #94A3B8; letter-spacing: 1px; text-transform: uppercase; }
.rg-role .name { font-size: 1.12rem; font-weight: 700; margin: 2px 0 6px; }
.rg-bar { height: 8px; border-radius: 99px; background: rgba(148,163,184,0.15); overflow: hidden; }
.rg-bar > div { height: 100%; border-radius: 99px; background: linear-gradient(90deg, #7C5CFC, #22D3EE); }

.rg-model-status { display: flex; align-items: center; gap: 10px; padding: 8px 10px; border-radius: 12px;
    background: rgba(52,211,153,0.08); border: 1px solid rgba(52,211,153,0.25); margin-bottom: 6px; font-size: 0.85rem; }
.rg-model-status .dot { width: 8px; height: 8px; flex: 0 0 8px; border-radius: 50%; background: #34D399; box-shadow: 0 0 8px #34D399; }
.rg-model-status small { color: #94A3B8; display: block; font-size: 0.72rem; }
.rg-model-status.loading { background: rgba(251,191,36,0.08); border-color: rgba(251,191,36,0.28); }
.rg-model-status.loading .dot { width: 14px; height: 14px; flex: 0 0 14px; background: transparent; box-shadow: none;
    border: 2px solid rgba(251,191,36,0.25); border-top-color: #FBBF24; animation: rg-spin 0.8s linear infinite; }
@keyframes rg-spin { to { transform: rotate(360deg); } }
.rg-model-status.failed { background: rgba(248,113,113,0.08); border-color: rgba(248,113,113,0.3); }
.rg-model-status.failed .dot { background: #F87171; box-shadow: 0 0 8px #F87171; }
@keyframes rg-pulse { 50% { opacity: 0.3; } }

.rg-pipeline { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin: 8px 0 4px; }
.rg-step { padding: 10px 14px; border-radius: 12px; background: rgba(21,27,40,0.9);
    border: 1px solid rgba(124,92,252,0.3); font-size: 0.86rem; font-weight: 600; }
.rg-arrow { color: #7C5CFC; font-weight: 800; }

.rg-admin-login { display: flex; align-items: center; gap: 14px; padding: 14px 16px; margin: 6px 0 12px;
    border-radius: 16px; background: linear-gradient(135deg, rgba(245,158,11,0.16), rgba(239,68,68,0.08));
    border: 1px solid rgba(245,158,11,0.40); }
.rg-admin-login .shield { font-size: 1.8rem; }
.rg-admin-login b { display: block; color: #FCD34D; }
.rg-admin-login span { color: #94A3B8; font-size: 0.85rem; }

div[data-testid="stFileUploader"] section { border-radius: 16px; border: 1.5px dashed rgba(124,92,252,0.45);
    background: rgba(124,92,252,0.05); }
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] button {
    border-radius: 12px; font-weight: 600; }
/* Brand accent everywhere, even when .streamlit/config.toml is not picked up (Streamlit's default is red). */
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"],
[data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"] {
    background: linear-gradient(90deg, #7C5CFC, #6366F1); border: none; color: #FFFFFF; }
.stButton > button[kind="primary"]:hover, .stDownloadButton > button[kind="primary"]:hover,
[data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"]:hover { filter: brightness(1.08); color: #FFFFFF; }
.stButton > button[kind="secondary"]:hover, .stDownloadButton > button[kind="secondary"]:hover {
    border-color: #7C5CFC; color: #C4B5FD; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] { border-radius: 10px 10px 0 0; padding: 8px 16px; }
.stTabs [data-baseweb="tab"][aria-selected="true"], .stTabs [data-baseweb="tab"][aria-selected="true"] p { color: #A78BFA; }
.stTabs [data-baseweb="tab-highlight"] { background-color: #7C5CFC; }
[data-testid="stDecoration"] { background-image: linear-gradient(90deg, #7C5CFC, #22D3EE); }
div[data-testid="stExpander"] { border-radius: 14px; }

/* ---------- admin panel ---------- */
.rg-hero.admin { background: linear-gradient(135deg, rgba(245,158,11,0.22), rgba(124,92,252,0.16));
    border-color: rgba(245,158,11,0.40); box-shadow: 0 10px 40px rgba(245,158,11,0.10); }
.rg-hero.admin .rg-eyebrow { color: #FCD34D; }
.rg-timeline { position: relative; padding-left: 22px; }
.rg-timeline:before { content: ""; position: absolute; left: 7px; top: 6px; bottom: 6px; width: 2px;
    background: linear-gradient(180deg, rgba(124,92,252,0.6), rgba(124,92,252,0.05)); }
.rg-tl-item { position: relative; padding: 9px 12px; margin-bottom: 8px; border-radius: 12px;
    background: rgba(21,27,40,0.85); border: 1px solid rgba(148,163,184,0.12); }
.rg-tl-item:before { content: ""; position: absolute; left: -20px; top: 15px; width: 10px; height: 10px;
    border-radius: 50%; background: var(--dot, #7C5CFC); box-shadow: 0 0 10px var(--dot, #7C5CFC); }
.rg-tl-item .top { display: flex; justify-content: space-between; gap: 10px; font-size: 0.88rem; }
.rg-tl-item .top b { color: #F1F5F9; }
.rg-tl-item .top small { color: #64748B; white-space: nowrap; }
.rg-tl-item .det { color: #94A3B8; font-size: 0.8rem; margin-top: 2px; }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def esc(text):
    return html.escape(str(text))


def hero(title, subtitle, eyebrow="RecruitGen", kind=""):
    st.markdown(
        f"""<div class="rg-hero {kind}"><div class="rg-eyebrow">{esc(eyebrow)}</div>
        <h1>{esc(title)}</h1><p>{esc(subtitle)}</p></div>""",
        unsafe_allow_html=True,
    )


def card(icon, title, body):
    st.markdown(
        f"""<div class="rg-card"><div class="rg-icon">{icon}</div>
        <h3>{esc(title)}</h3><p>{esc(body)}</p></div>""",
        unsafe_allow_html=True,
    )


def metric(label, value, sub="", color=None):
    style = f' style="color:{color}"' if color else ""
    st.markdown(
        f"""<div class="rg-metric"><div class="label">{esc(label)}</div>
        <div class="value"{style}>{esc(value)}</div><div class="sub">{esc(sub)}</div></div>""",
        unsafe_allow_html=True,
    )


def chips(items, kind=""):
    if not items:
        return
    st.markdown("".join(f'<span class="rg-chip {kind}">{esc(i)}</span>' for i in items), unsafe_allow_html=True)


def role_card(rank, role, score, detail=""):
    medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"#{rank}")
    st.markdown(
        f"""<div class="rg-role rank-{rank}"><div class="rank">{medal} Rank {rank}</div>
        <div class="name">{esc(role)}</div>
        <div class="rg-bar"><div style="width:{max(2, min(100, score)):.1f}%"></div></div>
        <div style="margin-top:8px;font-weight:700;font-size:1.3rem">{score:.1f}%</div>
        <div style="color:#94A3B8;font-size:0.8rem">{esc(detail)}</div></div>""",
        unsafe_allow_html=True,
    )


def pipeline(steps):
    parts = []
    for i, step in enumerate(steps):
        if i:
            parts.append('<span class="rg-arrow">→</span>')
        parts.append(f'<span class="rg-step">{esc(step)}</span>')
    st.markdown(f'<div class="rg-pipeline">{"".join(parts)}</div>', unsafe_allow_html=True)


def model_status(name, files, state="loaded"):
    st.markdown(
        f"""<div class="rg-model-status {state}"><span class="dot"></span>
        <div><b>{esc(name)}</b><small>{esc(files)}</small></div></div>""",
        unsafe_allow_html=True,
    )


def score_color(score):
    if score >= 75:
        return SUCCESS
    if score >= 50:
        return WARNING
    return DANGER


def initials(name):
    parts = [p for p in str(name).split() if p]
    return "".join(p[0] for p in parts[:2]).upper() or "?"


def sidebar_brand():
    st.markdown(
        """<div class="rg-side-brand"><div class="mark">🤖</div>
        <div><div class="name">RecruitGen</div><div class="tag">AI Recruitment &amp; Screening</div></div></div>""",
        unsafe_allow_html=True,
    )


def profile_card(user, since=""):
    admin = user.get("role") == "admin"
    role = "Administrator" if admin else "Recruiter"
    since_html = f'<span style="margin-top:3px">🕒 Signed in {esc(since)}</span>' if since else ""
    # Raw HTML block (no markdown) so the email is not auto-linked.
    st.markdown(
        f"""<div class="rg-profile"><div class="rg-avatar{' admin' if admin else ''}">{esc(initials(user['full_name']))}</div>
<div class="who"><b>{esc(user['full_name'])}</b><span>@{esc(user['username'])} · {esc(user['email'])}</span>
<div class="rg-badge{' admin' if admin else ''}">{'🛡️ ' if admin else ''}{role}</div>{since_html}</div></div>""",
        unsafe_allow_html=True,
    )


def side_label(text):
    st.markdown(f'<div class="rg-side-label">{esc(text)}</div>', unsafe_allow_html=True)


def timeline(items):
    """items: iterable of (title, details, time_text, color)."""
    rows = "".join(
        f"""<div class="rg-tl-item" style="--dot:{color}"><div class="top"><b>{esc(title)}</b>
<small>{esc(when)}</small></div><div class="det">{esc(details)}</div></div>"""
        for title, details, when, color in items
    )
    st.markdown(f'<div class="rg-timeline">{rows}</div>', unsafe_allow_html=True)


# ============================================================
# LOGIN PAGE (styles are injected only while the login page is shown)
# ============================================================

LOGIN_CSS = """
<style>
/* Aurora background: a glow layer that drifts with `transform` (GPU-composited, no per-frame repaint) */
.stApp { background: linear-gradient(160deg, #060912 0%, #0A0F1E 55%, #080C18 100%); }
.stApp::before {
    content: ""; position: fixed; inset: -20%; pointer-events: none;
    background:
        radial-gradient(38% 34% at 22% 24%, rgba(124,92,252,0.30), transparent 70%),
        radial-gradient(34% 32% at 80% 76%, rgba(34,211,238,0.17), transparent 70%),
        radial-gradient(28% 26% at 70% 18%, rgba(236,72,153,0.11), transparent 70%);
    animation: rg-drift 26s ease-in-out infinite alternate; will-change: transform;
}
@keyframes rg-drift {
    from { transform: translate3d(-3%, -2%, 0) scale(1); }
    to   { transform: translate3d(4%, 3%, 0) scale(1.06); }
}
.block-container, [data-testid="stAppViewBlockContainer"] { max-width: 1180px; }

/* Split layout: hero left, sign-in card right (card first on phones) */
[data-testid="stHorizontalBlock"]:has(.rg-login-hero) { flex-direction: row-reverse; align-items: center; }
@media (max-width: 640px) {
    [data-testid="stHorizontalBlock"]:has(.rg-login-hero) { flex-direction: column; }
}

/* Hero */
.rg-login-hero { padding: 6px 4px; animation: rg-rise .6s ease both; }
.rg-brandrow { display: flex; align-items: center; gap: 14px; margin-bottom: 26px; }
.rg-mark-lg { width: 54px; height: 54px; border-radius: 16px; display: grid; place-items: center; font-size: 1.7rem;
    background: linear-gradient(135deg, #7C5CFC, #22D3EE);
    box-shadow: 0 10px 30px rgba(124,92,252,0.45), inset 0 1px 0 rgba(255,255,255,0.25);
    animation: rg-bob 5s ease-in-out infinite; }
.rg-brandrow .name { font-size: 1.45rem; font-weight: 800; letter-spacing: -0.3px; color: #FFFFFF; line-height: 1.1; }
.rg-brandrow .tag { font-size: 0.8rem; color: #8B9AB8; }
.rg-pill { display: inline-flex; align-items: center; gap: 8px; padding: 6px 14px; border-radius: 999px;
    background: rgba(124,92,252,0.12); border: 1px solid rgba(124,92,252,0.35); color: #C4B5FD;
    font-size: 0.78rem; font-weight: 600; letter-spacing: 0.4px; }
.rg-pill .pulse { width: 8px; height: 8px; border-radius: 50%; background: #34D399; animation: rg-ping 2s infinite; }
.rg-headline { font-size: 2.75rem; line-height: 1.1; font-weight: 800; letter-spacing: -1px; color: #F8FAFC;
    margin: 18px 0 14px; }
.rg-headline span { background: linear-gradient(90deg, #A78BFA, #22D3EE); -webkit-background-clip: text;
    background-clip: text; -webkit-text-fill-color: transparent; }
.rg-lead { color: #A5B4CF; font-size: 1.05rem; line-height: 1.6; max-width: 540px; margin: 0 0 26px; }
.rg-feats { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 26px; }
.rg-feat { display: flex; gap: 12px; align-items: flex-start; padding: 14px; border-radius: 16px;
    background: rgba(21,27,40,0.55); border: 1px solid rgba(148,163,184,0.12);
    transition: transform .18s ease, border-color .18s ease, background .18s ease; }
.rg-feat:hover { transform: translateY(-3px); border-color: rgba(124,92,252,0.45); background: rgba(124,92,252,0.08); }
.rg-feat .ic { flex: 0 0 38px; height: 38px; border-radius: 11px; display: grid; place-items: center;
    font-size: 1.15rem; background: linear-gradient(135deg, rgba(124,92,252,0.30), rgba(34,211,238,0.18));
    border: 1px solid rgba(124,92,252,0.35); }
.rg-feat b { display: block; color: #F1F5F9; font-size: 0.92rem; margin-bottom: 2px; }
.rg-feat span { color: #8B9AB8; font-size: 0.8rem; line-height: 1.4; }
.rg-stats { display: flex; }
.rg-stat { padding: 0 22px; border-left: 1px solid rgba(148,163,184,0.18); }
.rg-stat:first-child { padding-left: 0; border-left: none; }
.rg-stat b { display: block; font-size: 1.7rem; font-weight: 800; line-height: 1.2;
    background: linear-gradient(90deg, #FFFFFF, #C4B5FD); -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent; }
.rg-stat span { color: #8B9AB8; font-size: 0.74rem; text-transform: uppercase; letter-spacing: 1px; }

/* Card header + segmented tabs */
.rg-auth-head { margin: 2px 2px 16px; animation: rg-rise .5s ease both; }
.rg-auth-head .t { font-size: 1.6rem; font-weight: 800; color: #F8FAFC; letter-spacing: -0.4px; }
.rg-auth-head .s { color: #8B9AB8; font-size: 0.9rem; margin-top: 4px; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; padding: 5px; border-radius: 14px;
    background: rgba(13,18,32,0.75); border: 1px solid rgba(148,163,184,0.14); }
.stTabs [data-baseweb="tab"] { flex: 1 1 0; justify-content: center; height: 42px; padding: 0 10px;
    border-radius: 10px !important; background: transparent; transition: background .15s ease; }
.stTabs [data-baseweb="tab"]:hover { background: rgba(124,92,252,0.10); }
.stTabs [data-baseweb="tab"] p { font-size: 0.92rem; font-weight: 600; color: #94A3B8; }
.stTabs [data-baseweb="tab"][aria-selected="true"] { background: linear-gradient(135deg, #7C5CFC, #6366F1);
    box-shadow: 0 8px 20px rgba(124,92,252,0.35); }
.stTabs [data-baseweb="tab"]:nth-of-type(3)[aria-selected="true"] { background: linear-gradient(135deg, #F59E0B, #EF4444);
    box-shadow: 0 8px 20px rgba(245,158,11,0.30); }
.stTabs [data-baseweb="tab"][aria-selected="true"] p { color: #FFFFFF; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 14px; }

/* Glass form card */
[data-testid="stForm"] { padding: 22px 22px 18px; border-radius: 20px; border: 1px solid rgba(124,92,252,0.28);
    background: linear-gradient(180deg, rgba(23,29,45,0.88), rgba(13,18,32,0.88));
    box-shadow: 0 24px 60px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.05); }
[data-testid="stForm"]:has(.rg-admin-form) { border-color: rgba(245,158,11,0.35); }
[data-testid="stForm"] [data-testid="stWidgetLabel"] p { font-size: 0.82rem; font-weight: 600; color: #CBD5E1;
    letter-spacing: 0.2px; }
[data-testid="stForm"] div[data-baseweb="input"] { border-radius: 12px; border: 1px solid rgba(148,163,184,0.20) !important;
    background: rgba(6,10,20,0.80); transition: border-color .15s ease, box-shadow .15s ease; }
[data-testid="stForm"] div[data-baseweb="input"]:focus-within { border-color: #7C5CFC !important;
    box-shadow: 0 0 0 3px rgba(124,92,252,0.25); }
[data-testid="stForm"]:has(.rg-admin-form) div[data-baseweb="input"]:focus-within { border-color: #F59E0B !important;
    box-shadow: 0 0 0 3px rgba(245,158,11,0.22); }
[data-testid="stForm"] div[data-baseweb="base-input"], [data-testid="stForm"] div[data-baseweb="base-input"] button {
    background: transparent; }
[data-testid="stForm"] input { color: #F1F5F9; font-size: 0.95rem; }
[data-testid="stForm"] input::placeholder { color: #5B6B88; }
[data-testid="stFormSubmitButton"] button { height: 48px; margin-top: 6px; border: none !important; color: #FFFFFF !important;
    background: linear-gradient(135deg, #7C5CFC 0%, #6366F1 60%, #0EA5E9 140%) !important;
    box-shadow: 0 10px 26px rgba(124,92,252,0.40); transition: transform .12s ease, box-shadow .12s ease, filter .12s ease; }
[data-testid="stFormSubmitButton"] button:hover { transform: translateY(-1px); filter: brightness(1.07);
    box-shadow: 0 14px 32px rgba(124,92,252,0.55); }
[data-testid="stFormSubmitButton"] button:active { transform: translateY(0); }
[data-testid="stFormSubmitButton"] button p { font-size: 1rem; font-weight: 700; letter-spacing: 0.3px; }
[data-testid="stForm"]:has(.rg-admin-form) [data-testid="stFormSubmitButton"] button {
    background: linear-gradient(135deg, #F59E0B, #EF4444) !important; box-shadow: 0 10px 26px rgba(245,158,11,0.35); }
.rg-form-sub { color: #8B9AB8; font-size: 0.86rem; margin: -2px 0 4px; }

.rg-trust { display: flex; flex-wrap: wrap; justify-content: center; gap: 6px 16px; margin-top: 16px;
    color: #7C8BA8; font-size: 0.78rem; }
.rg-login-foot { text-align: center; color: #56657F; font-size: 0.76rem; margin-top: 36px; }

@keyframes rg-rise { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
@keyframes rg-bob { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-5px); } }
@keyframes rg-ping { 0% { box-shadow: 0 0 0 0 rgba(52,211,153,0.55); } 70% { box-shadow: 0 0 0 9px rgba(52,211,153,0); }
    100% { box-shadow: 0 0 0 0 rgba(52,211,153,0); } }
@media (max-width: 900px) { .rg-feats { grid-template-columns: 1fr; } .rg-headline { font-size: 2.2rem; } }
@media (max-width: 640px) { .rg-login-hero { margin-top: 30px; } .rg-headline { font-size: 1.9rem; }
    .rg-stat { padding: 0 14px; } }
@media (prefers-reduced-motion: reduce) { .stApp::before, .rg-mark-lg, .rg-pill .pulse { animation: none; } }
</style>
"""

LOGIN_FEATURES = [
    ("🧹", "Resume denoising", "Autoencoder restores scanned and blurry resumes"),
    ("🌫️", "Image enhancement", "Diffusion model sharpens degraded resume scans for OCR"),
    ("🤖", "Role prediction", "Transformer V3 reads skills, projects and experience"),
    ("🧬", "Role-fit ranking", "CVAE ranks the three best-fit job roles"),
    ("🧪", "Synthetic candidates", "GAN generates labelled synthetic profiles for testing"),
    ("✨", "AI insights & PDFs", "Interview kits, job match and branded reports"),
]
LOGIN_STATS = [("5", "AI models"), ("15", "PDF reports"), ("13", "Resume formats")]


def login_styles():
    st.markdown(LOGIN_CSS, unsafe_allow_html=True)


def login_hero():
    feats = "".join(
        f'<div class="rg-feat"><div class="ic">{icon}</div><div><b>{esc(title)}</b><span>{esc(text)}</span></div></div>'
        for icon, title, text in LOGIN_FEATURES
    )
    stats = "".join(f'<div class="rg-stat"><b>{value}</b><span>{esc(label)}</span></div>' for value, label in LOGIN_STATS)
    st.markdown(
        '<div class="rg-login-hero">'
        '<div class="rg-brandrow"><div class="rg-mark-lg">🤖</div>'
        '<div><div class="name">RecruitGen</div><div class="tag">AI Recruitment &amp; Candidate Screening</div></div></div>'
        '<div class="rg-pill"><span class="pulse"></span>AI-powered hiring platform</div>'
        '<div class="rg-headline">Hire smarter with <span>AI-driven</span> resume screening</div>'
        '<div class="rg-lead">Clean, read, score and rank any resume in seconds — five deep-learning models and an '
        'AI assistant working together.</div>'
        f'<div class="rg-feats">{feats}</div><div class="rg-stats">{stats}</div></div>',
        unsafe_allow_html=True,
    )


def auth_header(title, subtitle):
    st.markdown(f'<div class="rg-auth-head"><div class="t">{esc(title)}</div><div class="s">{esc(subtitle)}</div></div>',
                unsafe_allow_html=True)


def form_subtitle(text, admin=False):
    marker = '<span class="rg-admin-form"></span>' if admin else ""
    st.markdown(f'<div class="rg-form-sub">{marker}{esc(text)}</div>', unsafe_allow_html=True)


def trust_badges(items):
    st.markdown('<div class="rg-trust">' + "".join(f"<span>{esc(i)}</span>" for i in items) + "</div>",
                unsafe_allow_html=True)


def login_footer(text):
    st.markdown(f'<div class="rg-login-foot">{esc(text)}</div>', unsafe_allow_html=True)
