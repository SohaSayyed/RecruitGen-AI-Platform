"""
RecruitGen — AI-Powered Recruitment & Candidate Screening System
================================================================
Five AI models:
    1. Autoencoder      resume image denoising          models/autoencoder/
    2. CVAE / VAE       candidate generation + role fit models/cvae/
    3. Transformer V3   feature extraction + role prediction models/transformer_v3/
    4. GAN              synthetic profiles for dataset augmentation models/gan/
    5. Diffusion        resume image enhancement for OCR models/diffusion/

Run:  streamlit run app.py
"""

import hashlib
import io
import os
import threading
import time
import warnings
from datetime import datetime, timedelta

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("KERAS_BACKEND", "torch")  # skip importing TensorFlow (much faster startup)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import pdf_downloads as pdf
from core import auth, ui
from core.resume_intelligence import (
    SUPPORTED_TYPES, IMAGE_TYPES, ai_available, read_resume, image_to_text,
    resume_insights, job_match, interview_questions, assistant_reply, format_ai_markdown,
    synthetic_profile_summary, explain_gan_results, enhance_resume_image, local_ocr, enhancement_assistant,
    diagnose_resume_image, local_image_diagnosis,
)

st.set_page_config(
    page_title="RecruitGen — AI Recruitment",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)
ui.inject_css()
auth.init_db()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_IMAGE = os.path.join(BASE_DIR, "sample_data", "sample_resume.png")
# Realistically blurred / noisy scan of a real (anonymised Kaggle) resume page + its clean original
BLURRED_SAMPLE = os.path.join(BASE_DIR, "sample_data", "sample_resume_blurred.png")
CLEAN_SAMPLE = os.path.join(BASE_DIR, "sample_data", "sample_resume_clean.png")
if not os.path.exists(BLURRED_SAMPLE):
    BLURRED_SAMPLE = SAMPLE_IMAGE
if not os.path.exists(CLEAN_SAMPLE):
    CLEAN_SAMPLE = SAMPLE_IMAGE
SAMPLE_TEXT = os.path.join(BASE_DIR, "sample_data", "sample_resume.txt")

FEATURES = [
    ("experience_years", "Experience Years", "Candidate's professional experience", 0.0, 20.0, 0.5),
    ("projects_count", "Projects Count", "Number of projects detected", 0, 20, 1),
    ("programming_score", "Programming Score", "Programming-related evidence", 0.0, 100.0, 1.0),
    ("data_analytics_score", "Data Analytics Score", "Data-analysis-related evidence", 0.0, 100.0, 1.0),
    ("ml_ai_score", "ML / AI Score", "ML/AI-related evidence", 0.0, 100.0, 1.0),
    ("cloud_score", "Cloud Score", "Cloud-related evidence", 0.0, 100.0, 1.0),
    ("communication_score", "Communication Score", "Communication / teamwork evidence", 0.0, 100.0, 1.0),
    ("certification_count", "Certification Count", "Certifications detected", 0, 15, 1),
    ("education_level", "Education Level", "1 Diploma · 2 Bachelor's · 3 Master's · 5 PhD", 1, 5, 1),
]
FEATURE_LABEL = {f[0]: f[1] for f in FEATURES}
EDUCATION = {1: "Diploma / 12th", 2: "Bachelor's", 3: "Master's", 4: "Advanced Degree", 5: "PhD"}
DEFAULT_FEATURES = {
    "experience_years": 1.5, "projects_count": 4, "programming_score": 30.0, "data_analytics_score": 60.0,
    "ml_ai_score": 90.0, "cloud_score": 45.0, "communication_score": 70.0, "certification_count": 3,
    "education_level": 2,
}


# ============================================================
# MODEL LOADING
# ============================================================

def _build_autoencoder():
    from models.autoencoder.autoencoder import ResumeDenoiser
    return ResumeDenoiser()


def _build_cvae():
    from models.cvae.cvae import CandidateCVAE
    return CandidateCVAE()


def _build_transformer():
    from models.transformer_v3.transformer_v3 import ResumeScreener
    return ResumeScreener()


def _build_gan():
    from models.gan.gan import CandidateGAN
    return CandidateGAN()


def _build_diffusion():
    from models.diffusion.diffusion import ResumeDiffusionEnhancer
    return ResumeDiffusionEnhancer()


MODEL_SPECS = [
    {"key": "autoencoder", "name": "Autoencoder", "files": "encoder.h5 + decoder.h5",
     "task": "Resume image denoising", "build": _build_autoencoder},
    {"key": "cvae", "name": "CVAE / VAE", "files": "vae_model.h5 + vae.weights.h5",
     "task": "Candidate generation & role fit", "build": _build_cvae},
    {"key": "transformer", "name": "Transformer V3", "files": "model.safetensors + tokenizer",
     "task": "Feature extraction & role prediction", "build": _build_transformer},
    {"key": "gan", "name": "GAN", "files": "generator.safetensors + config.json",
     "task": "Synthetic profiles for dataset augmentation", "build": _build_gan},
    {"key": "diffusion", "name": "Diffusion", "files": "diffusion_unet.safetensors + config.json",
     "task": "Resume image enhancement (optional)", "build": _build_diffusion},
]
DIFFUSION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "diffusion")
DIFFUSION_FILES = [os.path.join(DIFFUSION_DIR, name) for name in ("diffusion_unet.safetensors", "config.json")]


def diffusion_ready():
    """Plain file check (no torch import, so it is safe while models load in the background)."""
    return all(os.path.exists(p) for p in DIFFUSION_FILES)


def active_specs():
    """Models shown and loaded in the app. The optional Diffusion model appears only once its trained checkpoint
    exists, so users never see a missing-model error while it is being trained."""
    ready = diffusion_ready()
    return [s for s in MODEL_SPECS if s["key"] != "diffusion" or ready]


class ModelHub:
    """Process-wide model store. Models load in a background thread as soon as the app is first opened
    (while the login page is shown), so they are usually ready by the time a recruiter logs in.
    A page that needs a model before it is ready simply waits for it. Builds are serialized with one
    lock because importing torch / keras from two threads at once is not safe."""

    def __init__(self):
        self.models, self.errors, self.seconds = {}, {}, {}
        self.lock = threading.RLock()
        self.builders = {spec["key"]: spec["build"] for spec in MODEL_SPECS}
        self.thread = None
        self.started_at = None

    def get(self, key):
        if key not in self.models:
            with self.lock:
                if key not in self.models:
                    start = time.time()
                    try:
                        self.models[key] = self.builders[key]()
                        self.errors.pop(key, None)
                    except Exception as exc:
                        self.errors[key] = str(exc)
                        raise
                    finally:
                        self.seconds[key] = time.time() - start
        return self.models[key]

    def status(self, key):
        if key in self.models:
            return "loaded"
        return "failed" if key in self.errors else "loading"

    @property
    def all_loaded(self):
        return all(spec["key"] in self.models for spec in active_specs())

    @property
    def busy(self):
        return self.thread is not None and self.thread.is_alive()

    def elapsed(self):
        return time.time() - self.started_at if self.started_at else 0.0

    def warm_up(self, delay=0.0):
        """Start the background loader (no-op if everything is loaded or a load is in progress).
        `delay` lets the current page reach the browser first: importing torch / keras holds the GIL for
        long stretches and would otherwise stall the very first render."""
        active = {spec["key"] for spec in active_specs()}
        self.errors = {k: v for k, v in self.errors.items() if k in active}   # forget errors of hidden models
        if self.all_loaded or self.busy or self.errors:
            return

        def run():
            if delay:
                time.sleep(delay)
            for spec in active_specs():
                try:
                    self.get(spec["key"])
                except Exception:
                    pass

        self.started_at = self.started_at or time.time()
        self.thread = threading.Thread(target=run, name="recruitgen-model-loader", daemon=True)
        self.thread.start()


@st.cache_resource(show_spinner=False)
def model_hub():
    return ModelHub()


def _load(key):
    hub = model_hub()
    if key in hub.models:
        return hub.models[key]
    name = next(s["name"] for s in MODEL_SPECS if s["key"] == key)
    with st.spinner(f"Loading {name} model (first use only)..."):
        return hub.get(key)


def load_autoencoder():
    return _load("autoencoder")


def load_cvae():
    return _load("cvae")


def load_transformer():
    return _load("transformer")


def load_gan():
    return _load("gan")


def load_diffusion():
    return _load("diffusion")


# ============================================================
# HELPERS
# ============================================================

def feature_scores(features):
    """0-100 view of the 9 features for charts."""
    return {
        "Experience": min(float(features["experience_years"]) / 5.0, 1.0) * 100,
        "Projects": min(float(features["projects_count"]) / 5.0, 1.0) * 100,
        "Programming": float(features["programming_score"]),
        "Data Analytics": float(features["data_analytics_score"]),
        "ML / AI": float(features["ml_ai_score"]),
        "Cloud": float(features["cloud_score"]),
        "Communication": float(features["communication_score"]),
        "Certifications": min(float(features["certification_count"]) / 5.0, 1.0) * 100,
        "Education": min(float(features["education_level"]) / 3.0, 1.0) * 100,
    }


def chart_layout(fig, height=340):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#CBD5E1"),
    )
    return fig


def radar_chart(features):
    scores = feature_scores(features)
    labels = list(scores) + [list(scores)[0]]
    values = list(scores.values()) + [list(scores.values())[0]]
    fig = go.Figure(go.Scatterpolar(r=values, theta=labels, fill="toself", line=dict(color=ui.PRIMARY, width=2),
                                    fillcolor="rgba(124,92,252,0.30)"))
    fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)",
                                 radialaxis=dict(range=[0, 100], gridcolor="#253047", tickfont=dict(size=9)),
                                 angularaxis=dict(gridcolor="#253047")), showlegend=False)
    return chart_layout(fig, 380)


def bar_chart(values, title="", suffix="%"):
    items = sorted(values.items(), key=lambda kv: kv[1])
    fig = go.Figure(go.Bar(
        x=[v for _, v in items], y=[k for k, _ in items], orientation="h",
        marker=dict(color=[v for _, v in items], colorscale=[[0, "#3B3F8F"], [1, ui.PRIMARY]]),
        text=[f"{v:.1f}{suffix}" for _, v in items], textposition="outside",
    ))
    fig.update_layout(title=title, xaxis=dict(range=[0, max(105, max(values.values()) * 1.15)], gridcolor="#1E293B"))
    return chart_layout(fig, 60 + 52 * len(values))


def feature_rows(features):
    """[(label, value, meaning)] for the 9 features, as shown in tables and PDFs."""
    rows = []
    for key, label, meaning, *_ in FEATURES:
        value = features[key]
        if key == "education_level":
            value = f"{int(value)} ({EDUCATION.get(int(value), '-')})"
        rows.append((label, str(value), meaning))
    return rows


def features_table(features):
    rows = [{"#": i, "Feature": label, "Value": value, "Meaning": meaning}
            for i, (label, value, meaning) in enumerate(feature_rows(features), 1)]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# ============================================================
# PDF DOWNLOADS (builders live in pdf_downloads/)
# ============================================================

def prepared_by():
    user = st.session_state.get("user") or {}
    return f"{user['full_name']} (@{user['username']})" if user else ""


def signature(*parts):
    """Content fingerprint: the PDF is rebuilt only when the underlying result changes."""
    norm = [int(pd.util.hash_pandas_object(p, index=False).sum()) if isinstance(p, pd.DataFrame) else p
            for p in parts]
    return hashlib.md5(repr(norm).encode("utf-8", "ignore")).hexdigest()


def pdf_button(label, key, sig, build, file_name, lazy=False, primary=False):
    """Download button for a generated PDF, built once per result (cached in the session).
    lazy=True first shows a "Create" button — used for large admin exports."""
    cache = st.session_state.setdefault("_pdf_cache", {})
    hit = cache.get(key)
    if hit is None or hit[0] != sig:
        if lazy and not st.button(f"📄 Create {label}", key=f"make_{key}", use_container_width=True):
            return
        try:
            with st.spinner("Creating PDF..."):
                cache[key] = (sig, build())
        except Exception as exc:
            st.error(f"Could not create the PDF: {exc}")
            return
    st.download_button(f"📄 {label}", cache[key][1], file_name, "application/pdf", key=f"dl_{key}",
                       use_container_width=True, type="primary" if primary else "secondary",
                       on_click=log, args=("PDF exported", file_name))


def extract_text_from_images(images, use_autoencoder=True, progress_label="Reading resume text", method=None,
                             guard=False):
    """Optional image enhancement (Autoencoder by default, Diffusion optional), then text extraction.
    guard=True keeps the original page whenever local OCR reads the enhanced one less confidently.
    Returns (text, originals, enhanced)."""
    method = method or ("autoencoder" if use_autoencoder else "none")
    originals, cleaned = [], []
    texts = []
    for i, img in enumerate(images):
        source = img
        if method != "none":
            model = load_diffusion() if method == "diffusion" else load_autoencoder()
            steps = None
            if method == "diffusion":           # API first (image analysis -> steps), then the local Diffusion model
                diagnosis = (diagnose_resume_image(img) if ai_available() else None) or local_image_diagnosis(img)
                steps = diagnosis["recommended_steps"]
            res = enhance_resume_image(img, method, enhancer=model, guard=guard, steps=steps)
            originals.append(img)
            cleaned.append(res["enhanced"] or img)
            source = res["image"]
        texts.append(image_to_text(source))
    return "\n\n".join(t for t in texts if t), originals, cleaned


def get_resume_text_from_upload(uploaded, use_autoencoder=True, method=None, guard=False):
    parsed = read_resume(uploaded)
    originals, cleaned = [], []
    text = parsed["text"]
    if parsed["images"]:
        text, originals, cleaned = extract_text_from_images(parsed["images"], use_autoencoder, method=method,
                                                            guard=guard)
    return text, originals, cleaned


def png_bytes(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def client_info():
    """Short 'Browser · OS · IP' description of the current visitor (best effort)."""
    try:
        headers = st.context.headers  # Streamlit >= 1.37
    except Exception:
        try:
            from streamlit.web.server.websocket_headers import _get_websocket_headers
            headers = _get_websocket_headers() or {}
        except Exception:
            return ""
    ua = headers.get("User-Agent", "") or ""
    browser = next((b for k, b in [("Edg/", "Edge"), ("OPR/", "Opera"), ("Chrome/", "Chrome"),
                                   ("Firefox/", "Firefox"), ("Safari/", "Safari")] if k in ua), "Browser")
    system = next((o for k, o in [("Windows", "Windows"), ("Android", "Android"), ("iPhone", "iOS"),
                                  ("iPad", "iOS"), ("Mac OS", "macOS"), ("Linux", "Linux")] if k in ua), "")
    ip = (headers.get("X-Forwarded-For", "") or "").split(",")[0].strip()
    host = (headers.get("Host", "") or "").split(":")[0]
    if not ip and host in ("localhost", "127.0.0.1"):
        ip = "localhost"
    return " · ".join(p for p in [browser, system, ip] if p)


def log(event, details=""):
    """Record an activity event for the logged-in user."""
    try:
        auth.log_activity(st.session_state.get("user"), event, details,
                          st.session_state.get("session_id"), st.session_state.get("client", ""))
    except Exception:
        pass


def is_admin():
    return auth.is_admin(st.session_state.get("user"))


def ai_unavailable_notice():
    st.info("AI insights are temporarily unavailable. Please try again in a moment.", icon="⏳")


def render_insights(ins):
    if not ins:
        ai_unavailable_notice()
        return
    c1, c2, c3 = st.columns(3)
    with c1:
        s = int(ins.get("overall_score", 0) or 0)
        ui.metric("Resume Score", f"{s}/100", "Overall quality & hireability", ui.score_color(s))
    with c2:
        s = int(ins.get("ats_score", 0) or 0)
        ui.metric("ATS Score", f"{s}/100", "Applicant tracking friendliness", ui.score_color(s))
    with c3:
        ui.metric("Candidate", ins.get("candidate_name", "Candidate"), ins.get("headline", ""))
    st.markdown("#### 📝 Professional Summary")
    st.write(ins.get("summary", ""))
    st.markdown("#### 🛠️ Key Skills")
    ui.chips(ins.get("key_skills", []))
    a, b = st.columns(2)
    with a:
        st.markdown("#### ✅ Strengths")
        for s in ins.get("strengths", []):
            st.markdown(f"- {s}")
    with b:
        st.markdown("#### ⚠️ Gaps")
        for s in ins.get("gaps", []):
            st.markdown(f"- {s}")
    st.markdown("#### 🚀 Suggestions")
    for s in ins.get("suggestions", []):
        st.markdown(f"- {s}")
    st.markdown("#### 🎯 Suitable Roles")
    ui.chips(ins.get("suitable_roles", []), "good")
    if ins.get("interview_questions"):
        with st.expander("❓ Suggested interview questions"):
            for i, q in enumerate(ins["interview_questions"], 1):
                st.markdown(f"**{i}.** {q}")


@st.cache_data(show_spinner=False, ttl=3600)
def cached_insights(text, role, features_items):
    return resume_insights(text, role, dict(features_items) if features_items else None)


# ============================================================
# AUTH PAGES
# ============================================================

def attempt_login(admin):
    """Sign-in button callback for the user tab (recruiters) or the admin tab (administrators only).
    It runs before the page is drawn, so a successful login goes straight to the dashboard."""
    prefix = "admin" if admin else "user"
    identifier = st.session_state.get(f"{prefix}_identifier", "")
    password = st.session_state.get(f"{prefix}_password", "")
    user = auth.authenticate(identifier, password)
    client = client_info()
    username = (identifier or "").strip().lower() or "—"
    if not user:
        auth.log_activity(None, "Login failed", f"{'Admin' if admin else 'User'} login · invalid credentials",
                          client=client, username=username)
        st.session_state[f"{prefix}_login_error"] = "Invalid username/email or password."
        return
    if auth.is_admin(user) != admin:
        auth.log_activity(user, "Login failed", f"Used the {'Admin' if admin else 'User'} tab with "
                                                f"{'an admin' if auth.is_admin(user) else 'a recruiter'} account",
                          client=client)
        st.session_state[f"{prefix}_login_error"] = ("This is not an administrator account." if admin
                                                    else "Administrators must sign in from the 🛡️ Admin tab.")
        return
    st.session_state.user = user
    st.session_state.client = client
    st.session_state.session_id = auth.start_session(user, client)
    st.session_state.login_time = datetime.now()
    log("Login", f"{'Admin' if admin else 'User'} signed in")


def login_error(prefix):
    message = st.session_state.pop(f"{prefix}_login_error", None)
    if message:
        st.error(message, icon="🚫")


def login_page():
    ui.login_styles()
    # Code order is card → gap → hero; CSS shows the hero on the left on desktop and the card first on phones.
    card, _, hero = st.columns([1, 0.14, 1.25])
    with hero:
        ui.login_hero()
    with card:
        ui.auth_header("Welcome back 👋", "Sign in to your workspace, create an account or open the admin console.")
        login_tab, signup_tab, admin_tab = st.tabs(["🔐 Login", "📝 Sign Up", "🛡️ Admin"])
        with login_tab:
            with st.form("login_form"):
                ui.form_subtitle("Sign in with your username or email.")
                st.text_input("Username or Email", placeholder="e.g. Soha_Sayyed or you@company.com",
                              key="user_identifier")
                st.text_input("Password", type="password", placeholder="Enter your password", key="user_password")
                st.form_submit_button("Sign in  →", type="primary", use_container_width=True,
                                      on_click=attempt_login, args=(False,))
                login_error("user")
        with signup_tab:
            with st.form("signup_form", clear_on_submit=False):
                ui.form_subtitle("Create a recruiter account — it takes less than a minute.")
                full_name = st.text_input("Full Name", placeholder="e.g. Soha Sayyed")
                c1, c2 = st.columns(2)
                username = c1.text_input("Username", placeholder="3–20 letters, numbers or _")
                email = c2.text_input("Email", placeholder="you@company.com")
                c3, c4 = st.columns(2)
                p1 = c3.text_input("Password", type="password", placeholder="At least 6 characters")
                p2 = c4.text_input("Confirm Password", type="password", placeholder="Re-enter password")
                if st.form_submit_button("Create account  →", type="primary", use_container_width=True):
                    if p1 != p2:
                        st.error("Passwords do not match.")
                    else:
                        ok, msg = auth.create_user(full_name, username, email, p1)
                        if ok:
                            auth.log_activity(None, "Sign up", f"New account: {full_name.strip()} <{email.strip().lower()}>",
                                              client=client_info(), username=username.strip().lower())
                        (st.success if ok else st.error)(msg)
        with admin_tab:
            st.markdown(
                """<div class="rg-admin-login"><div class="shield">🛡️</div><div><b>Administrator access</b>
<span>Monitor logins, logouts, sessions and all user activity.</span></div></div>""",
                unsafe_allow_html=True,
            )
            with st.form("admin_login_form"):
                ui.form_subtitle("Restricted area — administrator credentials only.", admin=True)
                st.text_input("Admin username", placeholder="admin_recruitgen", key="admin_identifier")
                st.text_input("Admin password", type="password", placeholder="Enter the admin password",
                              key="admin_password")
                st.form_submit_button("🛡️ Login as Admin", type="primary", use_container_width=True,
                                      on_click=attempt_login, args=(True,))
                login_error("admin")
        ui.trust_badges(["🔒 Salted & hashed passwords", "🛡️ Activity monitoring", "📄 Branded PDF reports"])
    ui.login_footer(f"© {datetime.now().year} RecruitGen · AI Recruitment & Candidate Screening System")


# ============================================================
# PAGE: DASHBOARD
# ============================================================

def page_dashboard():
    user = st.session_state.user
    ui.hero(f"Welcome back, {user['full_name'].split()[0]} 👋",
            "RecruitGen combines computer vision, generative modeling and NLP into one AI-assisted recruitment pipeline.",
            eyebrow="Dashboard")

    history = auth.get_screenings(user["id"])
    c = st.columns(4)
    with c[0]:
        ui.metric("Resumes Screened", len(history), "All time")
    with c[1]:
        specs = active_specs()
        ready = sum(model_hub().status(spec["key"]) == "loaded" for spec in specs)
        ui.metric("Models Active", f"{ready} / {len(specs)}",
                  "All ready" if ready == len(specs) else "Loading in background…",
                  ui.SUCCESS if ready == len(specs) else ui.WARNING)
    with c[2]:
        avg = sum(h["confidence"] or 0 for h in history) / len(history) if history else 0
        ui.metric("Avg. Confidence", f"{avg:.1f}%", "Transformer V3 predictions")
    with c[3]:
        top = pd.Series([h["predicted_role"] for h in history]).mode()[0] if history else "—"
        ui.metric("Most Common Role", top, "Across screened candidates")

    st.markdown("### 🔄 Complete Pipeline")
    ui.pipeline(["📄 Input Resume (Image / Text)", "🧹 Autoencoder", "🔤 Text Extraction",
                 "📐 9 Features", "🤖 Transformer V3", "🧬 CVAE Role Fit", "🏆 Top 3 Roles"])
    st.caption("Autoencoder = resume cleaning · CVAE / VAE = candidate generation + role-fit · "
               "Transformer V3 = understanding + extraction + prediction · GAN = synthetic candidate profiles · "
               "Diffusion = resume image enhancement")

    st.markdown("### 🧠 5 AI Models")
    specs = active_specs()
    cols = st.columns(len(specs))
    details = {
        "autoencoder": ("🧹", "Encoder → Latent Space → Decoder (U-Net). Removes noise from scanned resumes. "
                              "Metrics: MSE, PSNR, SSIM."),
        "cvae": ("🧬", "Conditional VAE with reparameterization. Generates synthetic candidates and ranks "
                       "role fit for 5 job roles."),
        "transformer": ("🤖", "DistilBERT text encoder fused with 9 numerical features. Predicts the most "
                              "suitable job role with confidence."),
        "gan": ("🧪", "Conditional WGAN-GP trained on the Kaggle Resume Dataset. Generates clearly labelled "
                      "synthetic profiles for dataset augmentation and testing."),
        "diffusion": ("🌫️", "Conditional diffusion U-Net (DDIM). Restores noisy, blurred, shadowed or low-contrast "
                            "resume scans so OCR reads the text better."),
    }
    for col, spec in zip(cols, specs):
        with col:
            icon, body = details[spec["key"]]
            model = model_hub().models.get(spec["key"])
            params = f"{model.num_params:,}" if model else "loading…"
            ui.card(icon, spec["name"], f"{body}  •  Files: {spec['files']}  •  Parameters: {params}")

    st.markdown("### ⚡ Quick Start")
    q = st.columns(3)
    with q[0]:
        ui.card("📄", "Resume Analyzer", "Upload any resume (PDF, DOCX, image…) and run the full pipeline end to end.")
    with q[1]:
        ui.card("✨", "AI Assistant", "Resume insights, job-description matching, interview questions and chat.")
    with q[2]:
        ui.card("🧬", "CVAE Studio", "Check role fit for any profile and generate synthetic candidates.")


# ============================================================
# PAGE: RESUME ANALYZER (END-TO-END)
# ============================================================

def page_analyzer():
    ui.hero("Resume Analyzer", "Upload a resume in any format — RecruitGen cleans, reads, scores and ranks it.",
            eyebrow="End-to-end pipeline")
    ui.pipeline(["Upload", "Autoencoder", "Text Extraction", "9 Features", "Transformer V3", "CVAE Role Fit",
                 "AI Insights"])

    up_tab, paste_tab = st.tabs(["📤 Upload Resume", "📝 Paste Resume Text"])
    with up_tab:
        uploaded = st.file_uploader(
            "Drop a resume file", type=SUPPORTED_TYPES,
            help="PDF (text or scanned), DOCX, DOC, TXT, MD, RTF, ODT, HTML, PNG, JPG, WEBP, BMP, TIFF",
        )
        use_ae = st.toggle("Clean image / scanned resumes before reading", value=True)
        methods = {"Autoencoder (default)": "autoencoder"}
        if diffusion_ready():
            methods["Diffusion (optional)"] = "diffusion"
        c1, c2 = st.columns([1, 1.4])
        method = methods[c1.radio("Enhancement model", list(methods), horizontal=True, disabled=not use_ae)]
        guard = c2.checkbox("Use the original image if enhancement lowers OCR quality (local OCR check, slower)",
                            value=False, disabled=not use_ae)
    with paste_tab:
        pasted = st.text_area("Resume text", height=220, placeholder="Paste the candidate's resume here...")
        if os.path.exists(SAMPLE_TEXT) and st.button("Load sample resume"):
            st.session_state["sample_text"] = open(SAMPLE_TEXT, encoding="utf-8").read()
            st.rerun()
        if st.session_state.get("sample_text") and not pasted:
            pasted = st.session_state["sample_text"]
            st.caption("Sample resume loaded — click **Analyze Resume**.")

    if st.button("🚀 Analyze Resume", type="primary", use_container_width=True):
        if not uploaded and not (pasted and pasted.strip()):
            st.warning("Please upload a resume file or paste resume text.")
        else:
            run_analysis(uploaded, pasted, use_ae, method, guard)

    analysis = st.session_state.get("analysis")
    if analysis:
        render_analysis(analysis)


def run_analysis(uploaded, pasted, use_ae, method="autoencoder", guard=False):
    method_name = "Diffusion" if method == "diffusion" else "Autoencoder"
    result = {"file_name": uploaded.name if uploaded else "Pasted text", "originals": [], "denoised": [],
              "enhancer": method_name}
    with st.status("Running RecruitGen pipeline...", expanded=True) as status:
        if uploaded:
            st.write("📄 Reading resume file...")
            parsed = read_resume(uploaded)
            text = parsed["text"]
            if parsed["images"]:
                if use_ae:
                    st.write(f"🧹 {method_name}: enhancing resume image...")
                st.write("🔤 Extracting text from resume image...")
                text, result["originals"], result["denoised"] = extract_text_from_images(
                    parsed["images"], use_ae, method=method if use_ae else "none", guard=guard)
        else:
            text = pasted.strip()

        if not text or len(text.strip()) < 20:
            status.update(label="Could not read enough text from this resume", state="error")
            st.error("Not enough readable text was found. Try a clearer file or paste the resume text.")
            return
        result["text"] = text

        st.write("📐 Extracting 9 candidate features...")
        from models.transformer_v3.feature_extractor import extract_features
        features = extract_features(text)
        result["features"] = features

        st.write("🤖 Transformer V3: predicting job role...")
        role, conf, probs = load_transformer().predict(text, features)
        result["prediction"] = {"role": role, "confidence": conf, "probabilities": probs}

        st.write("🧬 CVAE: calculating role fit...")
        result["role_fit"] = load_cvae().role_fit(features)

        st.write("✨ Generating AI insights...")
        result["insights"] = cached_insights(text, role, tuple(features.items())) if ai_available() else None
        status.update(label="Analysis complete", state="complete", expanded=False)

    st.session_state.analysis = result
    st.session_state.last_features = result["features"]
    ins = result["insights"] or {}
    auth.save_screening(
        st.session_state.user["id"], ins.get("candidate_name", "Candidate"), result["file_name"],
        role, conf, result["role_fit"][0]["role"], result["role_fit"][0]["fit_score"], ins.get("overall_score"),
    )
    log("Resume analyzed", f"{result['file_name']} → {role} ({conf:.1f}%), best fit {result['role_fit'][0]['role']}")
    st.toast("Resume analysed successfully", icon="🎉")


def render_analysis(a):
    st.markdown("---")
    pred, fit = a["prediction"], a["role_fit"]
    ins = a.get("insights") or {}
    c = st.columns(4)
    with c[0]:
        ui.metric("Predicted Role", pred["role"], "Transformer V3", ui.PRIMARY)
    with c[1]:
        ui.metric("Confidence", f"{pred['confidence']:.2f}%", "Prediction probability", ui.score_color(pred["confidence"]))
    with c[2]:
        ui.metric("Best Role Fit", fit[0]["role"], f"CVAE fit score {fit[0]['fit_score']:.1f}%")
    with c[3]:
        s = ins.get("overall_score")
        ui.metric("Resume Score", f"{s}/100" if s is not None else "—", "AI quality assessment",
                  ui.score_color(s) if s is not None else None)

    st.markdown("<br>", unsafe_allow_html=True)
    name = ins.get("candidate_name") or os.path.splitext(a["file_name"])[0]
    pdf_button("Download full screening report (PDF)", "analysis",
               signature(a["file_name"], pred, fit, a.get("insights"), a["features"], len(a["text"])),
               lambda: pdf.build_analysis_pdf(a, feature_rows(a["features"]), feature_scores(a["features"]),
                                              prepared_by()),
               pdf.pdf_filename("screening report", name), primary=True)

    if a["denoised"]:
        enhancer = a.get("enhancer", "Autoencoder")
        with st.expander(f"🧹 {enhancer} — original vs enhanced resume", expanded=True):
            l, r = st.columns(2)
            l.image(a["originals"][0], caption="Original resume", use_column_width=True)
            r.image(a["denoised"][0], caption=f"Enhanced by {enhancer}", use_column_width=True)
            d1, d2 = st.columns(2)
            with d1:
                pdf_button("Clean resume (PDF)", "analysis_clean", signature(a["file_name"], a["denoised"][0].size,
                                                                              pred),
                           lambda: pdf.build_clean_resume_pdf(a["denoised"][0]),
                           pdf.pdf_filename("clean resume", name))
            d2.download_button("🖼️ Clean resume (PNG)", png_bytes(a["denoised"][0]), "denoised_resume.png",
                               "image/png", use_container_width=True)

    t1, t2, t3, t4, t5 = st.tabs(["🤖 Transformer V3", "🧬 CVAE Role Fit", "📐 9 Features", "✨ AI Insights",
                                  "📄 Resume Text"])
    with t1:
        l, r = st.columns([1, 1.4])
        with l:
            ui.role_card(1, pred["role"], pred["confidence"], "Highest-probability role")
        with r:
            st.plotly_chart(bar_chart(pred["probabilities"], "Role probabilities"), use_container_width=True)
    with t2:
        render_role_fit(fit)
    with t3:
        l, r = st.columns([1.2, 1])
        with l:
            features_table(a["features"])
        with r:
            st.plotly_chart(radar_chart(a["features"]), use_container_width=True)
    with t4:
        if ins:
            render_insights(ins)
        else:
            ai_unavailable_notice()
            if st.button("🔄 Retry AI insights"):
                cached_insights.clear()
                a["insights"] = cached_insights(a["text"], pred["role"], tuple(a["features"].items()))
                st.rerun()
    with t5:
        st.text_area("Extracted resume text", a["text"], height=320)


def render_role_fit(fit):
    st.markdown("#### 🏆 Top 3 Recommended Roles")
    cols = st.columns(3)
    for i, (col, r) in enumerate(zip(cols, fit[:3]), 1):
        with col:
            ui.role_card(i, r["role"], r["fit_score"],
                         f"CVAE {r['cvae_score']:.1f}% · Relevance {r['relevance_score']:.1f}%")
    st.markdown("#### 📋 Complete Role Ranking")
    df = pd.DataFrame([{
        "Rank": i, "Role": r["role"], "Fit Score (%)": round(r["fit_score"], 2),
        "CVAE Score (%)": round(r["cvae_score"], 2), "Relevance (%)": round(r["relevance_score"], 2),
        "Reconstruction MSE": round(r["mse"], 4),
    } for i, r in enumerate(fit, 1)])
    l, r = st.columns([1.3, 1])
    with l:
        st.dataframe(df, hide_index=True, use_container_width=True)
    with r:
        fig = go.Figure()
        fig.add_bar(name="Fit", x=df["Role"], y=df["Fit Score (%)"], marker_color=ui.PRIMARY)
        fig.add_bar(name="CVAE", x=df["Role"], y=df["CVAE Score (%)"], marker_color=ui.ACCENT)
        fig.add_bar(name="Relevance", x=df["Role"], y=df["Relevance (%)"], marker_color="#A78BFA")
        fig.update_layout(barmode="group", yaxis=dict(range=[0, 100], gridcolor="#1E293B"),
                          legend=dict(orientation="h", y=1.15))
        st.plotly_chart(chart_layout(fig), use_container_width=True)
    st.caption("Fit score = 40% CVAE role-conditioned reconstruction + 60% role-specific feature relevance.")


# ============================================================
# PAGE: BULK RESUME PROCESSING
# ============================================================

BULK_SECONDS_PER_PAGE = {"Diffusion": 65, "Autoencoder": 35, "None": 12}   # rough CPU medians incl. OCR (measured)


@st.cache_resource(show_spinner=False)
def bulk_manager():
    from core.bulk import BulkManager
    hub = model_hub()
    return BulkManager(lambda key: hub.get(key))      # models load once, serialized by the hub lock


def _fmt_secs(s):
    s = int(s)
    return f"{s // 3600}h {s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m {s % 60:02d}s" if s >= 60 else f"{s}s"


def bulk_rows_table(job):
    from core.bulk import BulkManager
    out = []
    for it in job["items"]:
        r = job["rows"].get(it["resume_id"], {})
        state = {"pending": "⏳ Waiting", "processing": "⚙️ Processing"}.get(it["state"], r.get("status", ""))
        if it["state"] == "processing" and job.get("current"):
            state = f"⚙️ {job['current']['stage']}"
        out.append({"Resume ID": it["resume_id"], "Filename": it["filename"], "Status": state,
                    "Enhancement": r.get("enhancement_method", ""), "Language": r.get("ocr_language", ""),
                    "Pages": r.get("pages_processed", ""), "Time (s)": r.get("processing_time_seconds"),
                    "Characters": len(r.get("extracted_text") or ""), "Error": r.get("error_message", "")})
    return pd.DataFrame(out)


def bulk_job_view(job, live):
    from core.bulk import BulkManager
    k = BulkManager.counts(job)
    c = st.columns(5)
    for col, (label, value, color) in zip(c, [("Total resumes", k["total"], None),
                                              ("Completed", k["completed"], ui.PRIMARY),
                                              ("Successful", k["success"], ui.SUCCESS),
                                              ("Partially successful", k["partial"], ui.WARNING),
                                              ("Failed", k["failed"], ui.DANGER)]):
        with col:
            ui.metric(label, value, "", color)
    st.progress(k["completed"] / max(1, k["total"]),
                text=f"{k['completed']} of {k['total']} resumes processed")
    started = job.get("started")
    elapsed = (job.get("finished") or time.time()) - started if started else 0
    if job["state"] in ("queued",):
        st.caption("Waiting in the queue — another batch is being processed.")
    elif job["state"] == "running":
        active = list((job.get("active") or {}).values())
        if job.get("cancel"):
            st.markdown("**Stopping…** finishing the current step, then the batch stops.")
        st.markdown(f"**{len(active)} of {job.get('workers_used', 1)} parallel workers busy** · batch elapsed "
                    f"**{_fmt_secs(elapsed)}**")
        for cur in active:
            st.caption(f"⚙️ {cur['resume_id']} · {cur['filename']} — {cur['stage']} "
                       f"({_fmt_secs(time.time() - cur['since'])})")
    elif job["state"] == "interrupted":
        st.caption("This batch was interrupted (the server restarted). Click **Resume / retry** to finish it.")
    st.dataframe(bulk_rows_table(job), hide_index=True, use_container_width=True,
                 column_config={"Error": st.column_config.TextColumn(width="large")})
    if live and job["state"] not in ("queued", "running"):
        st.rerun()                                     # leave the auto-refresh view once the batch is done


@st.fragment(run_every=2)
def bulk_live_view(batch_id):
    mgr = bulk_manager()
    job = mgr.get(st.session_state.user["id"], batch_id)
    if not job:
        return
    if st.button("⏹ Stop processing", key=f"bulk_stop_{batch_id}", disabled=bool(job.get("cancel")),
                 help="Finished resumes keep their results; the rest can be processed later with Resume."):
        mgr.stop(st.session_state.user["id"], batch_id)
        log("Bulk batch stopped", f"{_progress_text(job)} completed")
        st.rerun()
    bulk_job_view(job, live=True)


def _progress_text(job):
    from core.bulk import BulkManager
    k = BulkManager.counts(job)
    return f"{k['completed']}/{k['total']}"


def bulk_section():
    """Bulk Resume Processing — the Diffusion page: ZIP of up to 100 resumes → Diffusion enhancement →
    OCR → one CSV / Excel row per resume, processed by a background queue."""
    from core import bulk

    st.caption("Upload a ZIP of up to 100 resumes. Scanned pages and images are enhanced with the Diffusion model, then "
               "the text is read; text PDFs are read directly. Processing runs in the background — you can keep using "
               "the app.")
    user_id = st.session_state.user["id"]
    mgr = bulk_manager()
    jobs = mgr.list(user_id)

    with st.expander("📤 New batch", expanded=not jobs):
        up = st.file_uploader(f"ZIP with up to {bulk.LIMITS['max_files']} resumes (PDF, PNG, JPG, JPEG)", type=["zip"],
                              key="bulk_zip")
        c = st.columns(3)
        enhancement = c[0].radio("Image enhancement (scanned pages and images)", ["Diffusion", "None"],
                                 help="Diffusion had the lowest OCR error in the held-out test. Text PDFs are read "
                                      "directly and never enhanced.")
        engine = c[1].radio("OCR engine", ["Local OCR (offline, recommended for batches)",
                                           "AI vision OCR (existing engine, rate-limited)"])
        max_pages = c[2].slider("Max pages per PDF", 1, bulk.LIMITS["max_pdf_pages"], 5)
        c = st.columns([2, 1])
        guard = c[0].checkbox("Use the original page whenever enhancement lowers OCR quality", value=True)
        workers = c[1].slider("Parallel workers", 1, 4, bulk.default_workers(),
                              help="Resumes processed at the same time. The CPU is shared between the workers.")
        if up is not None:
            items, err = bulk.inspect_zip(up.getvalue())
            if err:
                st.error(err, icon="🚫")
            else:
                ok = [i for i in items if not i["error"]]
                bad = [i for i in items if i["error"]]
                pages_guess = sum(1 if i["kind"] == "image" else 2 for i in ok)
                st.markdown(f"**{len(items)} files found** · {len(ok)} ready to process · {len(bad)} rejected "
                            f"(they still get a row in the CSV) · rough time "
                            f"**{_fmt_secs(pages_guess * BULK_SECONDS_PER_PAGE[enhancement])}** on this CPU "
                            "(text PDFs are much faster).")
                if bad:
                    st.dataframe(pd.DataFrame([{"Filename": i["filename"], "Reason": i["error"]} for i in bad]),
                                 hide_index=True, use_container_width=True)
                if st.button("🚀 Start processing", type="primary", use_container_width=True):
                    opts = {"enhancement": enhancement.lower(), "guard": guard, "max_pages": max_pages,
                            "ocr_engine": "local" if engine.startswith("Local") else "ai_vision", "workers": workers}
                    st.session_state.bulk_selected = mgr.submit(user_id, up.name, items, opts)
                    log("Bulk batch started", f"{len(items)} files · {enhancement} · "
                                              f"{'local' if engine.startswith('Local') else 'AI vision'} OCR")
                    st.rerun()

    jobs = mgr.list(user_id)
    if not jobs:
        return
    labels = {j["id"]: f"{j['zip_name']} · {datetime.fromtimestamp(j['created']).strftime('%d %b %H:%M')} · "
                       f"{len(j['items'])} files · {j['state']}" for j in jobs}
    ids = list(labels)
    selected = st.session_state.get("bulk_selected")
    batch_id = st.selectbox("Batch", ids, index=ids.index(selected) if selected in ids else 0,
                            format_func=labels.get, key="bulk_batch_pick")
    st.session_state.bulk_selected = batch_id
    job = mgr.get(user_id, batch_id)
    if job["state"] in ("queued", "running"):
        bulk_live_view(batch_id)
        return

    bulk_job_view(job, live=False)
    k = mgr.counts(job)
    if job["state"] == "finished":
        took = _fmt_secs((job["finished"] or time.time()) - (job["started"] or job["created"]))
        st.success(f"Batch complete in {took}: {k['success']} successful · {k['partial']} partially successful · "
                   f"{k['failed']} failed — {k['total']} rows in the results.", icon="✅")
    elif job["state"] == "stopped":
        st.warning(f"Batch stopped by you — {k['completed']} of {k['total']} resumes processed. The downloads contain "
                   "every row (unprocessed files are marked as not processed). Click **Resume / retry** to finish.",
                   icon="⏹")
    rows, structured = mgr.results(user_id, batch_id)
    stem = os.path.splitext(job["zip_name"])[0][:40] or "batch"
    d = st.columns(3)
    d[0].download_button("⬇️ Extracted text (CSV)", bulk.to_csv(rows, bulk.TEXT_COLUMNS), f"{stem}_extracted_text.csv",
                         "text/csv", use_container_width=True, type="primary",
                         on_click=log, args=("Bulk results exported", f"text CSV · {len(rows)} rows"))
    d[1].download_button("⬇️ Candidate fields (CSV)", bulk.to_csv(structured, bulk.STRUCTURED_COLUMNS),
                         f"{stem}_candidates.csv", "text/csv", use_container_width=True,
                         on_click=log, args=("Bulk results exported", f"structured CSV · {len(rows)} rows"))
    d[2].download_button("⬇️ Excel report (dashboard + profiles)", bulk.to_excel(rows, structured, job),
                         f"{stem}_results.xlsx",
                         "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True,
                         on_click=log, args=("Bulk results exported", f"Excel · {len(rows)} rows"))
    st.caption("Structured fields (name, email, phone, skills, education, experience) are filled only when a clear "
               "pattern is found; otherwise they are left empty.")
    c = st.columns(2)
    retryable = sum(1 for it in job["items"] if it["state"] == "pending" or (
        not it["error"] and job["rows"].get(it["resume_id"], {}).get("status") == "Failed"))
    if c[0].button(f"🔁 Resume / retry ({retryable} files)", use_container_width=True, disabled=not retryable):
        mgr.retry(user_id, batch_id)
        log("Bulk batch retried", f"{retryable} files")
        st.rerun()
    if c[1].button("🗑️ Delete batch and its files", use_container_width=True):
        mgr.delete(user_id, batch_id)
        st.session_state.pop("bulk_selected", None)
        st.rerun()


# ============================================================
# PAGE: AUTOENCODER
# ============================================================

def run_autoencoder_clean(image, source_name, smart, run_ocr):
    """Step 1 AI image analysis (fallback: local) → Step 2 Autoencoder cleaning → Step 3 local OCR comparison."""
    from models.autoencoder.autoencoder import mse, psnr, ssim

    original = image.convert("L")
    diagnosis = None
    if smart:
        with st.spinner("Step 1 · analysing the resume image..."):
            diagnosis = (diagnose_resume_image(original) if ai_available() else None) or local_image_diagnosis(original)
    bar = st.progress(0.0, text="Step 2 · Autoencoder: encoder → latent space → decoder...")
    t = time.time()
    try:
        _, cleaned = load_autoencoder().denoise(original, progress=lambda p: bar.progress(
            p, text=f"Step 2 · cleaning the resume... {p * 100:.0f}%"))
        error = None
    except Exception as exc:                 # cleaning failure never blocks the page: keep the original
        cleaned, error = None, str(exc)
    seconds = time.time() - t
    outputs = {"Original": (original, None)}
    if cleaned is not None:
        from models.autoencoder.autoencoder import CONTENT_MIN_RATIO, content_ratio
        if content_ratio(original, cleaned) < CONTENT_MIN_RATIO:      # never show a result that erased the resume
            cleaned, error = None, "too_blurred"
        else:
            outputs["Cleaned"] = (cleaned, seconds)
    ocr = {}
    if run_ocr:
        bar.progress(1.0, text="Step 3 · reading the text...")
        ocr = {name: local_ocr(img) for name, (img, _) in outputs.items()}
    bar.empty()
    metrics = {"den": (mse(original, cleaned), psnr(original, cleaned), ssim(original, cleaned))} if cleaned else {}
    st.session_state.ae_clean = {"outputs": outputs, "ocr": ocr, "diagnosis": diagnosis, "source": source_name,
                                 "error": error, "metrics": metrics}
    st.session_state.pop("ae_vision", None)
    log("Autoencoder run", f"Clean · {source_name} · {original.size[0]}×{original.size[1]} · {seconds:.1f}s")


def page_autoencoder():
    ae = load_autoencoder()
    ui.hero("Autoencoder — Resume Image Cleaning",
            "Upload a resume → AI image analysis → Encoder → Latent Space → Decoder → clean resume → OCR text",
            eyebrow="Model 1 of 5 · " + ("encoder_v2.h5 + decoder_v2.h5" if ae.version == "v2"
                                         else "encoder.h5 + decoder.h5"))
    c = st.columns(4)
    with c[0]:
        ui.metric("Autoencoder", "Ready", "v2 · fine-tuned on real resumes" if ae.version == "v2" else "v1 · original",
                  ui.SUCCESS)
    with c[1]:
        ui.metric("Parameters", f"{ae.num_params:,}", "U-Net encoder + decoder")
    with c[2]:
        ui.metric("Latent Space", "32 × 32 × 256", "per 256 px tile")
    with c[3]:
        ui.metric("Runs", "Locally", "AI only analyses the image")

    clean_tab, demo_tab = st.tabs(["🧹 Clean My Resume", "🧪 Evaluation Demo"])
    with clean_tab:
        uploaded = st.file_uploader("Upload a resume image or scanned PDF", type=IMAGE_TYPES + ["pdf"], key="ae_upload")
        with st.expander("⚙️ Settings", expanded=False):
            c = st.columns(2)
            smart = c[0].checkbox("🧠 Smart mode — AI image analysis first, then the Autoencoder", value=True,
                                  key="ae_smart")
            run_ocr = c[1].checkbox("Compare OCR text (local OCR)", value=True, key="ae_ocr")
            use_sample = os.path.exists(BLURRED_SAMPLE) and st.checkbox("Use sample resume image (blurred scan)",
                                                                         value=False, key="ae_sample")
            rerun = st.button("🔁 Clean again", use_container_width=True, key="ae_rerun")

        image, source_name, key = None, "", None
        if uploaded:
            try:
                parsed = read_resume(uploaded)
                if parsed["images"]:
                    image, source_name, key = parsed["images"][0], uploaded.name, f"{uploaded.name}:{uploaded.size}"
                else:
                    st.caption("This PDF has selectable text, not a scanned image — upload an image or a scanned PDF.")
            except Exception as exc:
                st.caption(f"Could not read this file: {exc}")
        elif use_sample:
            from PIL import Image
            image, source_name, key = Image.open(BLURRED_SAMPLE).convert("RGB"), "sample image", "sample"
        if image is not None and (rerun or st.session_state.get("ae_auto_key") != key):
            st.session_state.ae_auto_key = key        # a new upload is cleaned automatically, once
            run_autoencoder_clean(image, source_name, smart, run_ocr)

        res = st.session_state.get("ae_clean")
        if res and image is not None:
            outputs, ocr = res["outputs"], res["ocr"]
            if res["error"] == "too_blurred":
                st.caption("This image is too blurred to restore — the text detail is no longer in the file, so the "
                           "original is kept and no resume content is lost. A sharper scan or photo will clean properly.")
            elif res["error"]:
                st.caption(f"The Autoencoder could not process this image ({res['error']}); the original is kept.")
            dg = res.get("diagnosis")
            if dg:
                st.markdown(f"**🧠 Image analysis:** {', '.join(dg['degradations']) or 'no visible degradation'} · "
                            f"severity **{dg['severity']}** · text legible **{'yes' if dg['text_legible'] else 'no'}**")
                if not dg["text_legible"]:
                    st.caption("The text in this image is barely legible — cleaning cannot restore detail that is no "
                               "longer in the image; a sharper scan gives better OCR.")
            cols = st.columns(len(outputs))
            for col, (name, (img, secs)) in zip(cols, outputs.items()):
                col.image(img, caption=f"{name} · {img.size[0]}×{img.size[1]}" + (f" · {secs:.1f}s" if secs else ""),
                          use_column_width=True)
            if ocr:
                st.dataframe(pd.DataFrame([ocr_row(n, img, ocr.get(n), secs) for n, (img, secs) in outputs.items()]),
                             hide_index=True, use_container_width=True)
            scored = {n: o["score"] for n, o in ocr.items() if o}
            best = max(scored, key=scored.get) if scored else list(outputs)[-1]
            if scored and best == "Original":
                st.caption("Cleaning did not improve OCR for this image, so the original is selected for text "
                           "extraction (the cleaned image is still available below).")
            choice = st.radio("Image to use for text extraction", list(outputs), index=list(outputs).index(best),
                              horizontal=True, key="ae_choice",
                              help="Defaults to the image read best by OCR. The original file is always preserved.")
            if ocr:
                with st.expander("🔤 OCR text", expanded=True):
                    tcols = st.columns(len(ocr))
                    for col, (name, o) in zip(tcols, ocr.items()):
                        col.markdown(f"**{name}**")
                        col.text_area(name, (o or {}).get("text", ""), height=240, label_visibility="collapsed",
                                      key=f"ae_ocr_{name}")
            if "Cleaned" in outputs:
                den = outputs["Cleaned"][0]
                report = {"mode": "denoise", "orig": outputs["Original"][0], "den": den, "metrics": res["metrics"]}
                sig = signature("clean", res["metrics"], den.size, res["source"])
                d = st.columns(4)
                with d[0]:
                    pdf_button("Clean resume (PDF)", "ae_clean_pdf", sig, lambda: pdf.build_clean_resume_pdf(den),
                               pdf.pdf_filename("clean resume", res["source"]), primary=True)
                d[1].download_button("🖼️ Clean resume (PNG)", png_bytes(den), "clean_resume.png", "image/png",
                                     use_container_width=True, key="ae_png")
                with d[2]:
                    pdf_button("Cleaning report (PDF)", "ae_clean_report", sig,
                               lambda: pdf.build_denoise_report_pdf(report, prepared_by(), res["source"]),
                               pdf.pdf_filename("denoising report", res["source"]))
                if ocr.get(choice):
                    d[3].download_button("⬇️ OCR text (TXT)", ocr[choice]["text"].encode(), "resume_ocr.txt",
                                         "text/plain", use_container_width=True, key="ae_txt")
            if st.button(f"🔤 Read the {choice.lower()} image with the AI vision OCR", use_container_width=True,
                         key="ae_vision_btn"):
                with st.spinner("Reading text..."):
                    st.session_state.ae_vision = (choice, image_to_text(outputs[choice][0]))
            if st.session_state.get("ae_vision"):
                who, text = st.session_state.ae_vision
                st.text_area(f"AI vision OCR — {who}", text or "(no text returned — use the local OCR text above)",
                             height=220, key="ae_vision_text")

    with demo_tab:
        autoencoder_demo()


def autoencoder_demo():
    """Evaluation demo: add synthetic noise to a clean resume → denoise → MSE / PSNR / SSIM against the clean page."""
    uploaded = st.file_uploader("Upload a clean resume image or scanned PDF", type=IMAGE_TYPES + ["pdf"],
                                key="ae_demo_upload")
    noise = st.slider("Synthetic noise level", 0.02, 0.20, 0.08, 0.01)
    image = None
    if uploaded:
        parsed = read_resume(uploaded)
        if parsed["images"]:
            image = parsed["images"][0]
        else:
            st.caption("This PDF has selectable text, not a scanned image — upload an image or a scanned PDF.")
    elif os.path.exists(CLEAN_SAMPLE) and st.checkbox("Use sample resume image (clean)", value=False,
                                                      key="ae_demo_sample"):
        from PIL import Image
        image = Image.open(CLEAN_SAMPLE).convert("RGB")
    if image is None:
        return

    if st.button("✨ Run evaluation demo", type="primary", use_container_width=True):
        from models.autoencoder.autoencoder import add_noise, mse, psnr, ssim
        bar = st.progress(0.0, text="Encoding → latent space → decoding...")
        update = lambda p: bar.progress(p, text=f"Processing tiles... {p * 100:.0f}%")  # noqa: E731
        clean = image.convert("L")
        noisy = add_noise(clean, noise)
        _, den = load_autoencoder().denoise(noisy, progress=update)
        st.session_state.ae_result = {"mode": "eval", "clean": clean, "noisy": noisy, "den": den, "metrics": {
            "noisy": (mse(clean, noisy), psnr(clean, noisy), ssim(clean, noisy)),
            "den": (mse(clean, den), psnr(clean, den), ssim(clean, den)),
        }}
        bar.empty()
        st.session_state.ae_source = uploaded.name if uploaded else "sample image"
        log("Autoencoder run", f"Evaluation demo · {st.session_state.ae_source} · noise {noise}")

    res = st.session_state.get("ae_result")
    if not res:
        return
    if res["mode"] == "eval":
        cols = st.columns(3)
        cols[0].image(res["clean"], caption="Clean (ground truth)", use_column_width=True)
        cols[1].image(res["noisy"], caption="Noisy input", use_column_width=True)
        cols[2].image(res["den"], caption=f"Denoised by Autoencoder · {res['den'].size[0]}×{res['den'].size[1]}",
                      use_column_width=True)
        n, d = res["metrics"]["noisy"], res["metrics"]["den"]
        m = st.columns(3)
        with m[0]:
            ui.metric("MSE", f"{d[0]:.5f}", f"Noisy: {n[0]:.5f}", ui.SUCCESS if d[0] < n[0] else ui.WARNING)
        with m[1]:
            ui.metric("PSNR", f"{d[1]:.2f} dB", f"Noisy: {n[1]:.2f} dB", ui.SUCCESS if d[1] > n[1] else ui.WARNING)
        with m[2]:
            ui.metric("SSIM", f"{d[2]:.4f}", f"Noisy: {n[2]:.4f}", ui.SUCCESS if d[2] > n[2] else ui.WARNING)
    else:
        l, r = st.columns(2)
        o, dn = res["orig"].size, res["den"].size
        upscale = f" · {dn[0] / o[0]:.1f}× super-resolved" if dn[0] > o[0] else ""
        l.image(res["orig"], caption=f"Original · {o[0]}×{o[1]}", use_column_width=True)
        r.image(res["den"], caption=f"Denoised by Autoencoder · {dn[0]}×{dn[1]}{upscale}", use_column_width=True)
        d = res["metrics"]["den"]
        m = st.columns(3)
        with m[0]:
            ui.metric("MSE", f"{d[0]:.5f}", "Original vs reconstructed")
        with m[1]:
            ui.metric("PSNR", f"{d[1]:.2f} dB", "Peak signal-to-noise ratio")
        with m[2]:
            ui.metric("SSIM", f"{d[2]:.4f}", "Structural similarity")
    st.markdown("<br>", unsafe_allow_html=True)
    sig = signature(res["mode"], res["metrics"], res["den"].size)
    source = st.session_state.get("ae_source", "")
    d1, d2, d3 = st.columns(3)
    with d1:
        pdf_button("Denoising report (PDF)", "ae_report", sig,
                   lambda: pdf.build_denoise_report_pdf(res, prepared_by(), source),
                   pdf.pdf_filename("denoising report", source), primary=True)
    with d2:
        pdf_button("Clean resume (PDF)", "ae_clean", sig, lambda: pdf.build_clean_resume_pdf(res["den"]),
                   pdf.pdf_filename("clean resume", source))
    d3.download_button("🖼️ Clean resume (PNG)", png_bytes(res["den"]), "denoised_resume.png", "image/png",
                       use_container_width=True)


# ============================================================
# PAGE: DIFFUSION IMAGE ENHANCEMENT
# ============================================================

DIFFUSION_TRAIN_CMD = "python -m models.diffusion.train\npython -m models.diffusion.evaluate"


def ocr_row(name, image, ocr, seconds=None, reference_text="", clean=None):
    from models.diffusion.evaluate import cer, ink_f1, psnr, ssim, wer
    row = {"Image": name, "Time (s)": round(seconds, 2) if seconds is not None else None}
    if ocr:
        row.update({"OCR score": round(ocr["score"], 1), "OCR confidence": round(ocr["confidence"], 3),
                    "Text boxes": ocr["boxes"], "Words": len(ocr["text"].split())})
    if reference_text.strip() and ocr:
        row.update({"CER": round(cer(reference_text, ocr["text"]), 4), "WER": round(wer(reference_text, ocr["text"]), 4)})
    if clean is not None and image is not clean:
        row.update({"PSNR (dB)": round(psnr(clean, image), 2), "SSIM": round(ssim(clean, image), 4),
                    "Ink F1": round(ink_f1(clean, image), 4)})
    return row


def page_diffusion():
    ui.hero("Diffusion",
            "Upload a ZIP of up to 100 resumes → local Diffusion enhancement → OCR text → CSV / Excel",
            eyebrow="Model 5 of 5 · diffusion_unet.safetensors")

    diff = None
    if diffusion_ready():
        try:
            diff = load_diffusion()
        except Exception:
            diff = None
    c = st.columns(4)
    with c[0]:
        if diff is None:
            ui.metric("Diffusion Model", "Preparing", "Model files are being written", ui.WARNING)
        elif diff.config["training"].get("provisional"):
            t = diff.config["training"]
            ui.metric("Diffusion Model", "Ready", f"provisional · training step {t.get('steps_done')} of {t.get('steps')}",
                      ui.SUCCESS)
        else:
            ui.metric("Diffusion Model", "Ready", f"trained {diff.config['training'].get('steps', '—')} steps", ui.SUCCESS)
    with c[1]:
        ui.metric("Parameters", f"{diff.num_params:,}" if diff else "—", "Conditional U-Net")
    with c[2]:
        ui.metric("Device", diff.device.type.upper() if diff else "CPU", "Enhancement runs locally")
    with c[3]:
        ui.metric("Sampling", f"{diff.steps} DDIM steps" if diff else "—", "Per enhanced page")

    st.markdown("### 📦 Bulk Resume Processing")
    bulk_section()


# ============================================================
# PAGE: CVAE
# ============================================================

def page_cvae():
    ui.hero("CVAE — Synthetic Candidate Generation & Role Fit",
            "Candidate Features → Encoder → Latent Space → Decoder + Condition → Role Fit / Synthetic Candidate",
            eyebrow="Model 2 of 5 · vae_model.h5 + vae.weights.h5")
    fit_tab, gen_tab = st.tabs(["🎯 Role Fit Analysis", "🧪 Synthetic Candidate Generation"])

    with fit_tab:
        base = st.session_state.get("last_features", DEFAULT_FEATURES)
        if "last_features" in st.session_state:
            st.caption("Pre-filled with the features of the last analysed resume.")
        candidate = {}
        cols = st.columns(3)
        for i, (key, label, meaning, lo, hi, step) in enumerate(FEATURES):
            with cols[i % 3]:
                val = base.get(key, DEFAULT_FEATURES[key])
                if isinstance(step, int):
                    candidate[key] = st.number_input(label, int(lo), int(hi), int(val), int(step), help=meaning,
                                                     key=f"cvae_{key}")
                else:
                    candidate[key] = st.slider(label, float(lo), float(hi), float(val), float(step), help=meaning,
                                               key=f"cvae_{key}")
        if st.button("🧬 Calculate Role Fit", type="primary", use_container_width=True):
            with st.spinner("CVAE encoding candidate and reconstructing for each role..."):
                st.session_state.cvae_fit = load_cvae().role_fit(candidate)
                st.session_state.cvae_fit_input = dict(candidate)
            top = st.session_state.cvae_fit[0]
            log("CVAE role fit", f"Best fit {top['role']} ({top['fit_score']:.1f}%)")
        if st.session_state.get("cvae_fit"):
            fit = st.session_state.cvae_fit
            render_role_fit(fit)
            profile = st.session_state.get("cvae_fit_input", candidate)
            pdf_button("Download role-fit report (PDF)", "cvae_fit", signature(fit, profile),
                       lambda: pdf.build_role_fit_pdf(fit, feature_rows(profile), feature_scores(profile),
                                                      prepared_by()),
                       pdf.pdf_filename("role fit", fit[0]["role"]), primary=True)

    with gen_tab:
        from models.cvae.cvae import ROLE_NAMES
        c = st.columns(3)
        role = c[0].selectbox("Target role (condition)", ROLE_NAMES)
        n = c[1].slider("Number of candidates", 1, 25, 8)
        seed = c[2].number_input("Random seed", 0, 9999, 42)
        if st.button("🧪 Generate Synthetic Candidates", type="primary", use_container_width=True):
            with st.spinner("Sampling latent space and decoding candidates..."):
                rows = load_cvae().generate(role, n, int(seed))
            st.session_state.cvae_gen = (role, rows)
            st.session_state.cvae_gen_seed = int(seed)
            log("CVAE generation", f"{n} synthetic candidates for {role} (seed {seed})")
        if st.session_state.get("cvae_gen"):
            role, rows = st.session_state.cvae_gen
            df = pd.DataFrame(rows).rename(columns={**FEATURE_LABEL, "role_relevance": f"{role} Relevance (%)"})
            df.insert(0, "Candidate", [f"SYN-{i:03d}" for i in range(1, len(df) + 1)])
            st.dataframe(df, hide_index=True, use_container_width=True)
            mean_profile = pd.DataFrame(rows)[[f[0] for f in FEATURES]].mean().to_dict()
            l, r = st.columns(2)
            with l:
                st.plotly_chart(radar_chart(mean_profile), use_container_width=True)
            with r:
                st.plotly_chart(bar_chart(feature_scores(mean_profile), "Average generated profile (0-100)", ""),
                                use_container_width=True)
            g1, g2 = st.columns(2)
            with g1:
                gen_seed = st.session_state.get("cvae_gen_seed")
                pdf_button("Synthetic candidates (PDF)", "cvae_gen", signature(role, rows, gen_seed),
                           lambda: pdf.build_synthetic_pdf(role, rows, feature_scores(mean_profile), prepared_by(),
                                                           gen_seed),
                           pdf.pdf_filename("synthetic candidates", role), primary=True)
            g2.download_button("⬇️ Synthetic candidates (CSV)", df.to_csv(index=False).encode(),
                               f"synthetic_{role.replace('/', '_').replace(' ', '_').lower()}.csv", "text/csv",
                               use_container_width=True)


# ============================================================
# PAGE: GAN CANDIDATE GENERATOR
# ============================================================

SYNTHETIC_LABEL = "SYNTHETIC (GAN)"
REAL_LABEL = "REAL (Kaggle)"
GAN_TRAIN_CMD = "python -m models.gan.train"


def gan_table(rows, start=1):
    """Generated rows -> display table. Every row carries an explicit synthetic label."""
    df = pd.DataFrame(rows).rename(columns={**FEATURE_LABEL, "job_role": "Job Role"})
    df.insert(0, "Candidate", [f"GAN-SYN-{i:04d}" for i in range(start, start + len(df))])
    df.insert(1, "Data Source", SYNTHETIC_LABEL)
    return df


def _slug(text):
    return text.replace("/", "_").replace(" ", "_").lower()


def _mean_sd(stat):
    return f"{stat['mean']:.4f} ± {stat['std']:.4f}"


def confusion_heatmap(matrix, labels, title):
    fig = go.Figure(go.Heatmap(z=matrix, x=labels, y=labels, colorscale="Purples", showscale=False,
                               hovertemplate="True %{y}<br>Predicted %{x}<br>%{z} resumes<extra></extra>"))
    fig.update_layout(title=title, xaxis=dict(title="Predicted", tickangle=45, tickfont=dict(size=9)),
                      yaxis=dict(title="True", autorange="reversed", tickfont=dict(size=9)))
    return chart_layout(fig, 620)


def gan_results_summary(m):
    """Compact view of metrics.json for the AI explanation (numbers only, no raw reports)."""
    aug = m["augmentation"]
    return {
        "dataset_rows": m["dataset"]["rows"], "job_roles": len(m["role_names"]), "split_sizes": aug["split_sizes"],
        "real_only_test": {k: round(v["mean"], 4) for k, v in aug["real_only"]["test"].items()},
        "augmented": {name: {"strategy": s["strategies"][s["chosen_strategy"]]["label"],
                             "test": {k: round(v["mean"], 4)
                                      for k, v in s["strategies"][s["chosen_strategy"]]["test"].items()},
                             "macro_f1_delta": round(s["test_f1_macro_delta_vs_real"]["mean"], 4),
                             "seeds_improved": f"{s['test_f1_macro_delta_vs_real']['seeds_improved']}/"
                                               f"{s['test_f1_macro_delta_vs_real']['seeds']}"}
                      for name, s in aug["samplers"].items()},
        "quality": {name: {k: round(v, 4) for k, v in q["quality"].items() if isinstance(v, float)}
                    for name, q in m["quality"].items()},
        "diversity": {name: {k: round(v, 4) for k, v in q["diversity"].items() if isinstance(v, float)}
                      for name, q in m["quality"].items()},
        "tstr_test_macro_f1": {k: round(v["f1_macro"]["mean"], 4) for k, v in m["tstr"].items()},
    }


HYBRID_TRAIN_CMD = "python -m models.gan.hybrid_train"
AI_STATUS_TEXT = {"unavailable": "the AI service is not configured", "offline": "no internet connection",
                  "rate_limited": "the AI service rate limit was reached", "invalid_json": "the AI response was invalid",
                  "error": "the AI service returned an error"}
MAX_DESCRIBED = 20


@st.cache_resource(show_spinner=False)
def load_hybrid():
    from models.gan.hybrid import HybridGenerator
    return HybridGenerator()


def hybrid_table(rows, descriptions):
    from models.gan.hybrid import DESCRIPTION_SOURCE, GAN_SOURCE
    df = pd.DataFrame(rows).rename(columns={**FEATURE_LABEL, "job_role": "Job Role"})
    df.insert(0, "Candidate", [f"HYB-SYN-{i:04d}" for i in range(1, len(df) + 1)])
    df.insert(1, "Data Source", "SYNTHETIC")
    df.insert(2, "Feature Values", GAN_SOURCE)
    df["Description Source"] = [DESCRIPTION_SOURCE if i in descriptions else "—" for i in range(len(rows))]
    for key, label in (("title", "Title"), ("skills", "Skills"), ("projects", "Projects"),
                       ("certifications", "Certifications"), ("degree", "Degree"), ("summary", "Summary")):
        df[label] = [descriptions.get(i, {}).get(key, "") for i in range(len(rows))]
    return df


def hybrid_generator_section():
    from models.gan import hybrid as hy
    from models.gan.gan import checkpoint_available

    if not checkpoint_available(hy.HYBRID_DIR):
        st.info("The hybrid 5-role generator has not been trained yet:", icon="ℹ️")
        st.code(f"cd RecruitGen\n{HYBRID_TRAIN_CMD}", language="bash")
        return
    try:
        hyb = load_hybrid()
    except Exception as exc:
        st.error(f"The hybrid generator could not be loaded: {exc}", icon="❌")
        return
    info = hyb.gan.training_info
    c = st.columns(4)
    with c[0]:
        ui.metric("Hybrid Model", "Loaded", f"best iter {info.get('best_iteration', '—')}", ui.SUCCESS)
    with c[1]:
        ui.metric("Job Roles", len(hyb.role_names), "RecruitGen roles")
    with c[2]:
        ui.metric("Validated Seed Profiles", f"{info.get('train_rows', 0):,}", "GAN training rows")
    with c[3]:
        ui.metric("Generator Parameters", f"{hyb.num_params:,}", "Conditional WGAN-GP")
    st.caption("Pipeline: AI-assisted seed profiles → feature extractor validation → GAN → plausibility, duplicate and "
               "preference filters → optional AI-assisted descriptions, kept only if they reproduce the GAN numbers. "
               "Feature values always come from the GAN.")

    with st.form("hybrid_form"):
        c = st.columns(3)
        role = c[0].selectbox("Job role", hy.ROLES)
        n = c[1].slider("Number of profiles", 1, 100, 10)
        seed = c[2].number_input("Random seed", 0, 99999, 42)
        c = st.columns(2)
        exp_range = c[0].slider("Experience range (years)", 0.0, 20.0, (0.0, 20.0), 0.5)
        prefs = c[1].multiselect("Skill preferences", list(hy.SKILL_FEATURES))
        min_score = st.slider("Minimum score for the preferred skills", 15, 100, 30, 15, disabled=False)
        with st.expander("⚙️ Generation settings"):
            g = st.columns(2)
            diversity = g[0].slider("Diversity (noise scale)", 0.5, 1.5, 1.0, 0.05,
                                    help="Below 1 = more typical profiles, above 1 = more varied profiles.")
            describe = g[1].checkbox(f"Add AI-assisted descriptions (first {MAX_DESCRIBED} profiles)", value=True)
            plausibility = g[0].checkbox("Plausibility filter (reject combinations far from the validated seed data)",
                                         value=True)
            dedupe = g[1].checkbox("Remove duplicate profiles", value=True)
            no_copy = g[1].checkbox("Exclude exact copies of training rows", value=True)
        submitted = st.form_submit_button("🧬 Generate Hybrid Profiles", type="primary", use_container_width=True)

    if submitted:
        constraints = {hy.SKILL_FEATURES[p]: float(min_score) for p in prefs}
        with st.spinner("Sampling the GAN and filtering profiles..."):
            rows, stats = hyb.generate(role, n, int(seed), diversity, exp_range, constraints, plausibility, dedupe,
                                       no_copy)
        descriptions, summary = {}, None
        if describe and rows:
            with st.spinner("Writing AI-assisted descriptions and validating them against the GAN numbers..."):
                try:
                    descriptions, summary = hy.describe_rows(role, rows[:MAX_DESCRIBED])
                except Exception as exc:
                    summary = {"validated": 0, "failed_validation": 0, "missing": 0, "status": {"error": 1},
                               "detail": str(exc)}
        st.session_state.hyb_gen = {"role": role, "rows": rows, "stats": stats, "descriptions": descriptions,
                                    "summary": summary, "seed": int(seed), "requested": n}
        log("Hybrid GAN generation", f"{len(rows)}/{n} profiles for {role} (seed {seed}) · "
                                     f"{len(descriptions)} validated descriptions")

    res = st.session_state.get("hyb_gen")
    if not res:
        return
    rows, stats, summary = res["rows"], res["stats"], res["summary"]
    m = st.columns(4)
    with m[0]:
        ui.metric("Profiles", f"{len(rows)} / {res['requested']}", "Accepted / requested",
                  ui.SUCCESS if len(rows) == res["requested"] else ui.WARNING)
    with m[1]:
        ui.metric("GAN Samples", f"{stats['sampled']:,}", "Drawn to fill the request")
    with m[2]:
        rate = stats["accepted"] / max(1, stats["sampled"])
        ui.metric("Acceptance Rate", f"{rate:.0%}", "Passed every enabled filter")
    with m[3]:
        if summary is None:
            ui.metric("Descriptions", "Off", "GAN-only output")
        else:
            ui.metric("Descriptions", f"{summary['validated']} / {min(len(rows), MAX_DESCRIBED)}",
                      f"Validated · {summary['failed_validation']} rejected")
    if len(rows) < res["requested"]:
        st.warning(f"Only {len(rows)} of {res['requested']} profiles passed the filters — widen the experience range, "
                   "lower the skill minimum or raise the diversity.", icon="⚠️")
    if summary is not None:
        problems = [s for s in summary["status"] if s not in ("ok", "cached")]
        if problems and not summary["validated"]:
            st.info("AI-assisted descriptions are unavailable right now ("
                    + ", ".join(AI_STATUS_TEXT.get(p, p) for p in problems) + ") — showing GAN-only profiles.", icon="⏳")
    if stats["rejected"]:
        with st.expander("Filter log (rejected GAN samples)"):
            st.dataframe(pd.DataFrame([{"Reason": k, "Samples": v} for k, v in stats["rejected"].items()]),
                         hide_index=True, use_container_width=True)
    if not rows:
        return
    df = hybrid_table(rows, res["descriptions"])
    st.dataframe(df, hide_index=True, use_container_width=True)
    mean_profile = pd.DataFrame(rows)[[f[0] for f in FEATURES]].mean().to_dict()
    l, r = st.columns(2)
    with l:
        st.plotly_chart(radar_chart(mean_profile), use_container_width=True)
    with r:
        st.plotly_chart(bar_chart(feature_scores(mean_profile), "Average hybrid profile (0-100)", ""),
                        use_container_width=True)
    st.download_button("⬇️ Hybrid synthetic profiles (CSV)", df.to_csv(index=False).encode(),
                       f"synthetic_hybrid_{_slug(res['role'])}_seed{res['seed']}.csv", "text/csv",
                       use_container_width=True, type="primary",
                       on_click=log, args=("Hybrid CSV exported", f"{len(df)} profiles · {res['role']}"))


def hybrid_comparison_section(original_gan):
    from models.gan import hybrid as hy
    from models.gan.gan import load_metrics

    st.markdown("### ⚖️ Live comparison — original vs improved output")
    res = st.session_state.get("hyb_gen")
    if res and res["rows"]:
        n = len(res["rows"])
        role = "Information Technology" if "Information Technology" in original_gan.role_names \
            else original_gan.role_names[0]
        orig_vals, _ = original_gan.generate_array({role: n}, seed=res["seed"])
        new_vals = np.array([[r[f] for f in hy.FEATURE_NAMES] for r in res["rows"]], dtype=float)
        live = {f"Original GAN · {role}": hy.batch_stats(orig_vals),
                f"Hybrid · {res['role']}": hy.batch_stats(new_vals)}
        st.dataframe(pd.DataFrame(live).T.rename(columns=COMPARE_LABELS).round(3), use_container_width=True)
        st.caption(f"Same batch size ({n}) and seed. Generate hybrid profiles in the first tab to refresh this table.")
    else:
        st.info("Generate profiles in the 🧬 Hybrid Generator tab to compare them with the original GAN here.", icon="ℹ️")

    metrics = load_metrics(hy.HYBRID_DIR)
    if not metrics:
        st.info(f"No hybrid evaluation report yet — run `{HYBRID_TRAIN_CMD}`.", icon="ℹ️")
        return
    if metrics.get("quick_run"):
        st.warning("This report comes from a QUICK test run — retrain without `--quick` for reportable numbers.", icon="⚠️")
    st.caption(f"Computed {metrics['generated_at']} · {metrics['note']}")

    st.markdown("### 📋 Original vs improved — full evaluation batch")
    st.dataframe(pd.DataFrame(metrics["original_vs_improved"]).T.rename(columns=COMPARE_LABELS).round(4),
                 use_container_width=True)
    seed_rep = metrics["seed_dataset"]
    st.caption(f"Seed data: {seed_rep['kept']} of {seed_rep['requested']} AI-assisted profiles passed validation "
               f"({', '.join(f'{k}: {v}' for k, v in sorted(seed_rep['rejected'].items(), key=lambda kv: -kv[1]))}).")

    st.markdown("### 🔬 Feature distributions & similarity (held-out seed test split)")
    st.dataframe(pd.DataFrame([{
        "Generator": name, "Mean KS": q["quality"]["mean_ks_statistic"],
        "Mean Wasserstein (0-1)": q["quality"]["mean_wasserstein_0_1"],
        "Correlation gap": q["quality"]["correlation_gap"], "C2ST AUC": q["quality"]["c2st_auc"],
        "Unique rows": q["diversity"]["unique_row_ratio"], "Exact train copies": q["diversity"]["exact_train_copy_rate"],
        "DCR median": q["diversity"]["dcr_median"], "Pairwise distance": q["diversity"]["mean_pairwise_distance"],
    } for name, q in metrics["quality"].items()]).round(4), hide_index=True, use_container_width=True)
    with st.expander("Per-feature comparison (hybrid vs seed test split)"):
        st.dataframe(pd.DataFrame(metrics["quality"]["Hybrid (GAN + filters)"]["quality"]["per_feature"]).round(4),
                     hide_index=True, use_container_width=True)

    aug_res = metrics["augmentation"]
    st.markdown("### 🎯 Downstream classifier — untouched seed test split")
    st.caption(f"{aug_res['classifier']} · train {aug_res['split_sizes']['train']} / val {aug_res['split_sizes']['val']} / "
               f"test {aug_res['split_sizes']['test']} validated seed profiles · mean ± std over seeds {aug_res['seeds']}.")
    rows = [{"Training data": "Seed only", "Strategy": "—",
             **{k: _mean_sd(aug_res["real_only"]["test"][k]) for k in ("accuracy", "precision_macro", "recall_macro",
                                                                       "f1_macro")},
             "Δ macro-F1": "—"}]
    for name, s in aug_res["samplers"].items():
        chosen = s["strategies"][s["chosen_strategy"]]
        d = s["test_f1_macro_delta_vs_real"]
        rows.append({"Training data": f"Seed + {name}", "Strategy": chosen["label"],
                     **{k: _mean_sd(chosen["test"][k]) for k in ("accuracy", "precision_macro", "recall_macro",
                                                                 "f1_macro")},
                     "Δ macro-F1": f"{d['mean']:+.4f} ({d['seeds_improved']}/{d['seeds']} seeds)"})
    st.dataframe(pd.DataFrame(rows).rename(columns=METRIC_LABELS), hide_index=True, use_container_width=True)

    detail = metrics["detailed_test_report"]
    l, r = st.columns(2)
    for col, key, title in ((l, "real_only", "Seed only"), (r, "augmented", "Seed + hybrid")):
        with col:
            st.plotly_chart(confusion_heatmap(detail[key]["confusion_matrix"], metrics["role_names"],
                                              f"{title} · accuracy {detail[key]['accuracy']:.3f}"),
                            use_container_width=True)

    real_eval = metrics.get("real_check")
    if real_eval:
        first = next(iter(real_eval.values()))
        st.markdown("### 🌍 Real check — Kaggle resumes labelled by their own job title")
        st.caption(f"{first['rows']} real resumes ({', '.join(f'{k}: {v}' for k, v in first['per_role'].items())}). "
                   f"Weak labels from the job title; never used for training or selection. Majority-class accuracy "
                   f"would be {first['majority_class_accuracy']:.3f}, so compare macro-F1.")
        st.dataframe(pd.DataFrame([{"Training data": name,
                                    **{METRIC_LABELS[k]: _mean_sd(v[k]) for k in ("accuracy", "precision_macro",
                                                                                  "recall_macro", "f1_macro")}}
                                   for name, v in real_eval.items()]), hide_index=True, use_container_width=True)


COMPARE_LABELS = {
    "rows": "Rows", "duplicate_rate": "Duplicate rate", "mean_pairwise_distance": "Diversity (pairwise distance)",
    "distinct_values_per_feature": "Distinct values / feature", "experience_zero_share": "Experience = 0 share",
    "experience_mean": "Mean experience", "experience_projects_corr": "Experience ↔ projects corr.",
    "many_certs_no_experience_share": "≥8 certs & <1 yr share", "exact_train_copy_rate": "Exact train copies",
    "validity_rate": "Validity (filter pass) rate", "similarity_to_real_median": "Distance to nearest real record",
}
METRIC_LABELS = {"accuracy": "Accuracy", "precision_macro": "Precision (macro)", "recall_macro": "Recall (macro)",
                 "f1_macro": "F1 (macro)"}


def page_gan():
    from models.gan.dataset import KAGGLE_URL, PROCESSED_PATH, load_processed
    from models.gan.gan import checkpoint_available, load_metrics

    ui.hero("GAN Candidate Generator — Synthetic Profiles for Dataset Augmentation",
            "Noise + Job Role → Generator → 9 Features ⇄ Critic (WGAN-GP) → Synthetic Candidate Profiles",
            eyebrow="Model 4 of 5 · generator.safetensors + config.json")
    st.warning("Every profile generated on this page is **synthetic** — produced by a GAN for dataset augmentation "
               "and testing. They are not real candidates and must not be used for hiring decisions.", icon="🧪")

    # ---------------- model status ----------------
    if not checkpoint_available():
        st.error("GAN checkpoint not found in `models/gan/`. Train it first (Kaggle dataset: "
                 f"{KAGGLE_URL}):", icon="❌")
        st.code(f"cd RecruitGen\n{GAN_TRAIN_CMD}", language="bash")
        return
    try:
        gan = load_gan()
    except Exception as exc:
        st.error(f"The GAN checkpoint could not be loaded: {exc}", icon="❌")
        return
    info = gan.training_info
    c = st.columns(4)
    with c[0]:
        ui.metric("Model Status", "Loaded", f"{info.get('seconds', 0):.0f}s training · best iter "
                                            f"{info.get('best_iteration', '—')}", ui.SUCCESS)
    with c[1]:
        ui.metric("Generator Parameters", f"{gan.num_params:,}", "Conditional WGAN-GP")
    with c[2]:
        ui.metric("Job Roles", len(gan.role_names), "Kaggle resume categories")
    with c[3]:
        ui.metric("Trained On", f"{info.get('train_rows', 0):,} resumes", "Real train split only")

    real = load_processed()
    gen_tab, aug_tab, eval_tab = st.tabs(["🧪 Generate Profiles", "📦 Dataset Augmentation",
                                          "📊 Evaluation & CVAE Comparison"])

    # ---------------- generation ----------------
    with gen_tab:
        c = st.columns(3)
        role = c[0].selectbox("Job role (condition)", gan.role_names,
                              index=gan.role_names.index("Information Technology")
                              if "Information Technology" in gan.role_names else 0)
        n = c[1].slider("Number of synthetic profiles", 1, 100, 10)
        seed = c[2].number_input("Random seed", 0, 99999, 42, key="gan_seed")
        if st.button("🧪 Generate Synthetic Profiles", type="primary", use_container_width=True):
            with st.spinner("Sampling noise and running the generator..."):
                st.session_state.gan_gen = (role, gan.generate(role, n, int(seed)), int(seed))
            st.session_state.pop("gan_ai", None)
            log("GAN generation", f"{n} synthetic profiles for {role} (seed {seed})")
        if st.session_state.get("gan_gen"):
            role, rows, gen_seed = st.session_state.gan_gen
            df = gan_table(rows)
            st.dataframe(df, hide_index=True, use_container_width=True)
            mean_profile = pd.DataFrame(rows)[[f[0] for f in FEATURES]].mean().to_dict()
            l, r = st.columns(2)
            with l:
                st.plotly_chart(radar_chart(mean_profile), use_container_width=True)
            with r:
                st.plotly_chart(bar_chart(feature_scores(mean_profile), "Average synthetic profile (0-100)", ""),
                                use_container_width=True)
            if real is not None and role in set(real["job_role"]):
                ref = real[real["job_role"] == role][[f[0] for f in FEATURES]].mean()
                comp = pd.DataFrame({"Feature": [FEATURE_LABEL[k] for k in ref.index],
                                     "Real mean (Kaggle)": ref.round(2).values,
                                     "Synthetic mean (this batch)": [round(mean_profile[k], 2) for k in ref.index]})
                with st.expander(f"Real vs synthetic feature means — {role}"):
                    st.dataframe(comp, hide_index=True, use_container_width=True)
            st.download_button("⬇️ Synthetic profiles (CSV)", df.to_csv(index=False).encode(),
                               f"synthetic_gan_{_slug(role)}_seed{gen_seed}.csv", "text/csv",
                               use_container_width=True, type="primary",
                               on_click=log, args=("GAN CSV exported", f"{len(df)} synthetic profiles · {role}"))

            st.markdown("#### ✨ Optional AI summary")
            st.caption("The GAN generates the numbers in Python; the AI only describes the selected synthetic profile.")
            a, b = st.columns([2, 1])
            pick = a.selectbox("Synthetic profile", list(df["Candidate"]), label_visibility="collapsed")
            if b.button("✨ Summarise profile", use_container_width=True):
                profile = rows[list(df["Candidate"]).index(pick)]
                if not ai_available():
                    st.session_state.gan_ai = (pick, None)
                else:
                    with st.spinner("Asking the AI to describe the synthetic profile..."):
                        st.session_state.gan_ai = (pick, synthetic_profile_summary(
                            {k: v for k, v in profile.items() if k != "job_role"}, role))
            if st.session_state.get("gan_ai"):
                who, summary = st.session_state.gan_ai
                if not summary:
                    st.info("AI summaries are unavailable right now (no API key, network or service error). "
                            "Generation, tables and CSV export are unaffected.", icon="⏳")
                else:
                    st.markdown(f"**{who} · {SYNTHETIC_LABEL}** — {summary.get('summary', '')}")
                    x, y = st.columns(2)
                    with x:
                        st.markdown("**Strengths**")
                        for s in summary.get("strengths", []) or []:
                            st.markdown(f"- {s}")
                    with y:
                        st.markdown("**Gaps**")
                        for s in summary.get("gaps", []) or []:
                            st.markdown(f"- {s}")
                    if summary.get("realism_note"):
                        st.caption(f"Realism: {summary['realism_note']}")

    # ---------------- augmentation ----------------
    with aug_tab:
        st.markdown("Build a training dataset that mixes real Kaggle profiles with GAN-generated ones. "
                    "Every row has a `data_source` and `is_synthetic` column.")
        roles = st.multiselect("Job roles to augment", gan.role_names, default=gan.role_names)
        mode = st.radio("Augmentation mode", ["➕ Fixed number per role", "⚖️ Balance every role up to a target"],
                        horizontal=True)
        include_real = False
        real_scope = "train"
        if real is None:
            st.info(f"Processed real data not found (`{os.path.relpath(PROCESSED_PATH, BASE_DIR)}`). Run "
                    f"`{GAN_TRAIN_CMD}` to create it — until then the export contains synthetic rows only.", icon="ℹ️")
        else:
            c = st.columns(2)
            include_real = c[0].checkbox("Include the real Kaggle rows", value=True)
            scope = c[1].radio("Real rows", ["Train split only (recommended — keeps the test split untouched)",
                                             "All real rows"], disabled=not include_real)
            real_scope = "train" if scope.startswith("Train") else "all"
        real_rows = None
        if real is not None:
            real_rows = real if real_scope == "all" else real[real["split"] == "train"]
        base_counts = (real_rows["job_role"].value_counts().to_dict() if real_rows is not None
                       else info.get("train_role_counts", {}))
        c = st.columns(2)
        if mode.startswith("➕"):
            per_role = c[0].number_input("Synthetic profiles per role", 1, 2000, 50)
            counts = {r: int(per_role) for r in roles}
        else:
            target = c[0].number_input("Target rows per role (real + synthetic)", 1, 5000,
                                       int(max(base_counts.values())) if base_counts else 100)
            counts = {r: max(0, int(target) - int(base_counts.get(r, 0))) for r in roles}
        aug_seed = c[1].number_input("Random seed", 0, 99999, 7, key="gan_aug_seed")
        st.caption(f"{sum(counts.values()):,} synthetic rows will be generated for {len(roles)} roles.")

        if st.button("📦 Build Augmented Dataset", type="primary", use_container_width=True, disabled=not roles):
            with st.spinner("Generating synthetic rows..."):
                syn = pd.DataFrame(gan.generate_many(counts, int(aug_seed)))
                syn.insert(0, "record_id", [f"GAN-SYN-{i:05d}" for i in range(1, len(syn) + 1)])
                syn["data_source"], syn["is_synthetic"], syn["split"] = SYNTHETIC_LABEL, True, "synthetic"
                parts = [syn]
                if include_real and real_rows is not None:
                    keep = real_rows[real_rows["job_role"].isin(roles)].copy()
                    keep.insert(0, "record_id", "KAGGLE-" + keep["resume_id"].astype(str))
                    keep["data_source"], keep["is_synthetic"] = REAL_LABEL, False
                    parts.insert(0, keep)
                cols = ["record_id", "data_source", "is_synthetic", "job_role"] + [f[0] for f in FEATURES] + ["split"]
                st.session_state.gan_aug = pd.concat(parts, ignore_index=True)[cols]
            log("GAN augmentation", f"{len(syn)} synthetic + {len(st.session_state.gan_aug) - len(syn)} real rows "
                                    f"(seed {aug_seed})")
        aug = st.session_state.get("gan_aug")
        if aug is not None:
            n_syn = int(aug["is_synthetic"].sum())
            m = st.columns(3)
            with m[0]:
                ui.metric("Total Rows", f"{len(aug):,}", "Augmented dataset")
            with m[1]:
                ui.metric("Real Rows", f"{len(aug) - n_syn:,}", REAL_LABEL)
            with m[2]:
                ui.metric("Synthetic Rows", f"{n_syn:,}", SYNTHETIC_LABEL, ui.WARNING)
            by_role = aug.groupby(["job_role", "data_source"]).size().unstack(fill_value=0)
            fig = go.Figure([go.Bar(name=src, y=by_role.index, x=by_role[src], orientation="h",
                                    marker_color=ui.PRIMARY if src == SYNTHETIC_LABEL else "#38BDF8")
                             for src in by_role.columns])
            fig.update_layout(barmode="stack", title="Rows per job role", legend=dict(orientation="h"))
            st.plotly_chart(chart_layout(fig, 80 + 22 * len(by_role)), use_container_width=True)
            st.dataframe(aug.head(200), hide_index=True, use_container_width=True)
            st.caption("Preview shows the first 200 rows.")
            st.download_button("⬇️ Augmented dataset (CSV)", aug.to_csv(index=False).encode(),
                               "augmented_dataset_gan.csv", "text/csv", use_container_width=True, type="primary",
                               on_click=log, args=("GAN CSV exported", f"Augmented dataset · {len(aug)} rows"))

    # ---------------- evaluation ----------------
    with eval_tab:
        metrics = load_metrics()
        if not metrics:
            st.info(f"No evaluation report yet. Run `{GAN_TRAIN_CMD}` (or GAN_Training.ipynb) to compute it.", icon="ℹ️")
            return
        if metrics.get("quick_run"):
            st.warning("This report comes from a QUICK test run (few iterations / seeds) — retrain without "
                       "`--quick` for reportable numbers.", icon="⚠️")
        st.caption(f"Computed {metrics['generated_at']} · {metrics['note']}")
        aug_res = metrics["augmentation"]
        sizes = aug_res["split_sizes"]
        st.markdown("### 🎯 Downstream job-role classifier — real only vs augmented")
        st.caption(f"{aug_res['classifier']} · {len(metrics['role_names'])} job roles · train {sizes['train']} / "
                   f"validation {sizes['val']} / test {sizes['test']} real resumes · mean ± std over seeds "
                   f"{aug_res['seeds']} · augmentation strategy chosen on the validation split, reported on the "
                   "same untouched real test split.")
        rows = [{"Training data": "Real only", "Strategy": "—", "Synthetic rows": 0,
                 **{k: _mean_sd(aug_res["real_only"]["test"][k])
                    for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro")},
                 "Δ macro-F1 vs real": "—", "Seeds improved": "—"}]
        for name, s in aug_res["samplers"].items():
            chosen = s["strategies"][s["chosen_strategy"]]
            delta = s["test_f1_macro_delta_vs_real"]
            rows.append({"Training data": f"Real + {name}", "Strategy": chosen["label"],
                         "Synthetic rows": chosen["synthetic_rows"],
                         **{k: _mean_sd(chosen["test"][k])
                            for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro")},
                         "Δ macro-F1 vs real": f"{delta['mean']:+.4f} ± {delta['std']:.4f}",
                         "Seeds improved": f"{delta['seeds_improved']}/{delta['seeds']}"})
        st.dataframe(pd.DataFrame(rows).rename(columns={"accuracy": "Accuracy", "precision_macro": "Precision (macro)",
                                                        "recall_macro": "Recall (macro)", "f1_macro": "F1 (macro)"}),
                     hide_index=True, use_container_width=True)
        with st.expander("All augmentation strategies (validation and test)"):
            st.dataframe(pd.DataFrame([
                {"Sampler": name, "Strategy": st_["label"], "Synthetic rows": st_["synthetic_rows"],
                 "Val macro-F1": _mean_sd(st_["val"]["f1_macro"]), "Test macro-F1": _mean_sd(st_["test"]["f1_macro"]),
                 "Test accuracy": _mean_sd(st_["test"]["accuracy"]),
                 "Chosen": "✅" if key == s["chosen_strategy"] else ""}
                for name, s in aug_res["samplers"].items() for key, st_ in s["strategies"].items()
            ]), hide_index=True, use_container_width=True)

        detail = metrics["detailed_test_report"]
        labels = metrics["role_names"]
        st.markdown(f"#### Classification reports & confusion matrices (test split, seed {detail['seed']})")
        l, r = st.columns(2)
        for col, key, title in ((l, "real_only", "Real only"), (r, "augmented", "Real + GAN")):
            rep = detail[key]["classification_report"]
            with col:
                st.markdown(f"**{title}** · accuracy {detail[key]['accuracy']:.4f} · macro-F1 {detail[key]['f1_macro']:.4f}")
                st.dataframe(pd.DataFrame(
                    [{"Job role": k, "Precision": v["precision"], "Recall": v["recall"], "F1": v["f1-score"],
                      "Support": int(v["support"])} for k, v in rep.items() if isinstance(v, dict)]
                ).round(4), hide_index=True, use_container_width=True, height=300)
                st.plotly_chart(confusion_heatmap(detail[key]["confusion_matrix"], labels, f"Confusion matrix — {title}"),
                                use_container_width=True)

        st.markdown("### 🔬 GAN quality & diversity (evaluated separately)")
        st.caption("Compared with the held-out real test split. 'Real reference' scores real training rows the same way, "
                   "i.e. what a perfect generator would achieve. Lower is better for KS / Wasserstein / gaps; "
                   "C2ST AUC near 0.5 means real and synthetic are hard to tell apart.")
        st.dataframe(pd.DataFrame([{
            "Generator": name,
            "Mean KS": q["quality"]["mean_ks_statistic"], "Mean Wasserstein (0-1)": q["quality"]["mean_wasserstein_0_1"],
            "Correlation gap": q["quality"]["correlation_gap"], "Per-role mean gap": q["quality"]["per_role_mean_gap_0_1"],
            "C2ST AUC": q["quality"]["c2st_auc"], "Unique rows": q["diversity"]["unique_row_ratio"],
            "Exact train copies": q["diversity"]["exact_train_copy_rate"], "DCR median": q["diversity"]["dcr_median"],
            "Pairwise distance": q["diversity"]["mean_pairwise_distance"],
        } for name, q in metrics["quality"].items()]).round(4), hide_index=True, use_container_width=True)
        tstr_rows = [{"Trained on": "Real train split", "Test accuracy": _mean_sd(aug_res["real_only"]["test"]["accuracy"]),
                      "Test macro-F1": _mean_sd(aug_res["real_only"]["test"]["f1_macro"])}]
        tstr_rows += [{"Trained on": f"{name} synthetic only", "Test accuracy": _mean_sd(v["accuracy"]),
                       "Test macro-F1": _mean_sd(v["f1_macro"])} for name, v in metrics["tstr"].items()]
        st.markdown("**Train on synthetic, test on real (TSTR)**")
        st.dataframe(pd.DataFrame(tstr_rows), hide_index=True, use_container_width=True)
        with st.expander("Per-feature comparison (GAN vs real test split)"):
            st.dataframe(pd.DataFrame(metrics["quality"]["GAN"]["quality"]["per_feature"]).round(4),
                         hide_index=True, use_container_width=True)
        hist = pd.DataFrame(metrics.get("training_history", []))
        if not hist.empty:
            with st.expander("Training curves"):
                fig = go.Figure([go.Scatter(x=hist["iteration"], y=hist["val_distance"], name="Validation distance"),
                                 go.Scatter(x=hist["iteration"], y=hist["wasserstein_estimate"], name="Critic W-estimate",
                                            yaxis="y2")])
                fig.update_layout(yaxis=dict(title="validation distance"),
                                  yaxis2=dict(title="W-estimate", overlaying="y", side="right"),
                                  legend=dict(orientation="h"), xaxis=dict(title="iteration"))
                st.plotly_chart(chart_layout(fig, 360), use_container_width=True)

        st.markdown("### 🧬 GAN vs the existing CVAE")
        st.dataframe(pd.DataFrame([
            {"": "Purpose in RecruitGen", "CVAE / VAE page": "Role-fit scoring + a few synthetic candidates",
             "GAN page": "Bulk synthetic profiles for dataset augmentation / testing"},
            {"": "Training data", "CVAE / VAE page": "Simulated candidates (CVAE_Training.ipynb)",
             "GAN page": "Real Kaggle resumes → 9 extracted features"},
            {"": "Condition", "CVAE / VAE page": "5 RecruitGen roles", "GAN page": f"{len(labels)} Kaggle job categories"},
            {"": "Generation", "CVAE / VAE page": "Decoder from latent z ~ N(0, I)",
             "GAN page": "Generator trained adversarially against a critic (WGAN-GP)"},
        ]), hide_index=True, use_container_width=True)
        st.caption("The 'CVAE baseline' in the tables above uses the same layer sizes as the shipped CVAE but is retrained "
                   "on the same Kaggle split, so the two generative approaches are compared on identical data. The "
                   "shipped CVAE weights are not modified.")

        st.markdown("#### ✨ Optional AI explanation")
        if st.button("✨ Explain these results", use_container_width=True):
            text = explain_gan_results(gan_results_summary(metrics)) if ai_available() else None
            st.session_state.gan_explain = text or ""
        if "gan_explain" in st.session_state:
            if st.session_state.gan_explain:
                st.markdown(st.session_state.gan_explain)
                st.caption("AI-written explanation of the Python-computed numbers above — the numbers in the tables are "
                           "authoritative.")
            else:
                st.info("AI explanations are unavailable right now. All metrics above are unaffected.", icon="⏳")


# ============================================================
# PAGE: TRANSFORMER V3
# ============================================================

def page_transformer():
    ui.hero("Transformer V3 — Intelligent Resume Screening",
            "Feature Extraction → DistilBERT Embedding → Feature Fusion → Role Prediction",
            eyebrow="Model 3 of 5 · model.safetensors")
    ui.pipeline(["1 · Resume Feature Extraction", "2 · Transformer Encoding", "3 · Feature Fusion",
                 "4 · Role Prediction"])

    src = st.radio("Resume input", ["📝 Paste text", "📤 Upload file"], horizontal=True)
    text = ""
    if src.startswith("📝"):
        default = st.session_state.get("analysis", {}).get("text", "")
        text = st.text_area("Textual resume", value=default, height=220,
                            placeholder="Machine Learning Engineer with 1.5 years of experience. Python, TensorFlow, "
                                        "PyTorch, SQL, AWS...")
    else:
        up = st.file_uploader("Upload resume", type=SUPPORTED_TYPES, key="tr_upload")
        if up:
            with st.spinner("Reading resume..."):
                text, _, _ = get_resume_text_from_upload(up, use_autoencoder=True)
            st.text_area("Extracted text", text, height=180)

    if st.button("🔍 Extract Features & Predict Role", type="primary", use_container_width=True):
        if len(text.strip()) < 20:
            st.warning("Please provide a more complete resume.")
        else:
            from models.transformer_v3.feature_extractor import extract_features
            with st.spinner("Extracting features and running Transformer V3..."):
                feats = extract_features(text)
                st.session_state.tr_result = {"text": text, "features": feats,
                                              "pred": load_transformer().predict(text, feats)}
                st.session_state.last_features = feats
            p_role, p_conf, _ = st.session_state.tr_result["pred"]
            log("Transformer prediction", f"{p_role} ({p_conf:.1f}%)")

    res = st.session_state.get("tr_result")
    if not res:
        return
    st.markdown("### 📐 Extracted 9 Features")
    feats = dict(res["features"])
    with st.expander("✏️ Adjust features and re-predict"):
        cols = st.columns(3)
        for i, (key, label, meaning, lo, hi, step) in enumerate(FEATURES):
            with cols[i % 3]:
                if isinstance(step, int):
                    feats[key] = st.number_input(label, int(lo), int(hi), int(feats[key]), int(step), key=f"tr_{key}")
                else:
                    feats[key] = st.number_input(label, float(lo), float(hi), float(feats[key]), float(step),
                                                 key=f"tr_{key}")
        if st.button("🔁 Re-predict with adjusted features"):
            res["features"] = feats
            res["pred"] = load_transformer().predict(res["text"], feats)
            st.rerun()

    l, r = st.columns([1.2, 1])
    with l:
        features_table(res["features"])
    with r:
        st.plotly_chart(radar_chart(res["features"]), use_container_width=True)

    role, conf, probs = res["pred"]
    st.markdown("### 🎯 Prediction")
    l, r = st.columns([1, 1.4])
    with l:
        ui.metric("Predicted Role", role, "Highest semantic + feature match", ui.PRIMARY)
        st.markdown("<br>", unsafe_allow_html=True)
        ui.metric("Confidence", f"{conf:.2f}%", "Softmax probability", ui.score_color(conf))
    with r:
        st.plotly_chart(bar_chart(probs, "Probability per job role"), use_container_width=True)
    pdf_button("Download prediction report (PDF)", "transformer", signature(res["features"], res["pred"]),
               lambda: pdf.build_transformer_pdf(res, feature_rows(res["features"]), feature_scores(res["features"]),
                                                 prepared_by()),
               pdf.pdf_filename("role prediction", role), primary=True)


# ============================================================
# PAGE: AI ASSISTANT
# ============================================================

def page_assistant():
    ui.hero("AI Recruitment Assistant", "Resume insights, job-description matching, interview questions and chat.",
            eyebrow="Insights")

    analysis = st.session_state.get("analysis")
    resume_text = analysis["text"] if analysis else ""
    resume_source = analysis["file_name"] if analysis else ""
    with st.expander("📄 Resume in context", expanded=not resume_text):
        if resume_text:
            st.success(f"Using the last analysed resume: **{analysis['file_name']}**")
        up = st.file_uploader("Upload a different resume (optional)", type=SUPPORTED_TYPES, key="assist_upload")
        if up:
            with st.spinner("Reading resume..."):
                resume_text, _, _ = get_resume_text_from_upload(up)
            resume_source = up.name
        typed = st.text_area("…or paste resume text", height=140, key="assist_text")
        if typed.strip():
            resume_text, resume_source = typed.strip(), "Pasted resume text"

    t1, t2, t3, t4 = st.tabs(["💡 Resume Insights", "🎯 Job Match", "❓ Interview Questions", "💬 Chat"])

    with t1:
        if st.button("Generate insights", type="primary", disabled=not resume_text):
            with st.spinner("Analysing resume..."):
                st.session_state.assist_insights = resume_insights(resume_text)
                st.session_state.assist_source = resume_source
            log("AI insights", "Generated resume insights")
        if "assist_insights" in st.session_state:
            ins = st.session_state.assist_insights
            render_insights(ins)
            if ins:
                src = st.session_state.get("assist_source", "")
                pdf_button("Download insights report (PDF)", "assist_insights", signature(ins, src),
                           lambda: pdf.build_insights_pdf(ins, prepared_by(), src),
                           pdf.pdf_filename("resume insights", ins.get("candidate_name", "")), primary=True)
        elif not resume_text:
            st.info("Add a resume above to generate insights.")

    with t2:
        jd = st.text_area("Job description", height=180, placeholder="Paste the job description here...")
        if st.button("Match resume to job", type="primary", disabled=not (resume_text and jd.strip())):
            with st.spinner("Comparing resume with the job description..."):
                st.session_state.jd_result = job_match(resume_text, jd)
                st.session_state.jd_text = jd
            jr = st.session_state.jd_result
            log("AI job match", f"Match score {jr.get('match_score', '—')}%" if jr else "AI unavailable")
        res = st.session_state.get("jd_result", "unset")
        if res is None:
            ai_unavailable_notice()
        elif res != "unset":
            s = int(res.get("match_score", 0) or 0)
            c = st.columns([1, 2])
            with c[0]:
                ui.metric("Match Score", f"{s}%", res.get("verdict", ""), ui.score_color(s))
            with c[1]:
                st.write(res.get("summary", ""))
            a, b = st.columns(2)
            with a:
                st.markdown("**✅ Matched skills**")
                ui.chips(res.get("matched_skills", []), "good")
            with b:
                st.markdown("**❌ Missing skills**")
                ui.chips(res.get("missing_skills", []), "bad")
            st.markdown("**🚀 Recommendations**")
            for item in res.get("recommendations", []):
                st.markdown(f"- {item}")
            jd_text = st.session_state.get("jd_text", "")
            pdf_button("Download job-match report (PDF)", "jd_match", signature(res, jd_text),
                       lambda: pdf.build_job_match_pdf(res, prepared_by(), jd_text),
                       pdf.pdf_filename("job match", res.get("verdict", "")), primary=True)

    with t3:
        c = st.columns(3)
        default_role = analysis["prediction"]["role"] if analysis else "Software Engineer"
        role = c[0].text_input("Role", default_role)
        level = c[1].selectbox("Level", ["Fresher", "Junior", "Mid", "Senior", "Lead"], index=2)
        count = c[2].slider("Questions", 4, 15, 8)
        if st.button("Generate questions", type="primary", disabled=not resume_text):
            with st.spinner("Preparing interview questions..."):
                st.session_state.iq = interview_questions(resume_text, role, level, count)
                st.session_state.iq_meta = (role, level)
            log("AI interview questions", f"{count} questions · {role} · {level}")
        qs = st.session_state.get("iq", "unset")
        if qs is None:
            ai_unavailable_notice()
        elif qs != "unset":
            icons = {"Technical": "🛠️", "Project": "📁", "Behavioural": "🤝"}
            for i, q in enumerate(qs, 1):
                with st.expander(f"{icons.get(q.get('type'), '❓')} Q{i}. {q.get('question', '')}"):
                    st.markdown(f"**Type:** {q.get('type', '-')}")
                    st.markdown(f"**What to look for:** {q.get('what_to_look_for', '-')}")
            if qs:
                q_role, q_level = st.session_state.get("iq_meta", (role, level))
                candidate = ((analysis or {}).get("insights") or {}).get("candidate_name", "")
                pdf_button("Download interview kit (PDF)", "iq", signature(qs, q_role, q_level, candidate),
                           lambda: pdf.build_interview_pdf(qs, q_role, q_level, prepared_by(), candidate),
                           pdf.pdf_filename("interview kit", f"{q_role} {q_level}"), primary=True)

    with t4:
        chat = st.session_state.setdefault("chat", [])
        box = st.container(height=420)
        with box:
            if not chat:
                st.caption("Ask anything about this resume, hiring, or career growth.")
            for m in chat:
                with st.chat_message(m["role"], avatar="🧑‍💼" if m["role"] == "user" else "🤖"):
                    st.markdown(format_ai_markdown(m["content"]) if m["role"] == "assistant" else m["content"])
        prompt = st.chat_input("Ask RecruitGen...")
        if prompt:
            chat.append({"role": "user", "content": prompt})
            log("AI chat", prompt[:160])
            with box:
                with st.chat_message("user", avatar="🧑‍💼"):
                    st.markdown(prompt)
                with st.chat_message("assistant", avatar="🤖"):
                    with st.spinner("Thinking..."):
                        reply = assistant_reply(chat, resume_text)
                    reply = reply or "I'm temporarily unavailable. Please try again in a moment."
                    st.markdown(reply)
            chat.append({"role": "assistant", "content": reply})
        if chat:
            c1, c2 = st.columns(2)
            with c1:
                transcript = [{"role": m["role"], "content": format_ai_markdown(m["content"])
                               if m["role"] == "assistant" else m["content"]} for m in chat]
                pdf_button("Download conversation (PDF)", "chat", signature(transcript, resume_source),
                           lambda: pdf.build_chat_pdf(transcript, prepared_by(), resume_source),
                           pdf.pdf_filename("ai chat"))
            if c2.button("🗑️ Clear chat", use_container_width=True):
                st.session_state.chat = []
                st.rerun()


# ============================================================
# PAGE: HISTORY (ADMIN ONLY)
# ============================================================

def admin_guard():
    if is_admin():
        return True
    st.error("🔒 This page is available to administrators only.")
    log("Access denied", "Tried to open an admin-only page")
    return False


def page_history():
    if not admin_guard():
        return
    ui.hero("Screening History", "Every resume screened by every recruiter, with predictions and role-fit results.",
            eyebrow="🛡️ Admin only · all recruiters", kind="admin")
    rows = auth.get_all_screenings()
    if not rows:
        st.info("No resumes screened yet. Screenings from all users will appear here.", icon="📭")
        return
    df = pd.DataFrame(rows)
    df["recruiter"] = df["recruiter"].fillna("Deleted user") + " (@" + df["username"].fillna("—") + ")"
    df = df.rename(columns={"created_at": "Date", "recruiter": "Screened By", "candidate": "Candidate",
                            "file_name": "File", "predicted_role": "Predicted Role", "confidence": "Confidence (%)",
                            "top_fit_role": "Best Fit Role", "fit_score": "Fit Score (%)", "ai_score": "Resume Score"})
    df = df[["Date", "Screened By", "Candidate", "File", "Predicted Role", "Confidence (%)", "Best Fit Role",
             "Fit Score (%)", "Resume Score"]]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Confidence (%)"] = df["Confidence (%)"].round(2)
    df["Fit Score (%)"] = df["Fit Score (%)"].round(2)

    c = st.columns(4)
    with c[0]:
        ui.metric("Total Screened", len(df), "All recruiters")
    with c[1]:
        ui.metric("Recruiters", df["Screened By"].nunique(), "Users who screened resumes")
    with c[2]:
        ui.metric("Avg. Confidence", f"{df['Confidence (%)'].mean():.1f}%", "Transformer V3")
    with c[3]:
        ui.metric("Avg. Fit Score", f"{df['Fit Score (%)'].mean():.1f}%", "CVAE role fit")

    st.markdown("<br>", unsafe_allow_html=True)
    f1, f2 = st.columns([1, 2])
    who = f1.multiselect("Recruiter", sorted(df["Screened By"].unique()))
    search = f2.text_input("🔎 Search candidates, files or roles")
    view = df[df["Screened By"].isin(who)] if who else df
    if search:
        view = view[view.apply(lambda r: search.lower() in " ".join(map(str, r.values)).lower(), axis=1)]
    st.dataframe(view, hide_index=True, use_container_width=True, column_config={
        "Date": st.column_config.DatetimeColumn(format="DD MMM YYYY, hh:mm a"),
        "Confidence (%)": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
        "Fit Score (%)": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
    })

    if not view.empty:
        l, r = st.columns(2)
        with l:
            counts = view["Predicted Role"].value_counts()
            fig = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=0.55,
                                   marker=dict(colors=[ui.PRIMARY, ui.ACCENT, "#A78BFA", "#34D399", "#FBBF24"])))
            fig.update_layout(title="Predicted roles")
            st.plotly_chart(chart_layout(fig, 320), use_container_width=True)
        with r:
            per = view["Screened By"].value_counts()
            st.plotly_chart(bar_chart(per.to_dict(), "Screenings per recruiter", ""), use_container_width=True)

    a, p, b = st.columns(3)
    a.download_button("⬇️ Export CSV", view.to_csv(index=False).encode(), "recruitgen_history.csv", "text/csv",
                      use_container_width=True, on_click=log, args=("History exported", f"{len(view)} rows"))
    with p:
        filters = "; ".join(x for x in [f"Recruiter: {', '.join(who)}" if who else "",
                                        f"Search: {search}" if search else ""] if x)
        pdf_button("History report (PDF)", "history", signature(view, filters),
                   lambda: pdf.build_history_pdf(view, prepared_by(), filters),
                   pdf.pdf_filename("screening history"), lazy=True)
    with b.popover("🗑️ Clear all history", use_container_width=True):
        st.warning("This permanently deletes the screening history of **all** users.")
        if st.button("Yes, delete everything", type="primary", use_container_width=True):
            auth.clear_all_screenings()
            log("History cleared", f"{len(df)} screenings deleted")
            st.rerun()


# ============================================================
# PAGE: ADMIN PANEL
# ============================================================

ONLINE_WINDOW = timedelta(minutes=10)
TIME_FMT = "DD MMM YYYY, hh:mm:ss a"
EVENT_STYLE = {
    "Login": ("🔓", "Authentication", ui.SUCCESS),
    "Logout": ("🔒", "Authentication", "#94A3B8"),
    "Login failed": ("⛔", "Authentication", ui.DANGER),
    "Sign up": ("🆕", "Authentication", ui.ACCENT),
    "Session resumed": ("🔁", "Authentication", ui.ACCENT),
    "Page view": ("🧭", "Navigation", "#64748B"),
    "Resume analyzed": ("📄", "AI Models", ui.PRIMARY),
    "Autoencoder run": ("🧹", "AI Models", ui.PRIMARY),
    "CVAE role fit": ("🧬", "AI Models", ui.PRIMARY),
    "CVAE generation": ("🧪", "AI Models", ui.PRIMARY),
    "Transformer prediction": ("🤖", "AI Models", ui.PRIMARY),
    "GAN generation": ("🧪", "AI Models", ui.PRIMARY),
    "GAN augmentation": ("📦", "AI Models", ui.PRIMARY),
    "GAN CSV exported": ("⬇️", "AI Models", ui.PRIMARY),
    "Diffusion enhancement": ("🌫️", "AI Models", ui.PRIMARY),
    "Bulk batch started": ("📦", "AI Models", ui.PRIMARY),
    "Bulk batch retried": ("🔁", "AI Models", ui.PRIMARY),
    "Bulk batch stopped": ("⏹", "AI Models", ui.PRIMARY),
    "Bulk results exported": ("⬇️", "AI Models", ui.PRIMARY),
    "AI insights": ("💡", "AI Assistant", ui.WARNING),
    "AI job match": ("🎯", "AI Assistant", ui.WARNING),
    "AI interview questions": ("❓", "AI Assistant", ui.WARNING),
    "AI chat": ("💬", "AI Assistant", ui.WARNING),
    "History exported": ("⬇️", "Admin", "#F59E0B"),
    "History cleared": ("🗑️", "Admin", "#F59E0B"),
    "Activity log cleared": ("🧾", "Admin", "#F59E0B"),
    "Admin export": ("⬇️", "Admin", "#F59E0B"),
    "PDF exported": ("📄", "Exports", ui.ACCENT),
    "Access denied": ("🚫", "Security", ui.DANGER),
}
ACTIVITY_COLS = ["id", "user_id", "username", "full_name", "event", "details", "session_id", "client", "created_at"]
SESSION_COLS = ["id", "user_id", "username", "full_name", "client", "login_at", "last_seen", "logout_at"]


def event_style(event):
    return EVENT_STYLE.get(event, ("•", "Other", "#64748B"))


def fmt_duration(seconds):
    if seconds is None or pd.isna(seconds):
        return "—"
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else (f"{m}m {s:02d}s" if m else f"{s}s")


def time_ago(ts, now):
    if ts is None or pd.isna(ts):
        return "—"
    secs = (now - ts).total_seconds()
    if secs < 60:
        return "just now"
    if secs < 3600:
        return f"{int(secs // 60)} min ago"
    if secs < 86400:
        return f"{int(secs // 3600)} h ago"
    return ts.strftime("%d %b %Y, %I:%M %p")


def session_status(row, now):
    if pd.notna(row["logout_at"]):
        return "🔴 Logged out"
    if now - row["last_seen"] <= ONLINE_WINDOW:
        return "🟢 Online"
    return "🟠 Idle / tab closed"


def load_admin_data():
    now = datetime.now()
    acts = pd.DataFrame(auth.get_activity(), columns=ACTIVITY_COLS)
    acts["created_at"] = pd.to_datetime(acts["created_at"], errors="coerce")
    acts["Icon"] = acts["event"].map(lambda e: event_style(e)[0])
    acts["Category"] = acts["event"].map(lambda e: event_style(e)[1])

    sess = pd.DataFrame(auth.get_sessions(), columns=SESSION_COLS)
    for col in ["login_at", "last_seen", "logout_at"]:
        sess[col] = pd.to_datetime(sess[col], errors="coerce")
    if len(sess):
        sess["Status"] = sess.apply(lambda r: session_status(r, now), axis=1)
        sess["seconds"] = (sess["logout_at"].fillna(sess["last_seen"]) - sess["login_at"]).dt.total_seconds()
    else:
        sess["Status"], sess["seconds"] = pd.Series(dtype=str), pd.Series(dtype=float)

    users = pd.DataFrame(auth.get_users())
    online_ids = set(sess.loc[sess["Status"] == "🟢 Online", "user_id"])
    users["Status"] = users["id"].map(lambda i: "🟢 Online" if i in online_ids else "⚪ Offline")
    return now, acts, sess, users


def activity_view(acts):
    return pd.DataFrame({
        "Time": acts["created_at"],
        "User": "@" + acts["username"].fillna("—"),
        "Name": acts["full_name"].fillna("—"),
        "Event": acts["Icon"] + " " + acts["event"],
        "Category": acts["Category"],
        "Details": acts["details"].fillna(""),
        "Device / IP": acts["client"].fillna(""),
        "Session": acts["session_id"].fillna("").str[:8],
    })


def sessions_view(sess):
    return pd.DataFrame({
        "User": "@" + sess["username"].fillna("—"),
        "Name": sess["full_name"].fillna("—"),
        "Status": sess["Status"],
        "Logged in": sess["login_at"],
        "Logged out": sess["logout_at"],
        "Last activity": sess["last_seen"],
        "Duration": sess["seconds"].map(fmt_duration),
        "Device / IP": sess["client"].fillna(""),
        "Session": sess["id"].str[:8],
    })


def users_view(users):
    return pd.DataFrame({
        "Name": users["full_name"], "Username": "@" + users["username"], "Email": users["email"],
        "Role": users["role"].map({"admin": "🛡️ Admin"}).fillna("👤 Recruiter"), "Status": users["Status"],
        "Joined": pd.to_datetime(users["created_at"], errors="coerce"),
        "Last login": pd.to_datetime(users["last_login"], errors="coerce"),
        "Logins": users["logins"], "Screenings": users["screenings"], "Events": users["events"],
    })


def render_timeline(acts, now, limit=15):
    if acts.empty:
        st.caption("No activity recorded yet.")
        return
    items = []
    for _, r in acts.head(limit).iterrows():
        icon, _, color = event_style(r["event"])
        who = r["full_name"] if pd.notna(r["full_name"]) else f"@{r['username']}"
        items.append((f"{icon} {r['event']} · {who}", r["details"] or "", time_ago(r["created_at"], now), color))
    ui.timeline(items)


def page_admin():
    if not admin_guard():
        return
    ui.hero("Admin Control Center",
            "Every login, logout, session and user action across RecruitGen — monitored in one place.",
            eyebrow="🛡️ Administrator · restricted area", kind="admin")
    top = st.columns([5, 1])
    if top[1].button("🔄 Refresh", use_container_width=True):
        st.rerun()

    now, acts, sess, users = load_admin_data()
    today = now.date()
    today_acts = acts[acts["created_at"].dt.date == today]

    m = st.columns(6)
    with m[0]:
        ui.metric("Users", len(users), f"{int((users['role'] == 'admin').sum())} admin")
    with m[1]:
        ui.metric("Online now", int((users["Status"] == "🟢 Online").sum()), "Active in last 10 min", ui.SUCCESS)
    with m[2]:
        ui.metric("Logins today", int((today_acts["event"] == "Login").sum()), f"{len(sess)} sessions all time")
    with m[3]:
        failed = int((today_acts["event"] == "Login failed").sum())
        ui.metric("Failed logins", failed, "Today", ui.DANGER if failed else None)
    with m[4]:
        ui.metric("Screenings", int(users["screenings"].sum()), "All recruiters")
    with m[5]:
        ui.metric("Events today", len(today_acts), f"{len(acts)} total")

    st.markdown("<br>", unsafe_allow_html=True)
    t_over, t_act, t_sess, t_users, t_stats, t_maint = st.tabs(
        ["🛰️ Live Overview", "📜 Activity Log", "🔐 Login / Logout", "👥 Users", "📈 Analytics", "⚙️ Maintenance"])

    with t_over:
        l, r = st.columns([1.4, 1])
        with l:
            st.markdown("#### 🕒 Latest activity")
            render_timeline(acts, now, 15)
        with r:
            st.markdown("#### 🟢 Who's online")
            online = sess[sess["Status"] == "🟢 Online"]
            if online.empty:
                st.caption("Nobody is active right now.")
            else:
                ui.timeline([(f"{s['full_name']} (@{s['username']})",
                              f"Since {s['login_at']:%I:%M %p} · {s['client'] or 'unknown device'}",
                              time_ago(s["last_seen"], now), ui.SUCCESS) for _, s in online.iterrows()])
            st.markdown("#### 🔐 Recent logins & logouts")
            render_timeline(acts[acts["event"].isin(["Login", "Logout", "Login failed", "Sign up"])], now, 8)

    with t_act:
        f = st.columns([1.2, 1.2, 1.2, 1])
        sel_users = f[0].multiselect("User", sorted(acts["username"].dropna().unique()))
        sel_cats = f[1].multiselect("Category", sorted(acts["Category"].unique()))
        sel_events = f[2].multiselect("Event", sorted(acts["event"].unique()))
        period = f[3].selectbox("Period", ["All time", "Today", "Last 7 days", "Last 30 days"])
        search = st.text_input("🔎 Search details, devices, names…", key="admin_act_search")
        view = acts
        if sel_users:
            view = view[view["username"].isin(sel_users)]
        if sel_cats:
            view = view[view["Category"].isin(sel_cats)]
        if sel_events:
            view = view[view["event"].isin(sel_events)]
        if period != "All time":
            start = {"Today": datetime.combine(today, datetime.min.time()),
                     "Last 7 days": now - timedelta(days=7), "Last 30 days": now - timedelta(days=30)}[period]
            view = view[view["created_at"] >= start]
        table = activity_view(view)
        if search:
            table = table[table.apply(lambda row: search.lower() in " ".join(map(str, row.values)).lower(), axis=1)]
        st.caption(f"Showing **{len(table)}** of {len(acts)} events")
        st.dataframe(table, hide_index=True, use_container_width=True, height=480,
                     column_config={"Time": st.column_config.DatetimeColumn(format=TIME_FMT),
                                    "Details": st.column_config.TextColumn(width="large")})
        e1, e2 = st.columns(2)
        e1.download_button("⬇️ Export filtered activity (CSV)", table.to_csv(index=False).encode(),
                           "recruitgen_activity.csv", "text/csv", use_container_width=True,
                           on_click=log, args=("Admin export", f"Activity log · {len(table)} rows"))
        with e2:
            filters = "; ".join(x for x in [
                f"Users: {', '.join(sel_users)}" if sel_users else "", f"Categories: {', '.join(sel_cats)}" if sel_cats
                else "", f"Events: {', '.join(sel_events)}" if sel_events else "",
                f"Period: {period}" if period != "All time" else "", f"Search: {search}" if search else ""] if x)
            pdf_button("Activity log (PDF)", "admin_activity", signature(table, filters),
                       lambda: pdf.build_activity_pdf(table, prepared_by(), filters),
                       pdf.pdf_filename("activity log"), lazy=True)

    with t_sess:
        c = st.columns(4)
        with c[0]:
            ui.metric("Total sessions", len(sess))
        with c[1]:
            ui.metric("Online", int((sess["Status"] == "🟢 Online").sum()), "", ui.SUCCESS)
        with c[2]:
            ui.metric("Logged out", int(sess["logout_at"].notna().sum()))
        with c[3]:
            ui.metric("Avg. session", fmt_duration(sess["seconds"].mean()) if len(sess) else "—")
        st.markdown("<br>", unsafe_allow_html=True)
        f = st.columns(2)
        status_filter = f[0].multiselect("Status", ["🟢 Online", "🟠 Idle / tab closed", "🔴 Logged out"])
        user_filter = f[1].multiselect("User", sorted(sess["username"].dropna().unique()), key="sess_users")
        view = sess
        if status_filter:
            view = view[view["Status"].isin(status_filter)]
        if user_filter:
            view = view[view["username"].isin(user_filter)]
        table = sessions_view(view)
        st.dataframe(table, hide_index=True, use_container_width=True, height=440, column_config={
            "Logged in": st.column_config.DatetimeColumn(format=TIME_FMT),
            "Logged out": st.column_config.DatetimeColumn(format=TIME_FMT),
            "Last activity": st.column_config.DatetimeColumn(format=TIME_FMT),
        })
        st.caption("🟠 *Idle / tab closed* = no activity for 10+ minutes without pressing Logout.")
        e1, e2 = st.columns(2)
        e1.download_button("⬇️ Export sessions (CSV)", table.to_csv(index=False).encode(), "recruitgen_sessions.csv",
                           "text/csv", use_container_width=True,
                           on_click=log, args=("Admin export", f"Sessions · {len(table)} rows"))
        with e2:
            filters = "; ".join(x for x in [f"Status: {', '.join(status_filter)}" if status_filter else "",
                                            f"Users: {', '.join(user_filter)}" if user_filter else ""] if x)
            pdf_button("Sessions report (PDF)", "admin_sessions", signature(table, filters),
                       lambda: pdf.build_sessions_pdf(table, prepared_by(), filters),
                       pdf.pdf_filename("login sessions"), lazy=True)

    with t_users:
        table = users_view(users)
        st.dataframe(table, hide_index=True, use_container_width=True, column_config={
            "Joined": st.column_config.DatetimeColumn(format="DD MMM YYYY"),
            "Last login": st.column_config.DatetimeColumn(format=TIME_FMT),
        })
        e1, e2 = st.columns(2)
        e1.download_button("⬇️ Export users (CSV)", table.to_csv(index=False).encode(), "recruitgen_users.csv",
                           "text/csv", use_container_width=True, on_click=log, args=("Admin export", "User list"))
        with e2:
            pdf_button("Users report (PDF)", "admin_users", signature(table),
                       lambda: pdf.build_users_pdf(table, prepared_by()), pdf.pdf_filename("users"), lazy=True)
        st.markdown("#### 🔍 Inspect a user")
        labels = {f"{u['full_name']} (@{u['username']})": u["id"] for _, u in users.iterrows()}
        uid = labels[st.selectbox("User", list(labels), label_visibility="collapsed")]
        u = users[users["id"] == uid].iloc[0]
        u_acts = acts[acts["user_id"] == uid]
        u_sess = sess[sess["user_id"] == uid]
        last_login = pd.to_datetime(u["last_login"]) if u["last_login"] else None
        c = st.columns(4)
        with c[0]:
            ui.metric("Status", u["Status"], "Admin" if u["role"] == "admin" else "Recruiter")
        with c[1]:
            ui.metric("Logins", int(u["logins"]), f"Last: {last_login:%d %b, %I:%M %p}" if last_login else "Never")
        with c[2]:
            ui.metric("Time in app", fmt_duration(u_sess["seconds"].sum()) if len(u_sess) else "—", "All sessions")
        with c[3]:
            ui.metric("Screenings", int(u["screenings"]), f"{int(u['events'])} events logged")
        l, r = st.columns([1.3, 1])
        with l:
            st.markdown("**Recent activity**")
            render_timeline(u_acts, now, 20)
        with r:
            st.markdown("**Sessions**")
            if u_sess.empty:
                st.caption("No sessions yet.")
            else:
                st.dataframe(sessions_view(u_sess)[["Status", "Logged in", "Logged out", "Duration"]],
                             hide_index=True, use_container_width=True, column_config={
                                 "Logged in": st.column_config.DatetimeColumn(format="DD MMM, hh:mm a"),
                                 "Logged out": st.column_config.DatetimeColumn(format="DD MMM, hh:mm a")})

    with t_stats:
        if acts.empty:
            st.info("Analytics will appear once there is some activity.", icon="📈")
        else:
            recent = acts[acts["created_at"] >= now - timedelta(days=14)]
            daily = recent.groupby([recent["created_at"].dt.date, "Category"]).size().unstack(fill_value=0)
            palette = {"Authentication": ui.SUCCESS, "Navigation": "#64748B", "AI Models": ui.PRIMARY,
                       "AI Assistant": ui.WARNING, "Admin": "#F59E0B", "Security": ui.DANGER, "Exports": ui.ACCENT}
            fig = go.Figure()
            for cat in daily.columns:
                fig.add_bar(name=cat, x=daily.index, y=daily[cat], marker_color=palette.get(cat, "#94A3B8"))
            fig.update_layout(barmode="stack", title="Activity per day (last 14 days)",
                              legend=dict(orientation="h", y=-0.2), yaxis=dict(gridcolor="#1E293B"))
            st.plotly_chart(chart_layout(fig, 360), use_container_width=True)
            l, r = st.columns(2)
            with l:
                logins = acts[acts["event"] == "Login"]["username"].value_counts()
                if len(logins):
                    st.plotly_chart(bar_chart(logins.to_dict(), "Logins per user", ""), use_container_width=True)
            with r:
                usage = acts[~acts["Category"].isin(["Navigation", "Authentication"])]["event"].value_counts()
                if len(usage):
                    fig = go.Figure(go.Pie(labels=usage.index, values=usage.values, hole=0.55))
                    fig.update_layout(title="Feature usage")
                    st.plotly_chart(chart_layout(fig, 340), use_container_width=True)
            hours = acts["created_at"].dt.hour.value_counts().reindex(range(24), fill_value=0)
            fig = go.Figure(go.Bar(x=[f"{h:02d}:00" for h in hours.index], y=hours.values, marker_color=ui.ACCENT))
            fig.update_layout(title="Activity by hour of day", yaxis=dict(gridcolor="#1E293B"))
            st.plotly_chart(chart_layout(fig, 300), use_container_width=True)

    with t_maint:
        st.markdown("#### 📄 Full admin report")
        st.caption("Users, the latest 150 sessions and the latest 400 events in one branded PDF.")
        full_users, full_sess, full_acts = users_view(users), sessions_view(sess), activity_view(acts)
        pdf_button("Full admin report (PDF)", "admin_full", signature(full_users, full_sess, full_acts),
                   lambda: pdf.build_admin_report_pdf(full_users, full_sess, full_acts, prepared_by()),
                   pdf.pdf_filename("admin report"), lazy=True)
        st.markdown("#### ⬇️ Full exports")
        c = st.columns(3)
        c[0].download_button("Activity log", activity_view(acts).to_csv(index=False).encode(), "activity_full.csv",
                             "text/csv", use_container_width=True, on_click=log, args=("Admin export", "Full activity log"))
        c[1].download_button("Sessions", sessions_view(sess).to_csv(index=False).encode(), "sessions_full.csv",
                             "text/csv", use_container_width=True, on_click=log, args=("Admin export", "All sessions"))
        c[2].download_button("Users", users.drop(columns=["id"]).to_csv(index=False).encode(), "users.csv",
                             "text/csv", use_container_width=True, on_click=log, args=("Admin export", "User list"))
        st.markdown("#### 🧾 Danger zone")
        with st.expander("Clear the activity log"):
            st.warning("Deletes every activity event. Login sessions and user accounts are kept.")
            confirm = st.checkbox("I understand this cannot be undone")
            if st.button("Clear activity log", type="primary", disabled=not confirm):
                count = len(acts)
                auth.clear_activity()
                log("Activity log cleared", f"{count} events deleted")
                st.rerun()


# ============================================================
# MAIN
# ============================================================

USER_PAGES = {
    "🏠 Dashboard": page_dashboard,
    "📄 Resume Analyzer": page_analyzer,
    "🧹 Autoencoder": page_autoencoder,
    "🌫️ Diffusion": page_diffusion,
    "🧬 CVAE / VAE": page_cvae,
    "🧪 GAN Candidate Generator": page_gan,
    "🤖 Transformer V3": page_transformer,
    "✨ AI Assistant": page_assistant,
}
# Admins only monitor the platform — no AI models are loaded for them.
ADMIN_PAGES = {
    "🛡️ Admin Panel": page_admin,
    "📊 History": page_history,
}


def logout():
    started = st.session_state.get("login_time")
    duration = fmt_duration((datetime.now() - started).total_seconds()) if started else "—"
    log("Logout", f"Signed out after {duration}")
    if st.session_state.get("session_id"):
        auth.end_session(st.session_state.session_id)
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


def keep_session_alive():
    """Mark the login session as active (throttled to one DB write every 30 s)."""
    if time.time() - st.session_state.get("last_touch", 0) > 30:
        auth.touch_session(st.session_state.session_id)
        st.session_state.last_touch = time.time()


def _render_model_status(panel):
    hub = model_hub()
    with panel.container():
        for spec in active_specs():
            key = spec["key"]
            state = hub.status(key)
            if state == "loaded":
                label = f"ready · {hub.seconds.get(key, 0):.1f}s"
            elif state == "failed":
                label = "failed"
            else:
                label = f"loading… {hub.elapsed():.0f}s"
            ui.model_status(f"{spec['name']} · {label}", spec["files"], state)


def watch_models(panel):
    """Runs at the very end of the script (page already drawn): refresh the sidebar status while
    models load, then rerun once so the "all loaded" notification fires. Any click interrupts this
    loop safely and the next run picks it up again."""
    hub = model_hub()
    if not hub.busy:
        return
    while hub.busy:
        time.sleep(0.5)
        _render_model_status(panel)
    st.rerun()


def announce_models():
    hub = model_hub()
    if hub.all_loaded and not st.session_state.get("models_announced"):
        st.session_state.models_announced = True
        st.toast("All models loaded successfully", icon="✅")
    active = {spec["key"] for spec in active_specs()}
    for key, err in hub.errors.items():
        if key in active and not st.session_state.get(f"model_err_{key}"):
            st.session_state[f"model_err_{key}"] = True
            name = next(s["name"] for s in MODEL_SPECS if s["key"] == key)
            st.toast(f"{name} failed to load: {err}", icon="❌")


def sidebar(pages):
    user = st.session_state.user
    with st.sidebar:
        ui.sidebar_brand()
        since = st.session_state.get("login_time")
        ui.profile_card(user, since.strftime("%d %b, %I:%M %p") if since else "")
        ui.side_label("Navigation")
        if st.session_state.get("nav") not in pages:      # e.g. a tab left on a page that is hidden now
            st.session_state.pop("nav", None)
        page = st.radio("Navigation", list(pages), label_visibility="collapsed", key="nav")
        panel = None
        if not is_admin():
            ui.side_label("Model status")
            panel = st.empty()
            _render_model_status(panel)
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            logout()
    return page, panel


def main():
    if not st.session_state.get("user"):
        login_page()
        model_hub().warm_up(delay=1.5)  # load models while the user types, after the page is on screen
        return
    user = st.session_state.user
    user.setdefault("role", "user")
    if not st.session_state.get("session_id"):
        st.session_state.client = client_info()
        st.session_state.session_id = auth.start_session(user, st.session_state.client)
        st.session_state.login_time = datetime.now()
        log("Session resumed", "Existing browser session re-registered")
    keep_session_alive()
    if not is_admin():
        model_hub().warm_up()
        announce_models()
    pages = ADMIN_PAGES if is_admin() else USER_PAGES
    page, panel = sidebar(pages)
    if st.session_state.get("last_page") != page:
        st.session_state.last_page = page
        log("Page view", page.split(" ", 1)[1])
    pages[page]()
    if panel is not None:
        watch_models(panel)


main()
