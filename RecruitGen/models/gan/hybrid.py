"""
Hybrid GAN + LLM generator — 5 RecruitGen roles
-----------------------------------------------
1. Seed data      the LLM suggests structured profiles per role (llm_assist.seed_prompt); each one is converted to
                  resume text, its 9 features are extracted with the app's extractor and it is kept only if it passes
                  validate_profile + duplicate checks.                      -> data/gan_hybrid/llm_seed_profiles.csv
2. GAN            the conditional WGAN-GP (training.py) learns the joint feature distribution of the validated seed
                  train split.                                              -> models/gan/hybrid/
3. Generation     GAN samples -> controlled filtering: per-role plausibility (nearest-neighbour distance to the seed
                  data), duplicates, exact copies of training rows, the user's experience range / skill preferences.
4. Descriptions   optional: the LLM writes content for accepted GAN rows; it is kept only if re-extracting the features
                  from it reproduces the GAN numbers. The LLM never changes a number.

Data sources are always labelled: feature values "GAN", descriptions "AI-assisted (validated)", seed rows
"AI-assisted seed (validated)". Without the LLM everything still works GAN-only.
"""

import json
import os
import re

import numpy as np
import pandas as pd

from models.gan import llm_assist as llm
from models.gan.dataset import FEATURE_BOUNDS, FEATURE_NAMES, PROJECT_DIR, find_resume_csv, load_raw
from models.gan.gan import MODEL_DIR, CandidateGAN

HYBRID_DIR = os.path.join(MODEL_DIR, "hybrid")
SEED_DIR = os.path.join(PROJECT_DIR, "data", "gan_hybrid")
SEED_PATH = os.path.join(SEED_DIR, "llm_seed_profiles.csv")
ROLES = llm.ROLES

GAN_SOURCE = "GAN"
SEED_SOURCE = "AI-assisted seed (validated)"
DESCRIPTION_SOURCE = "AI-assisted (validated)"

SKILL_FEATURES = {"Programming": "programming_score", "Data Analytics": "data_analytics_score",
                  "ML / AI": "ml_ai_score", "Cloud": "cloud_score", "Communication": "communication_score"}


def unit(values):
    lo = np.array([FEATURE_BOUNDS[f][0] for f in FEATURE_NAMES])
    hi = np.array([FEATURE_BOUNDS[f][1] for f in FEATURE_NAMES])
    return (np.asarray(values, dtype=np.float64) - lo) / (hi - lo)


# ============================================================
# 1. LLM seed dataset
# ============================================================

def _jaccard(a, b):
    a, b = {x.lower() for x in a}, {x.lower() for x in b}
    return len(a & b) / max(1, len(a | b))


def build_seed_dataset(calls_per_role=12, per_call=10, client=None, log=print, real_hints=None):
    """Ask the LLM for profiles, validate them, drop duplicates. Returns (table, report)."""
    client = client or llm.LLMClient()
    if not client.available():
        raise RuntimeError("The AI chat service is not configured (.streamlit/secrets.toml [ai] or AI_* env vars).")
    rows, rejected, statuses = [], {}, {}
    for role in ROLES:
        kept_role = []
        for batch in range(calls_per_role):
            data, status = client.chat_json(llm.seed_prompt(role, per_call, batch, (real_hints or {}).get(role)))
            statuses[status] = statuses.get(status, 0) + 1
            for raw in (data or {}).get("profiles", []) if isinstance(data, dict) else []:
                p, err = llm.normalise_profile(raw)
                if p is None:
                    rejected[err] = rejected.get(err, 0) + 1
                    continue
                feats, reason = llm.validate_profile(p, role)
                if feats is None:
                    rejected[reason] = rejected.get(reason, 0) + 1
                    continue
                vec = tuple(feats[f] for f in FEATURE_NAMES)
                if any(vec == k["vec"] for k in kept_role):
                    rejected["duplicate feature vector"] = rejected.get("duplicate feature vector", 0) + 1
                    continue
                if any(_jaccard(p["skills"], k["skills"]) >= 0.9 for k in kept_role):
                    rejected["near-duplicate skill set"] = rejected.get("near-duplicate skill set", 0) + 1
                    continue
                kept_role.append({"vec": vec, "skills": p["skills"]})
                rows.append({"profile_id": f"SEED-{len(rows) + 1:04d}", "job_role": role, "source": SEED_SOURCE,
                             **feats, "title": p["title"], "skills": json.dumps(p["skills"]),
                             "projects": json.dumps(p["projects"]), "certifications": json.dumps(p["certifications"]),
                             "degree": p["degree"], "soft_skills": json.dumps(p["soft_skills"]),
                             "summary": p["summary"], "batch": batch})
        if log:
            log(f"{role:18s}: {len(kept_role)} validated profiles · calls {client.calls}")
    table = pd.DataFrame(rows)
    report = {"requested": len(ROLES) * calls_per_role * per_call, "kept": len(table), "rejected": rejected,
              "call_status": statuses, "per_role": table["job_role"].value_counts().to_dict() if len(table) else {}}
    return table, report


def save_seed(table, path=SEED_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    table.to_csv(path, index=False)
    return path


def load_seed(path=SEED_PATH):
    return pd.read_csv(path) if os.path.exists(path) else None


# ============================================================
# Real check: Kaggle resumes whose stated job title matches a RecruitGen role
# ============================================================

TITLE_RULES = [  # applied to the first line (job title) of each Kaggle resume, in this order
    ("AI/ML Engineer", r"machine learning|data scientist|artificial intelligence|deep learning"),
    ("Cloud Engineer", r"\bcloud\b|devops|site reliability|\baws\b|\bazure\b"),
    ("Data Analyst", r"\bdata analyst|business intelligence|\bbi analyst|reporting analyst|data analytics"),
    ("Software Engineer", r"software (engineer|developer)|\bdeveloper\b|programmer|software engineering"),
    ("General IT", r"information technology|\bit (specialist|manager|consultant|coordinator|support|director|"
                   r"technician|administrator|analyst)|network (admin|engineer|technician)|system(s)? administrator|"
                   r"help ?desk|desktop support|technical support"),
]


def kaggle_title_subset(csv_path=None, with_text=False):
    """Real Kaggle resumes labelled by their own job title (weak labels). Returns a table with 9 features + job_role."""
    from models.transformer_v3.feature_extractor import extract_features

    csv_path = csv_path or find_resume_csv()
    if not csv_path:
        return None
    raw = load_raw(csv_path).drop_duplicates("Resume_str")
    titles = raw["Resume_str"].fillna("").str.strip().str.split(r"\s{3,}|\n").str[0].str.strip().str.lower()

    def label(t):
        return next((role for role, pat in TITLE_RULES if re.search(pat, t)), None)

    raw = raw.assign(title=titles, job_role=titles.map(label)).dropna(subset=["job_role"])
    feats = pd.DataFrame([extract_features(t) for t in raw["Resume_str"]])[FEATURE_NAMES].astype(float)
    for name, (lo, hi) in FEATURE_BOUNDS.items():
        feats[name] = feats[name].clip(lo, hi)
    cols = ["ID", "title", "job_role"] + (["Resume_str"] if with_text else [])
    return pd.concat([raw[cols].reset_index(drop=True), feats], axis=1)


def real_pattern_hints(subset, min_rows=5, top=10):
    """Most frequent extractor keywords and education levels in the real title-matched resumes, per role.
    Roles with fewer than `min_rows` real resumes get no hint (nothing real to learn from)."""
    from models.transformer_v3 import feature_extractor as fx

    groups = (fx.PROGRAMMING_KEYWORDS + fx.DATA_ANALYTICS_KEYWORDS + fx.ML_AI_KEYWORDS + fx.CLOUD_KEYWORDS)
    keywords = sorted({k for k in groups if len(k) > 3})            # skip 2-3 letter substrings such as "ai", "ml"
    hints = {}
    for role, g in subset.groupby("job_role"):
        if len(g) < min_rows:
            continue
        texts = g["Resume_str"].str.lower()
        freq = {k: float(texts.str.contains(re.escape(k), regex=True).mean()) for k in keywords}
        best = [f"{k} ({v:.0%})" for k, v in sorted(freq.items(), key=lambda kv: -kv[1])[:top] if v > 0]
        edu = g["education_level"].map({1: "diploma/other", 2: "bachelor's", 3: "master's", 5: "PhD"}).value_counts(
            normalize=True)
        hints[role] = (f"{len(g)} real resumes with this title; most common tools: {', '.join(best)}; education: "
                       + ", ".join(f"{k} {v:.0%}" for k, v in edu.items()))
    return hints


# ============================================================
# 2-3. Hybrid generator: GAN + controlled filtering
# ============================================================

class HybridGenerator:
    def __init__(self, model_dir=HYBRID_DIR):
        self.gan = CandidateGAN(model_dir)
        self.config = self.gan.config
        self.role_names = self.gan.role_names
        ref = pd.read_csv(os.path.join(model_dir, "reference.csv"))
        self.reference = {r: unit(g[FEATURE_NAMES].values) for r, g in ref.groupby("job_role")}
        self.train_rows = {r: set(map(tuple, np.round(g[FEATURE_NAMES].values.astype(float), 4)))
                           for r, g in ref.groupby("job_role")}
        self.thresholds = self.config["plausibility"]["thresholds"]

    @staticmethod
    def fit_thresholds(reference, quantile=0.95):
        """Per-role plausibility radius = quantile of leave-one-out nearest-neighbour distances in the seed data."""
        out = {}
        for role, g in reference.groupby("job_role"):
            u = unit(g[FEATURE_NAMES].values)
            d = np.sqrt(((u[:, None, :] - u[None, :, :]) ** 2).sum(-1))
            np.fill_diagonal(d, np.inf)
            out[role] = float(np.quantile(d.min(1), quantile))
        return out

    @property
    def num_params(self):
        return self.gan.num_params

    def nn_distance(self, role, values):
        ref = self.reference[role]
        u = unit(values)
        return np.sqrt(((u[:, None, :] - ref[None, :, :]) ** 2).sum(-1)).min(1)

    def generate(self, role, n=10, seed=0, noise_scale=1.0, experience_range=None, min_scores=None,
                 plausibility=True, dedupe=True, exclude_train_copies=True, max_rounds=25):
        """Rejection sampling: draw GAN batches until n profiles pass every enabled filter. Returns (rows, stats)."""
        stats = {"sampled": 0, "accepted": 0, "rejected": {}}
        kept, seen = [], set()
        rng_seed = int(seed)
        for round_ in range(max_rounds):
            if len(kept) >= n:
                break
            batch = max(32, 3 * (n - len(kept)))
            values, _ = self.gan.generate_array({role: batch}, seed=rng_seed + round_ * 7919, noise_scale=noise_scale)
            stats["sampled"] += len(values)
            dist = self.nn_distance(role, values)
            for vec, d in zip(values, dist):
                key = tuple(np.round(vec, 4))
                reason = None
                if plausibility and d > self.thresholds[role]:
                    reason = "implausible combination (far from seed data)"
                elif experience_range and not experience_range[0] <= vec[0] <= experience_range[1]:
                    reason = "outside experience range"
                elif min_scores and any(vec[FEATURE_NAMES.index(f)] < v for f, v in min_scores.items()):
                    reason = "below skill preference"
                elif dedupe and key in seen:
                    reason = "duplicate"
                elif exclude_train_copies and key in self.train_rows.get(role, set()):
                    reason = "copy of a training row"
                if reason:
                    stats["rejected"][reason] = stats["rejected"].get(reason, 0) + 1
                    continue
                seen.add(key)
                kept.append(vec)
                if len(kept) >= n:
                    break
        stats["accepted"] = len(kept)
        rows = []
        for vec in kept:
            row = {name: int(round(v)) if name in self.gan.integer_features else round(float(v), 1)
                   for name, v in zip(FEATURE_NAMES, vec)}
            row["job_role"] = role
            rows.append(row)
        return rows, stats

    def generate_array(self, counts, seed=None, **kwargs):
        """Evaluation interface (same as CandidateGAN): default filters, no user constraints."""
        blocks, roles = [], []
        for i, (role, n) in enumerate(counts.items()):
            if int(n) <= 0:
                continue
            rows, _ = self.generate(role, int(n), seed=(seed or 0) + 101 * i, **kwargs)
            blocks.append(np.array([[r[f] for f in FEATURE_NAMES] for r in rows], dtype=np.float64))
            roles += [role] * len(rows)
        if not blocks:
            return np.zeros((0, len(FEATURE_NAMES))), []
        return np.concatenate(blocks), roles


# ============================================================
# 4. Validated descriptions for GAN rows
# ============================================================

TOLERANCE = {"experience_years": 0.05, "projects_count": 0, "certification_count": 0, "education_level": 0,
             "programming_score": 15, "data_analytics_score": 15, "ml_ai_score": 15, "cloud_score": 15,
             "communication_score": 10}


def describe_rows(role, rows, client=None, chunk=5):
    """Ask the LLM for content matching each GAN row, keep it only if the extracted features match the GAN numbers
    within TOLERANCE. Returns (descriptions {index: dict}, status summary)."""
    client = client or llm.LLMClient()
    out, summary = {}, {"validated": 0, "failed_validation": 0, "missing": 0, "status": {}}
    if not client.available():
        summary["status"]["unavailable"] = len(rows)
        return out, summary
    for start in range(0, len(rows), chunk):
        part = list(enumerate(rows[start:start + chunk], start))
        data, status = client.chat_json(llm.enrichment_prompt(role, [(f"P{i}", r) for i, r in part]),
                                        max_tokens=4000, temperature=0.7)
        summary["status"][status] = summary["status"].get(status, 0) + 1
        answers = {str(p.get("id")): p for p in (data or {}).get("profiles", []) if isinstance(p, dict)} \
            if isinstance(data, dict) else {}
        for i, row in part:
            raw = answers.get(f"P{i}")
            if raw is None:
                summary["missing"] += 1
                continue
            raw["experience_years"] = row["experience_years"]
            p, err = llm.normalise_profile(raw)
            ok = p is not None
            if ok:
                feats = llm.fx.extract_features(llm.profile_text(p))
                ok = all(abs(float(feats[f]) - float(row[f])) <= TOLERANCE[f] for f in FEATURE_NAMES)
            if ok:
                summary["validated"] += 1
                out[i] = {"title": p["title"], "skills": ", ".join(p["skills"]), "projects": "; ".join(p["projects"]),
                          "certifications": "; ".join(p["certifications"]), "degree": p["degree"],
                          "summary": p["summary"]}
            else:
                summary["failed_validation"] += 1
    return out, summary


# ============================================================
# Batch quality statistics (used by the app comparison and the evaluation)
# ============================================================

def batch_stats(values, roles=None, train_values=None):
    """Diversity / duplicates / realism of one batch of feature vectors."""
    v = np.asarray(values, dtype=np.float64)
    if len(v) == 0:
        return {}
    df = pd.DataFrame(v, columns=FEATURE_NAMES)
    keys = [tuple(r) for r in np.round(v, 4)]
    u = unit(v)
    pair = float(np.sqrt(((u[:, None, :] - u[None, :, :]) ** 2).sum(-1))[np.triu_indices(len(u), 1)].mean()) \
        if len(u) > 1 else 0.0
    out = {
        "rows": int(len(v)),
        "duplicate_rate": float(1 - len(set(keys)) / len(keys)),
        "mean_pairwise_distance": pair,
        "distinct_values_per_feature": float(np.mean([df[f].nunique() for f in FEATURE_NAMES])),
        "experience_zero_share": float((df["experience_years"] == 0).mean()),
        "experience_mean": float(df["experience_years"].mean()),
        "experience_projects_corr": float(np.nan_to_num(df["experience_years"].corr(df["projects_count"]))),
        "many_certs_no_experience_share": float(((df["certification_count"] >= 8) & (df["experience_years"] < 1)).mean()),
    }
    if train_values is not None:
        train = set(map(tuple, np.round(np.asarray(train_values, dtype=np.float64), 4)))
        out["exact_train_copy_rate"] = float(np.mean([k in train for k in keys]))
    return out
