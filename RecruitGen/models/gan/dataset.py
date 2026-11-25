"""
GAN dataset — Kaggle Resume Dataset → 9 RecruitGen features
------------------------------------------------------------
Source: https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset

    Resume/Resume.csv               ID, Resume_str, Resume_html, Category   (2,484 resumes, 24 categories)
    data/data/<CATEGORY>/<ID>.pdf   the same resumes as PDF files (not used here)

The dataset contains raw resume text and a job category only — none of the 9 numerical features.
They are extracted from `Resume_str` with the same rule-based extractor the app uses
(models/transformer_v3/feature_extractor.py), so generated profiles live in the app's feature space.

Preprocessing:
    1. drop resumes too short for the extractor (< 20 characters)
    2. drop exact duplicate resume texts (keep the first)
    3. extract the 9 features
    4. clip each feature to the input range the app accepts (extractor outliers such as "500 certificates")
    5. stratified train / validation / test split — the test split is never used for training or selection
"""

import os
import glob

import numpy as np
import pandas as pd

from models.transformer_v3.feature_extractor import FEATURE_ORDER, extract_features

KAGGLE_DATASET = "snehaanbhawal/resume-dataset"
KAGGLE_URL = "https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset"

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DATA_DIR = os.path.join(PROJECT_DIR, "data", "kaggle_resume")          # manual download location
PROCESSED_PATH = os.path.join(PROJECT_DIR, "data", "gan", "kaggle_features.csv")

FEATURE_NAMES = list(FEATURE_ORDER)
INTEGER_FEATURES = ("projects_count", "certification_count", "education_level")

# Same limits as the feature inputs in app.py
FEATURE_BOUNDS = {
    "experience_years": (0.0, 20.0), "projects_count": (0.0, 20.0), "programming_score": (0.0, 100.0),
    "data_analytics_score": (0.0, 100.0), "ml_ai_score": (0.0, 100.0), "cloud_score": (0.0, 100.0),
    "communication_score": (0.0, 100.0), "certification_count": (0.0, 15.0), "education_level": (1.0, 5.0),
}

REQUIRED_COLUMNS = ["ID", "Resume_str", "Category"]
MIN_TEXT_LENGTH = 20


def category_label(category):
    """INFORMATION-TECHNOLOGY -> Information Technology; short acronyms (HR, BPO) stay upper case."""
    words = str(category).strip().replace("_", "-").split("-")
    return " ".join(w.upper() if len(w) <= 3 else w.capitalize() for w in words if w)


def find_resume_csv(path=None):
    """Locate Resume.csv: an explicit file/folder, $RECRUITGEN_KAGGLE_DIR, data/kaggle_resume/ or the kagglehub cache."""
    candidates = [path, os.getenv("RECRUITGEN_KAGGLE_DIR"), DEFAULT_DATA_DIR,
                  os.path.join(os.path.expanduser("~"), ".cache", "kagglehub", "datasets",
                               *KAGGLE_DATASET.split("/"))]
    for base in filter(None, candidates):
        if os.path.isfile(base):
            return base
        if os.path.isdir(base):
            hits = sorted(glob.glob(os.path.join(base, "**", "Resume.csv"), recursive=True))
            if hits:
                return hits[-1]
    return None


def download_dataset():
    """Download with kagglehub (public dataset; uses ~/.kaggle/kaggle.json if present). Returns the Resume.csv path."""
    import kagglehub

    return find_resume_csv(kagglehub.dataset_download(KAGGLE_DATASET))


def load_raw(csv_path):
    raw = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"{csv_path} is not the Kaggle Resume.csv (missing columns: {', '.join(missing)})")
    return raw


def build_feature_table(raw):
    """Raw Kaggle rows -> one row per resume with the 9 features and the job role. Returns (table, report)."""
    report = {"raw_rows": int(len(raw)), "raw_categories": int(raw["Category"].nunique())}
    text = raw["Resume_str"].fillna("").astype(str)

    short = text.str.strip().str.len() < MIN_TEXT_LENGTH
    duplicate = text.duplicated(keep="first") & ~short
    keep = ~(short | duplicate)
    report["dropped_too_short"] = int(short.sum())
    report["dropped_duplicate_text"] = int(duplicate.sum())

    kept = raw[keep].reset_index(drop=True)
    features = pd.DataFrame([extract_features(t) for t in text[keep]])[FEATURE_NAMES].astype(float)

    clipped = {}
    for name, (lo, hi) in FEATURE_BOUNDS.items():
        outside = (features[name] < lo) | (features[name] > hi)
        if outside.any():
            clipped[name] = int(outside.sum())
        features[name] = features[name].clip(lo, hi)
    report["clipped_values"] = clipped

    table = pd.concat([
        pd.DataFrame({"resume_id": kept["ID"].values, "category": kept["Category"].values,
                      "job_role": kept["Category"].map(category_label).values}),
        features,
    ], axis=1)
    report["rows"] = int(len(table))
    report["role_counts"] = {k: int(v) for k, v in table["job_role"].value_counts().sort_index().items()}
    return table, report


def split_table(table, seed=42, val_size=0.15, test_size=0.15):
    """Stratified train / validation / test split (adds a `split` column)."""
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(table))
    train_val, test = train_test_split(idx, test_size=test_size, random_state=seed, stratify=table["job_role"])
    train, val = train_test_split(train_val, test_size=val_size / (1 - test_size), random_state=seed,
                                  stratify=table["job_role"].iloc[train_val])
    out = table.copy()
    out["split"] = "train"
    out.loc[out.index[val], "split"] = "val"
    out.loc[out.index[test], "split"] = "test"
    return out


def save_processed(table, path=PROCESSED_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    table.to_csv(path, index=False)
    return path


def load_processed(path=PROCESSED_PATH):
    return pd.read_csv(path) if os.path.exists(path) else None
