"""
GAN evaluation — every number is computed here from real data and generated samples
-----------------------------------------------------------------------------------
1. Quality    per-feature KS statistic + Wasserstein distance, correlation-matrix gap, per-role mean gap,
              classifier two-sample test (C2ST ROC-AUC: 0.5 = real and synthetic are indistinguishable)
2. Diversity  unique-row ratio, exact copies of training rows, distance to closest training record (DCR),
              mean pairwise distance relative to real data
3. Utility    downstream job-role classifier (Random Forest on the 9 features) trained on
                  real only  vs  real + synthetic  vs  real + random oversampling
              and evaluated on the SAME untouched real test split; the augmentation strategy is chosen on the
              validation split. Also TSTR (train on synthetic only, test on real).

Every metric is reported next to a real-data reference (real train rows scored exactly like synthetic rows),
because the extractor's discrete features make some values (e.g. exact matches) high even for real data.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

from models.gan.dataset import FEATURE_BOUNDS, FEATURE_NAMES

STRATEGIES = [
    {"key": "ratio_0.5", "label": "+50% synthetic per role", "ratio": 0.5},
    {"key": "ratio_1.0", "label": "+100% synthetic per role", "ratio": 1.0},
    {"key": "ratio_2.0", "label": "+200% synthetic per role", "ratio": 2.0},
    {"key": "balance", "label": "Balance every role to the largest", "ratio": None},
]


def unit(values):
    lo = np.array([FEATURE_BOUNDS[f][0] for f in FEATURE_NAMES])
    hi = np.array([FEATURE_BOUNDS[f][1] for f in FEATURE_NAMES])
    return (np.asarray(values, dtype=np.float64) - lo) / (hi - lo)


def _role_matrix(roles, role_names):
    out = np.zeros((len(roles), len(role_names)))
    out[np.arange(len(roles)), [role_names.index(r) for r in roles]] = 1.0
    return out


def _corr(values):
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.corrcoef(values, rowvar=False)
    return np.nan_to_num(c)


# ============================================================
# 1. QUALITY
# ============================================================

def c2st_auc(real_u, real_roles, syn_u, syn_roles, role_names, seed=0, folds=5):
    """Classifier two-sample test: ROC-AUC of a classifier separating real from synthetic (features + role)."""
    X = np.vstack([np.hstack([real_u, _role_matrix(real_roles, role_names)]),
                   np.hstack([syn_u, _role_matrix(syn_roles, role_names)])])
    y = np.r_[np.zeros(len(real_u)), np.ones(len(syn_u))]
    aucs = []
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, random_state=seed, n_jobs=-1)
        clf.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs))


def quality_metrics(real_df, syn_values, syn_roles, role_names, seed=0):
    """Compare generated rows with held-out real rows (same role mix)."""
    real = real_df[FEATURE_NAMES].values.astype(float)
    real_u, syn_u = unit(real), unit(syn_values)
    per_feature = []
    for j, name in enumerate(FEATURE_NAMES):
        per_feature.append({
            "feature": name,
            "real_mean": float(real[:, j].mean()), "synthetic_mean": float(np.mean(syn_values[:, j])),
            "real_std": float(real[:, j].std()), "synthetic_std": float(np.std(syn_values[:, j])),
            "ks_statistic": float(ks_2samp(real[:, j], syn_values[:, j]).statistic),
            "wasserstein_0_1": float(wasserstein_distance(real_u[:, j], syn_u[:, j])),
        })
    roles = np.asarray(syn_roles)
    real_roles = real_df["job_role"].values
    role_gaps = []
    for role in role_names:
        r, s = real_u[real_roles == role], syn_u[roles == role]
        if len(r) and len(s):
            role_gaps.append(float(np.mean(np.abs(r.mean(0) - s.mean(0)))))
    auc, auc_std = c2st_auc(real_u, list(real_roles), syn_u, list(syn_roles), role_names, seed)
    return {
        "per_feature": per_feature,
        "mean_ks_statistic": float(np.mean([p["ks_statistic"] for p in per_feature])),
        "mean_wasserstein_0_1": float(np.mean([p["wasserstein_0_1"] for p in per_feature])),
        "correlation_gap": float(np.mean(np.abs(_corr(real_u) - _corr(syn_u)))),
        "per_role_mean_gap_0_1": float(np.mean(role_gaps)) if role_gaps else None,
        "c2st_auc": auc, "c2st_auc_std": auc_std,
    }


# ============================================================
# 2. DIVERSITY
# ============================================================

def diversity_metrics(syn_values, train_df, seed=0, max_pairs_rows=1500):
    rng = np.random.default_rng(seed)
    syn_u = unit(syn_values)
    train_u = unit(train_df[FEATURE_NAMES].values)
    rounded = pd.DataFrame(np.round(syn_values, 4))
    train_rows = set(map(tuple, np.round(train_df[FEATURE_NAMES].values.astype(float), 4)))
    dcr = NearestNeighbors(n_neighbors=1).fit(train_u).kneighbors(syn_u)[0][:, 0]

    def pairwise(u):
        if len(u) > max_pairs_rows:
            u = u[rng.choice(len(u), max_pairs_rows, replace=False)]
        return float(pdist(u).mean()) if len(u) > 1 else 0.0

    return {
        "rows": int(len(syn_values)),
        "unique_row_ratio": float(len(rounded.drop_duplicates()) / max(1, len(rounded))),
        "exact_train_copy_rate": float(np.mean([tuple(r) in train_rows for r in rounded.values])),
        "dcr_median": float(np.median(dcr)),
        "dcr_mean": float(np.mean(dcr)),
        "mean_pairwise_distance": pairwise(syn_u),
        "train_mean_pairwise_distance": pairwise(train_u),
    }


def real_reference(train_df, test_df, role_names, seed=0):
    """Score real held-out rows exactly like synthetic rows. For quality the "generator" is a role-matched
    sample of real training rows; for diversity the held-out test rows are compared with the training rows."""
    rng = np.random.default_rng(seed)
    parts = []
    for role, n in test_df["job_role"].value_counts().items():
        pool = train_df[train_df["job_role"] == role]
        parts.append(pool.iloc[rng.choice(len(pool), n, replace=len(pool) < n)])
    sample = pd.concat(parts)
    return {
        "quality": quality_metrics(test_df, sample[FEATURE_NAMES].values.astype(float),
                                   list(sample["job_role"]), role_names, seed),
        "diversity": diversity_metrics(test_df[FEATURE_NAMES].values.astype(float), train_df, seed),
    }


def generator_report(model, train_df, test_df, role_names, seed=0, multiplier=1):
    """Quality + diversity of one generator against the held-out real test rows (same role mix)."""
    counts = {r: int(n) * multiplier for r, n in test_df["job_role"].value_counts().items()}
    values, roles = model.generate_array(counts, seed=seed)
    return {"quality": quality_metrics(test_df, values, roles, role_names, seed),
            "diversity": diversity_metrics(values, train_df, seed)}


# ============================================================
# 3. DOWNSTREAM JOB-ROLE CLASSIFIER
# ============================================================

def make_classifier(seed):
    return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)


def classification_metrics(y_true, y_pred, role_names, detailed=False):
    kw = {"labels": role_names, "zero_division": 0}
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", **kw)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", **kw)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", **kw)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", **kw)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", **kw)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", **kw)),
    }
    if detailed:
        out["classification_report"] = classification_report(y_true, y_pred, output_dict=True, **kw)
        out["classification_report_text"] = classification_report(y_true, y_pred, digits=4, **kw)
        out["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=role_names).tolist()
    return out


def strategy_counts(train_df, strategy):
    counts = train_df["job_role"].value_counts()
    if strategy["ratio"] is None:
        return {r: int(counts.max() - c) for r, c in counts.items()}
    return {r: int(round(strategy["ratio"] * c)) for r, c in counts.items()}


def _fit_eval(X, y, seed, evals, role_names, detailed=False):
    clf = make_classifier(seed).fit(X, y)
    return {name: classification_metrics(yt, clf.predict(Xt), role_names, detailed) for name, (Xt, yt) in evals.items()}


def _summary(runs, split):
    keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]
    return {k: {"mean": float(np.mean([r[split][k] for r in runs])), "std": float(np.std([r[split][k] for r in runs]))}
            for k in keys}


def augmentation_experiment(train_df, val_df, test_df, samplers, role_names, seeds=(0, 1, 2, 3, 4), log=print):
    """samplers: {name: object with generate_array(counts, seed)}. The test split is only used for reporting."""
    X_tr, y_tr = train_df[FEATURE_NAMES].values.astype(float), train_df["job_role"].values
    evals = {"val": (val_df[FEATURE_NAMES].values.astype(float), val_df["job_role"].values),
             "test": (test_df[FEATURE_NAMES].values.astype(float), test_df["job_role"].values)}

    real_runs = [_fit_eval(X_tr, y_tr, s, evals, role_names) for s in seeds]
    result = {
        "classifier": "RandomForestClassifier(n_estimators=300) on the 9 features",
        "seeds": list(seeds),
        "split_sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "real_only": {"val": _summary(real_runs, "val"), "test": _summary(real_runs, "test"),
                      "per_seed_test_f1_macro": [r["test"]["f1_macro"] for r in real_runs]},
        "samplers": {},
    }
    if log:
        log(f"real only            | test macro-F1 {result['real_only']['test']['f1_macro']['mean']:.4f}")

    for name, sampler in samplers.items():
        per_strategy = {}
        for strategy in STRATEGIES:
            counts = strategy_counts(train_df, strategy)
            if sum(counts.values()) == 0:      # e.g. "balance" on an already balanced dataset = no augmentation
                continue
            runs = []
            for s in seeds:
                values, roles = sampler.generate_array(counts, seed=1000 + s)
                X = np.vstack([X_tr, values]) if len(values) else X_tr
                y = np.r_[y_tr, roles] if len(values) else y_tr
                runs.append(_fit_eval(X, y, s, evals, role_names))
            per_strategy[strategy["key"]] = {
                "label": strategy["label"], "synthetic_rows": int(sum(counts.values())),
                "val": _summary(runs, "val"), "test": _summary(runs, "test"),
                "per_seed_test_f1_macro": [r["test"]["f1_macro"] for r in runs],
            }
        chosen = max(per_strategy, key=lambda k: per_strategy[k]["val"]["f1_macro"]["mean"])
        deltas = np.array(per_strategy[chosen]["per_seed_test_f1_macro"]) - np.array(
            result["real_only"]["per_seed_test_f1_macro"])
        result["samplers"][name] = {
            "strategies": per_strategy,
            "chosen_strategy": chosen,
            "chosen_by": "highest mean validation macro-F1",
            "test_f1_macro_delta_vs_real": {"mean": float(deltas.mean()), "std": float(deltas.std()),
                                            "seeds_improved": int((deltas > 0).sum()), "seeds": len(deltas)},
        }
        if log:
            c = per_strategy[chosen]
            log(f"{name:20s} | chosen {chosen:10s} | test macro-F1 {c['test']['f1_macro']['mean']:.4f} "
                f"(Δ {deltas.mean():+.4f} ± {deltas.std():.4f}, improved on {(deltas > 0).sum()}/{len(deltas)} seeds)")
    return result


def detailed_comparison(train_df, test_df, sampler, strategy_key, role_names, seed=0):
    """Full classification report + confusion matrix on the test split for real-only vs the chosen augmentation."""
    X_tr, y_tr = train_df[FEATURE_NAMES].values.astype(float), train_df["job_role"].values
    evals = {"test": (test_df[FEATURE_NAMES].values.astype(float), test_df["job_role"].values)}
    strategy = next(s for s in STRATEGIES if s["key"] == strategy_key)
    values, roles = sampler.generate_array(strategy_counts(train_df, strategy), seed=1000 + seed)
    return {
        "seed": seed,
        "strategy": strategy_key,
        "real_only": _fit_eval(X_tr, y_tr, seed, evals, role_names, detailed=True)["test"],
        "augmented": _fit_eval(np.vstack([X_tr, values]), np.r_[y_tr, roles], seed, evals, role_names,
                               detailed=True)["test"],
    }


def tstr(train_df, test_df, sampler, role_names, seeds=(0, 1, 2, 3, 4)):
    """Train on synthetic only (same role counts as the real train split), test on real."""
    counts = train_df["job_role"].value_counts().to_dict()
    evals = {"test": (test_df[FEATURE_NAMES].values.astype(float), test_df["job_role"].values)}
    runs = []
    for s in seeds:
        values, roles = sampler.generate_array(counts, seed=2000 + s)
        runs.append(_fit_eval(values, np.asarray(roles), s, evals, role_names))
    return _summary(runs, "test")
