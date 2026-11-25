"""
Hybrid GAN pipeline — LLM seed → validation → 5-role GAN → filters → evaluation → models/gan/hybrid/
--------------------------------------------------------------------------------------------------
Run from the RecruitGen folder:

    python -m models.gan.hybrid_train                # full run (LLM calls are cached in data/llm_cache/)
    python -m models.gan.hybrid_train --stage seed   # only build / extend the validated seed dataset
    python -m models.gan.hybrid_train --quick        # fast smoke test (fewer calls, iterations and seeds)

Every number in metrics.json is computed here. The 5-role training/test data is LLM-generated (synthetic) because the
Kaggle dataset contains almost no resumes for these roles; the only real 5-role check is the small set of Kaggle
resumes whose own job title matches a role (kaggle_title_subset) — it is never used for training or selection.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

from models.gan import hybrid as hy
from models.gan.baselines import OversamplingBaseline
from models.gan.dataset import FEATURE_NAMES, load_processed, split_table
from models.gan.evaluation import (augmentation_experiment, classification_metrics, detailed_comparison,
                                   generator_report, make_classifier, real_reference, strategy_counts, STRATEGIES,
                                   tstr)
from models.gan.gan import MODEL_DIR, CandidateGAN, checkpoint_available, save_checkpoint
from models.gan.llm_assist import LLMClient
from models.gan.training import train_gan


def real_check(train, real, sampler=None, strategy_key=None, seeds=(0, 1, 2, 3, 4)):
    """Classifier trained on seed (+ synthetic) data, scored on the real title-matched Kaggle resumes."""
    labels = [r for r in hy.ROLES if r in set(real["job_role"])]
    X_tr, y_tr = train[FEATURE_NAMES].values.astype(float), train["job_role"].values
    X_real, y_real = real[FEATURE_NAMES].values.astype(float), real["job_role"].values
    runs, detail = [], None
    for s in seeds:
        X, y = X_tr, y_tr
        if sampler is not None:
            strategy = next(st for st in STRATEGIES if st["key"] == strategy_key)
            values, roles = sampler.generate_array(strategy_counts(train, strategy), seed=1000 + s)
            if len(values):
                X, y = np.vstack([X_tr, values]), np.r_[y_tr, roles]
        pred = make_classifier(s).fit(X, y).predict(X_real)
        runs.append(classification_metrics(y_real, pred, labels, detailed=(s == seeds[0])))
        if s == seeds[0]:
            detail = runs[-1]
    keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    return {"labels": labels, "rows": int(len(real)), "per_role": real["job_role"].value_counts().to_dict(),
            "majority_class_accuracy": float(real["job_role"].value_counts(normalize=True).max()),
            **{k: {"mean": float(np.mean([r[k] for r in runs])), "std": float(np.std([r[k] for r in runs]))} for k in keys},
            "detail_seed0": {k: detail[k] for k in ("classification_report", "classification_report_text",
                                                    "confusion_matrix")}}


def similarity_to_real(values, real):
    """Median distance (features scaled 0-1) from each row to its nearest real title-matched Kaggle record."""
    if real is None or not len(values):
        return None
    r = hy.unit(real[FEATURE_NAMES].values)
    u = hy.unit(values)
    return float(np.median(np.sqrt(((u[:, None, :] - r[None, :, :]) ** 2).sum(-1)).min(1)))


def compare_original(hybrid, gan, train, test, real, n_per_role=100, seed=7):
    """Original Kaggle-category GAN vs the 5-role raw GAN vs the hybrid (GAN + filters), same batch size."""
    out = {}
    if checkpoint_available(MODEL_DIR):
        original = CandidateGAN(MODEL_DIR)
        role = "Information Technology" if "Information Technology" in original.role_names else original.role_names[0]
        values, _ = original.generate_array({role: n_per_role * len(hy.ROLES)}, seed=seed)
        kaggle = load_processed()
        train_vals = kaggle[(kaggle["split"] == "train") & (kaggle["job_role"] == role)][FEATURE_NAMES].values \
            if kaggle is not None else None
        out["Original GAN (Kaggle category: %s)" % role] = {
            **hy.batch_stats(values, train_values=train_vals), "validity_rate": None,
            "similarity_to_real_median": similarity_to_real(values, real)}
    raw_vals, _ = gan.generate_array({r: n_per_role for r in hy.ROLES}, seed=seed)
    out["Improved: GAN only (5 roles, no filters)"] = {
        **hy.batch_stats(raw_vals, train_values=train[FEATURE_NAMES].values), "validity_rate": None,
        "similarity_to_real_median": similarity_to_real(raw_vals, real)}
    blocks, sampled, accepted = [], 0, 0
    for i, role in enumerate(hy.ROLES):
        rows, stats = hybrid.generate(role, n_per_role, seed=seed + i)
        blocks += [[r[f] for f in FEATURE_NAMES] for r in rows]
        sampled, accepted = sampled + stats["sampled"], accepted + stats["accepted"]
    hyb_vals = np.array(blocks, dtype=float)
    out["Improved: hybrid (GAN + validation filters)"] = {
        **hy.batch_stats(hyb_vals, train_values=train[FEATURE_NAMES].values),
        "validity_rate": accepted / max(1, sampled), "similarity_to_real_median": similarity_to_real(hyb_vals, real)}
    out["LLM seed test split (reference)"] = {
        **hy.batch_stats(test[FEATURE_NAMES].values, train_values=train[FEATURE_NAMES].values), "validity_rate": None,
        "similarity_to_real_median": similarity_to_real(test[FEATURE_NAMES].values, real)}
    return out


def run(calls_per_role=12, per_call=10, iterations=None, seed=42, quick=False, stage="all", seed_csv=None,
        out_dir=hy.HYBRID_DIR, log=print):
    start = time.time()
    client = LLMClient()
    real = hy.kaggle_title_subset(with_text=True)
    hints = hy.real_pattern_hints(real) if real is not None else {}
    if stage == "train":                     # reuse a saved seed dataset — no LLM calls
        seed_table = pd.read_csv(seed_csv or hy.SEED_PATH)
        report_path = os.path.splitext(seed_csv or hy.SEED_PATH)[0] + "_report.json"
        seed_report = json.load(open(report_path)) if os.path.exists(report_path) else {
            "requested": None, "kept": len(seed_table), "rejected": {}, "call_status": {}}
        log(f"Using saved seed dataset: {len(seed_table)} rows")
    else:
        if quick:
            calls_per_role = min(calls_per_role, 3)
        log(f"Building the validated LLM seed dataset ({calls_per_role} calls × {per_call} profiles per role) ...")
        seed_table, seed_report = hy.build_seed_dataset(calls_per_role, per_call, client, log, hints)
        path = hy.save_seed(seed_table)
        with open(os.path.splitext(path)[0] + "_report.json", "w") as f:
            json.dump(seed_report, f, indent=2)
        log(f"Seed: {seed_report['kept']}/{seed_report['requested']} kept · rejected {seed_report['rejected']} · "
            f"calls {seed_report['call_status']}")
        if stage == "seed":
            return seed_report

    table = split_table(seed_table, seed=seed)
    train, val, test = (table[table["split"] == s].reset_index(drop=True) for s in ("train", "val", "test"))
    iterations = iterations or (300 if quick else 6000)
    log(f"Training the 5-role GAN for {iterations} iterations on {len(train)} validated seed rows ...")
    model, critic, history, config = train_gan(train, val, hy.ROLES, iterations=iterations, seed=seed,
                                               eval_every=50 if quick else 100, dataset_report=seed_report, log=log)
    config["model_name"] = "RecruitGenHybridGAN"
    config["training"].update({"dataset": "LLM-suggested profiles validated with the app's feature extractor "
                                          "(synthetic seed data)", "dataset_url": None})
    config["plausibility"] = {"quantile": 0.95, "thresholds": hy.HybridGenerator.fit_thresholds(train)}
    save_checkpoint(model.generator, critic, config, out_dir)
    train[["job_role"] + FEATURE_NAMES].to_csv(os.path.join(out_dir, "reference.csv"), index=False)

    hybrid = hy.HybridGenerator(out_dir)
    a, _ = model.generate_array({hy.ROLES[0]: 16}, seed=123)
    b, _ = hybrid.gan.generate_array({hy.ROLES[0]: 16}, seed=123)
    check = {"reloaded": True, "identical_samples": bool(np.array_equal(a, b)), "params": hybrid.num_params}
    log(f"Checkpoint saved to {out_dir} · reload identical: {check['identical_samples']}")

    seeds = (0, 1) if quick else (0, 1, 2, 3, 4)
    log("Quality & diversity against the held-out seed test split ...")
    quality = {"Seed reference (train vs test)": real_reference(train, test, hy.ROLES),
               "GAN only": generator_report(model, train, test, hy.ROLES),
               "Hybrid (GAN + filters)": generator_report(hybrid, train, test, hy.ROLES)}
    log("Original vs improved batch statistics ...")
    comparison = compare_original(hybrid, model, train, test, real)
    log("Downstream classifier: seed only vs augmented (untouched seed test split) ...")
    samplers = {"Hybrid (GAN + filters)": hybrid, "GAN only": model, "Random oversampling": OversamplingBaseline(train)}
    augmentation = augmentation_experiment(train, val, test, samplers, hy.ROLES, seeds=seeds, log=log)
    chosen = augmentation["samplers"]["Hybrid (GAN + filters)"]["chosen_strategy"]
    detail = detailed_comparison(train, test, hybrid, chosen, hy.ROLES, seed=seeds[0])
    log("Real check on title-matched Kaggle resumes ...")
    real_eval = None
    if real is not None and len(real):
        real_feats = real.drop(columns=["Resume_str"])
        real_eval = {"Seed only": real_check(train, real_feats, seeds=seeds),
                     "Seed + hybrid": real_check(train, real_feats, hybrid, chosen, seeds),
                     "Seed + GAN only": real_check(train, real_feats, model,
                                                   augmentation["samplers"]["GAN only"]["chosen_strategy"], seeds)}
    tstr_scores = {name: tstr(train, test, s, hy.ROLES, seeds) for name, s in samplers.items()
                   if name != "Random oversampling"}

    metrics = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "quick_run": quick,
        "note": ("All values computed by models/gan/hybrid_train.py. Training, validation and test data are LLM-suggested "
                 "profiles validated with the app's feature extractor (synthetic); the GAN never saw the test split. "
                 "'Real check' uses real Kaggle resumes labelled by their own job title (weak labels, only roles with "
                 "real examples)."),
        "role_names": hy.ROLES,
        "seed_dataset": seed_report,
        "llm_calls": client.calls,
        "real_pattern_hints": hints,
        "training": config["training"],
        "plausibility": config["plausibility"],
        "training_history": history,
        "checkpoint_check": check,
        "quality": quality,
        "original_vs_improved": comparison,
        "augmentation": augmentation,
        "detailed_test_report": detail,
        "real_check": real_eval,
        "tstr": tstr_scores,
        "seconds_total": round(time.time() - start, 1),
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    log(f"metrics.json written ({metrics['seconds_total']:.0f}s total)")
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build, train and evaluate the hybrid GAN + LLM generator")
    parser.add_argument("--calls-per-role", type=int, default=12)
    parser.add_argument("--per-call", type=int, default=10)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--stage", choices=["all", "seed", "train"], default="all",
                        help="seed = only build the seed data · train = reuse the saved seed data (no API calls)")
    parser.add_argument("--seed-csv", help="seed CSV for --stage train (default data/gan_hybrid/llm_seed_profiles.csv)")
    parser.add_argument("--out", default=hy.HYBRID_DIR, help="output folder (default models/gan/hybrid)")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    run(args.calls_per_role, args.per_call, args.iterations, args.seed, args.quick, args.stage, args.seed_csv, args.out)


if __name__ == "__main__":
    main()
