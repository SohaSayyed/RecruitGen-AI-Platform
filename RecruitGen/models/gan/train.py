"""
GAN pipeline — dataset → training → checkpoint → reload test → evaluation → metrics.json
---------------------------------------------------------------------------------------
Run from the RecruitGen folder:

    python -m models.gan.train                         # download (kagglehub) or reuse the Kaggle dataset
    python -m models.gan.train --data path/to/Resume.csv
    python -m models.gan.train --quick                 # fast end-to-end test run

Writes generator.safetensors, discriminator.safetensors, config.json and metrics.json to models/gan/
(or --out), and the processed real features to data/gan/kaggle_features.csv.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

from models.gan import dataset as ds
from models.gan.baselines import CVAEBaseline, OversamplingBaseline
from models.gan.evaluation import (augmentation_experiment, detailed_comparison, generator_report, real_reference,
                                   tstr)
from models.gan.gan import MODEL_DIR, CandidateGAN, save_checkpoint
from models.gan.training import train_gan


def prepare_data(data=None, download=True, seed=42, log=print):
    csv_path = ds.find_resume_csv(data)
    if csv_path is None and download:
        log(f"Downloading {ds.KAGGLE_URL} with kagglehub ...")
        csv_path = ds.download_dataset()
    if csv_path is None:
        raise FileNotFoundError(
            f"Resume.csv not found. Download {ds.KAGGLE_URL} and extract it to data/kaggle_resume/ "
            "(or pass --data path/to/Resume.csv).")
    log(f"Dataset: {csv_path}")
    table, report = ds.build_feature_table(ds.load_raw(csv_path))
    table = ds.split_table(table, seed=seed)
    report["source_file"] = os.path.basename(csv_path)
    report["split_sizes"] = {k: int(v) for k, v in table["split"].value_counts().items()}
    ds.save_processed(table)
    log(f"Processed {report['rows']} resumes ({report['dropped_too_short']} too short, "
        f"{report['dropped_duplicate_text']} duplicates dropped) -> {ds.PROCESSED_PATH}")
    return table, report


def verify_checkpoint(model, out_dir, seed=123):
    """Reload the saved checkpoint and check it reproduces the in-memory generator exactly."""
    reloaded = CandidateGAN(out_dir)
    counts = {model.role_names[0]: 16, model.role_names[-1]: 16}
    a, _ = model.generate_array(counts, seed=seed)
    b, _ = reloaded.generate_array(counts, seed=seed)
    return {"reloaded": True, "identical_samples": bool(np.array_equal(a, b)), "params": reloaded.num_params}


def evaluate(model, train, val, test, role_names, seeds, cvae_epochs=150, log=print):
    support = model.support
    log("Training the CVAE baseline (same layer sizes as models/cvae) on the same split ...")
    cvae = CVAEBaseline(role_names, support).fit(train, val, epochs=cvae_epochs)
    log(f"CVAE baseline: best val loss {cvae.best_val_loss:.4f} after {cvae.epochs_run} epochs")
    samplers = {"GAN": model, "CVAE baseline": cvae, "Random oversampling": OversamplingBaseline(train)}

    log("Quality & diversity against the held-out test split ...")
    quality = {"Real reference": real_reference(train, test, role_names)}
    quality.update({name: generator_report(s, train, test, role_names) for name, s in samplers.items()
                    if name != "Random oversampling"})

    log("Downstream job-role classifier: real only vs augmented (same untouched test split) ...")
    augmentation = augmentation_experiment(train, val, test, samplers, role_names, seeds=seeds, log=log)
    chosen = augmentation["samplers"]["GAN"]["chosen_strategy"]
    detail = detailed_comparison(train, test, model, chosen, role_names, seed=seeds[0])

    log("TSTR (train on synthetic only, test on real) ...")
    tstr_scores = {name: tstr(train, test, s, role_names, seeds) for name, s in samplers.items()
                   if name != "Random oversampling"}
    return {"quality": quality, "augmentation": augmentation, "detailed_test_report": detail, "tstr": tstr_scores,
            "cvae_baseline": {"epochs_run": cvae.epochs_run, "best_val_loss": cvae.best_val_loss}}


def run(data=None, out_dir=MODEL_DIR, quick=False, seed=42, iterations=None, download=True, log=print):
    start = time.time()
    table, report = prepare_data(data, download, seed, log)
    train, val, test = (table[table["split"] == s].reset_index(drop=True) for s in ("train", "val", "test"))
    role_names = sorted(table["job_role"].unique())

    iterations = iterations or (300 if quick else 6000)
    log(f"Training conditional WGAN-GP for {iterations} iterations on {len(train)} rows / {len(role_names)} roles")
    model, critic, history, config = train_gan(train, val, role_names, iterations=iterations, seed=seed,
                                               eval_every=50 if quick else 100, dataset_report=report, log=log)
    save_checkpoint(model.generator, critic, config, out_dir)
    check = verify_checkpoint(model, out_dir)
    log(f"Checkpoint saved to {out_dir} · reload identical: {check['identical_samples']}")

    seeds = (0, 1) if quick else (0, 1, 2, 3, 4)
    results = evaluate(model, train, val, test, role_names, seeds, cvae_epochs=20 if quick else 150, log=log)
    metrics = build_metrics(report, role_names, config, history, check, results, quick, time.time() - start)
    write_metrics(metrics, out_dir)
    log(f"metrics.json written ({metrics['seconds_total']:.0f}s total)")
    return metrics


def build_metrics(report, role_names, config, history, check, results, quick, seconds):
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "quick_run": quick,
        "note": "All values computed by models/gan/evaluation.py on the Kaggle Resume Dataset; "
                "the test split was never used for training or model/strategy selection.",
        "dataset": report,
        "role_names": role_names,
        "training": config["training"],
        "training_history": history,
        "checkpoint_check": check,
        **results,
        "seconds_total": round(seconds, 1),
    }


def write_metrics(metrics, out_dir=MODEL_DIR):
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train and evaluate the RecruitGen GAN")
    parser.add_argument("--data", help="Resume.csv or the folder that contains it")
    parser.add_argument("--out", default=MODEL_DIR, help="output folder (default: models/gan)")
    parser.add_argument("--iterations", type=int, help="generator iterations (default 6000, quick 300)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="fast end-to-end test run")
    parser.add_argument("--no-download", action="store_true", help="never download the dataset")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    run(args.data, args.out, args.quick, args.seed, args.iterations, not args.no_download)


if __name__ == "__main__":
    main()
