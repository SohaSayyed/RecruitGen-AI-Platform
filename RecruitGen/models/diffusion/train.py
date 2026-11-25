"""
Diffusion training — run from the RecruitGen folder
---------------------------------------------------
    python -m models.diffusion.train                  # CPU-friendly default (~30 min on 8 cores)
    python -m models.diffusion.train --steps 20000 --batch 32 --base 48   # full recipe (GPU recommended)
    python -m models.diffusion.train --quick          # smoke test
    python -m models.diffusion.train --resume         # continue from data/diffusion/train_state.pt

Writes models/diffusion/diffusion_unet.safetensors + config.json (EMA weights of the checkpoint with the lowest
validation L1 after DDIM sampling). Never touches the Autoencoder files.
"""

import argparse
import copy
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch

from models.diffusion import data as dd
from models.diffusion.diffusion import MODEL_DIR, DiffusionSchedule, ResumeDiffusionEnhancer, get_device, save_model
from models.diffusion.unet import ConditionalUNet

STATE_PATH = os.path.join(dd.DATA_DIR, "train_state.pt")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def validation_l1(model, schedule, val_batch, steps, device):
    deg, clean = (torch.from_numpy(a).to(device) for a in val_batch)
    noise = torch.randn(clean.shape, generator=torch.Generator().manual_seed(0)).to(device)
    model.eval()
    pred = schedule.ddim(model, deg, noise, steps)
    model.train()
    return float((pred - clean).abs().mean()), float(((pred - clean).abs() * (clean < 0).float()).sum()
                                                    / (clean < 0).float().sum().clamp_min(1))


def train(steps=1800, batch=16, patch=96, base=32, lr=2e-4, ema_decay=0.999, sample_steps=4, eval_every=200,
          seed=42, resume=False, device=None, out_dir=MODEL_DIR, log=print):
    set_seed(seed)
    device = get_device(device)
    if device.type == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() or 1))
    splits = dd.prepare_pages(seed=seed, log=log)
    train_pages = [dd.load_page(p) for p in splits["train"]]
    val_pages = [dd.load_page(p) for p in splits["val"]]
    sampler = dd.PatchSampler(train_pages, patch=patch, seed=seed)
    val_batch = dd.PatchSampler(val_pages, patch=128, seed=seed + 1).sample(48)   # fixed validation patches

    model = ConditionalUNet(base=base).to(device)
    ema = copy.deepcopy(model).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    schedule = DiffusionSchedule(1000)
    start_step, history, best = 0, [], {"val_l1": float("inf"), "step": 0, "state": None}
    if resume and os.path.exists(STATE_PATH):
        state = torch.load(STATE_PATH, map_location=device, weights_only=False)
        model.load_state_dict(state["model"]); ema.load_state_dict(state["ema"]); opt.load_state_dict(state["opt"])
        start_step, history, best = state["step"], state["history"], state["best"]
        sampler.rng = np.random.default_rng(seed + start_step)
        log(f"Resumed from step {start_step}")
    log(f"Training on {device.type}: {sum(p.numel() for p in model.parameters()):,} params · {len(train_pages)} pages · "
        f"{steps} steps × batch {batch} × {patch}px")

    t0, run_loss = time.time(), []
    for step in range(start_step + 1, steps + 1):
        warm = min(1.0, step / 200)
        for g in opt.param_groups:
            g["lr"] = lr * warm * (0.5 * (1 + np.cos(np.pi * step / steps)) * 0.9 + 0.1)
        deg, clean = (torch.from_numpy(a).to(device) for a in sampler.sample(batch))
        loss = schedule.loss(model, clean, deg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            d = min(ema_decay, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(d).add_(pm.detach(), alpha=1 - d)
        run_loss.append(loss.item())
        if step % eval_every == 0 or step == steps:
            val, val_ink = validation_l1(ema, schedule, val_batch, sample_steps, device)
            history.append({"step": step, "train_loss": float(np.mean(run_loss)), "val_l1": val, "val_ink_l1": val_ink,
                            "seconds": round(time.time() - t0, 1)})
            run_loss = []
            if val < best["val_l1"]:
                best = {"val_l1": val, "step": step, "state": copy.deepcopy(ema.state_dict())}
            log(f"step {step:5d} | train {history[-1]['train_loss']:.4f} | val L1 {val:.4f} (ink {val_ink:.4f}) | "
                f"best {best['val_l1']:.4f} @ {best['step']} | {time.time() - t0:.0f}s")
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step,
                        "history": history, "best": best}, STATE_PATH)

    ema.load_state_dict(best["state"])
    config = build_config(base, sample_steps, device.type, {
        "train_pages": len(train_pages), "val_pages": len(val_pages), "steps": steps, "batch": batch, "patch": patch,
        "learning_rate": lr, "ema_decay": ema_decay, "seed": seed, "device": device.type, "best_step": best["step"],
        "best_val_l1": best["val_l1"], "seconds": round(time.time() - t0, 1), "history": history})
    save_model(ema.cpu(), config, out_dir)
    reloaded = ResumeDiffusionEnhancer(out_dir, device="cpu")
    same = all(torch.equal(a, b) for a, b in zip(reloaded.model.state_dict().values(), ema.state_dict().values()))
    log(f"Saved {out_dir} · best step {best['step']} · reload identical: {same}")
    return config


def build_config(base, sample_steps, device_type, training):
    return {
        "model_name": "ResumeDiffusionEnhancer",
        "architecture": {"type": "ConditionalUNet", "base": base, "mults": [1, 2, 4], "in_channels": 2},
        "schedule": {"T": 1000, "type": "cosine", "prediction": "x0", "loss": "L1, ink pixels weighted x5"},
        "sampling": {"sampler": "DDIM (eta=0)", "steps": sample_steps, "tile": 256, "overlap": 32,
                     "batch": 16 if device_type == "cpu" else 64, "blank_tile_steps": 1},
        "training": {"dataset": "Kaggle Resume Dataset PDFs (first page, 150 dpi) + controlled degradations",
                     "trained_at": datetime.now().isoformat(timespec="seconds"), **training},
    }


def export_from_state(state_path=STATE_PATH, out_dir=MODEL_DIR, total_steps=1800, base=32, sample_steps=4, log=print):
    """Export the best EMA weights saved so far by a (running) training job as a PROVISIONAL checkpoint, so the app
    can use the model before training finishes. The final training run overwrites it."""
    import shutil
    import tempfile

    tmp = os.path.join(tempfile.mkdtemp(), "state.pt")
    shutil.copy(state_path, tmp)                     # read a copy: the trainer may rewrite the file at any time
    state = torch.load(tmp, map_location="cpu", weights_only=False)
    model = ConditionalUNet(base=base)
    model.load_state_dict(state["best"]["state"] or state["ema"])
    config = build_config(base, sample_steps, "cpu", {
        "provisional": True, "steps": total_steps, "steps_done": state["step"], "best_step": state["best"]["step"],
        "best_val_l1": state["best"]["val_l1"], "history": state["history"],
        "seconds": state["history"][-1]["seconds"] if state["history"] else 0})
    save_model(model, config, out_dir)
    if log:
        log(f"Provisional checkpoint exported: step {state['step']} (best {state['best']['step']}) -> {out_dir}")
    return config


def main(argv=None):
    p = argparse.ArgumentParser(description="Train the resume-image diffusion model")
    p.add_argument("--steps", type=int, default=1800)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--patch", type=int, default=96)
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--sample-steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", help="cpu / cuda / mps (default: auto)")
    p.add_argument("--out", default=MODEL_DIR)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--export-provisional", action="store_true",
                   help="export the best weights of a running/interrupted training job and exit")
    a = p.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    if a.export_provisional:
        export_from_state(out_dir=a.out, total_steps=a.steps, base=a.base, sample_steps=a.sample_steps)
        return
    if a.quick:
        a.steps, a.batch = 40, 4
    train(a.steps, a.batch, a.patch, a.base, a.lr, sample_steps=a.sample_steps, eval_every=20 if a.quick else 200,
          seed=a.seed, resume=a.resume, device=a.device, out_dir=a.out)


if __name__ == "__main__":
    main()
