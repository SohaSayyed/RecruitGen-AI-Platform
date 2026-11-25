"""
GAN training — conditional WGAN-GP on the 9 features
----------------------------------------------------
Critic loss    = D(fake) - D(real) + λ · gradient penalty
Generator loss = -D(fake)

Model selection uses the validation split only: every `eval_every` iterations the generator samples a batch
with the validation role mix, and the checkpoint with the lowest mean per-feature Wasserstein distance to the
validation data is kept. The test split is never touched here.
"""

import copy
import time
from datetime import datetime

import numpy as np
import torch

from models.gan.dataset import FEATURE_BOUNDS, FEATURE_NAMES, INTEGER_FEATURES, KAGGLE_URL
from models.gan.gan import HIDDEN_DIM, NOISE_DIM, CandidateGAN, Critic, Generator, scale_features


def _one_hot(indices, n):
    out = torch.zeros(len(indices), n)
    out[torch.arange(len(indices)), torch.as_tensor(indices)] = 1.0
    return out


def gradient_penalty(critic, real, fake, role):
    alpha = torch.rand(real.size(0), 1)
    mixed = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    grads = torch.autograd.grad(critic(mixed, role).sum(), mixed, create_graph=True)[0]
    return ((grads.norm(2, dim=1) - 1.0) ** 2).mean()


def build_config(train_df, role_names, hparams=None, dataset_report=None):
    support = {f: sorted({round(float(v), 4) for v in train_df[f]}) for f in FEATURE_NAMES}
    return {
        "model_name": "RecruitGenConditionalWGANGP",
        "feature_names": FEATURE_NAMES,
        "role_names": list(role_names),
        "feature_bounds": {k: list(v) for k, v in FEATURE_BOUNDS.items()},
        "feature_support": support,
        "integer_features": list(INTEGER_FEATURES),
        "noise_dim": NOISE_DIM,
        "hidden_dim": HIDDEN_DIM,
        "training": {
            "dataset": "Kaggle Resume Dataset (snehaanbhawal/resume-dataset)",
            "dataset_url": KAGGLE_URL,
            "train_rows": int(len(train_df)),
            "train_role_counts": {k: int(v) for k, v in train_df["job_role"].value_counts().sort_index().items()},
            "dataset_report": dataset_report or {},
            **(hparams or {}),
        },
    }


def validation_distance(model, val_df, seed=0):
    """Mean per-feature 1-D Wasserstein distance (features scaled to 0-1) between generated and validation rows."""
    from scipy.stats import wasserstein_distance

    counts = val_df["job_role"].value_counts().to_dict()
    synthetic, _ = model.generate_array(counts, seed=seed)
    real = val_df[FEATURE_NAMES].values
    dists = []
    for j, name in enumerate(FEATURE_NAMES):
        lo, hi = FEATURE_BOUNDS[name]
        dists.append(wasserstein_distance((real[:, j] - lo) / (hi - lo), (synthetic[:, j] - lo) / (hi - lo)))
    return float(np.mean(dists))


def train_gan(train_df, val_df, role_names, iterations=6000, batch_size=64, n_critic=5, gp_weight=10.0,
              lr=2e-4, seed=42, eval_every=100, log_every=250, dataset_report=None, log=print):
    """Train the conditional WGAN-GP. Returns (CandidateGAN with the best generator, critic, history, config)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    start = time.time()

    role_index = {r: i for i, r in enumerate(role_names)}
    X = torch.tensor(scale_features(train_df[FEATURE_NAMES].values, FEATURE_NAMES, FEATURE_BOUNDS))
    R = _one_hot(train_df["job_role"].map(role_index).values, len(role_names))
    n = len(X)

    G = Generator(len(FEATURE_NAMES), len(role_names), NOISE_DIM, HIDDEN_DIM)
    D = Critic(len(FEATURE_NAMES), len(role_names), HIDDEN_DIM)
    opt_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.9))

    hparams = {"iterations": iterations, "batch_size": batch_size, "n_critic": n_critic, "gp_weight": gp_weight,
               "learning_rate": lr, "seed": seed, "selection": "lowest validation Wasserstein distance"}
    config = build_config(train_df, role_names, hparams, dataset_report)

    history, best = [], {"distance": float("inf"), "iteration": 0, "state": None}
    for it in range(1, iterations + 1):
        G.train()
        for _ in range(n_critic):
            idx = torch.randint(0, n, (batch_size,))
            real, role = X[idx], R[idx]
            fake = G(torch.randn(batch_size, NOISE_DIM), role).detach()
            d_real, d_fake = D(real, role).mean(), D(fake, role).mean()
            gp = gradient_penalty(D, real, fake, role)
            loss_d = d_fake - d_real + gp_weight * gp
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

        role = R[torch.randint(0, n, (batch_size,))]
        loss_g = -D(G(torch.randn(batch_size, NOISE_DIM), role), role).mean()
        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()

        if it % eval_every == 0 or it == iterations:
            G.eval()
            dist = validation_distance(CandidateGAN(generator=G, config=config), val_df, seed=seed)
            entry = {"iteration": it, "critic_loss": loss_d.item(), "generator_loss": loss_g.item(),
                     "wasserstein_estimate": (d_real - d_fake).item(), "gradient_penalty": gp.item(),
                     "val_distance": dist}
            history.append(entry)
            if dist < best["distance"]:
                best = {"distance": dist, "iteration": it, "state": copy.deepcopy(G.state_dict())}
            if log and (it % log_every == 0 or it == iterations):
                log(f"iter {it:5d} | critic {entry['critic_loss']:8.4f} | generator {entry['generator_loss']:8.4f} | "
                    f"W-est {entry['wasserstein_estimate']:7.4f} | val dist {dist:.4f} "
                    f"(best {best['distance']:.4f} @ {best['iteration']})")

    G.load_state_dict(best["state"])
    G.eval()
    config["training"].update({
        "best_iteration": best["iteration"],
        "best_val_distance": best["distance"],
        "seconds": round(time.time() - start, 1),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    })
    return CandidateGAN(generator=G, config=config), D, history, config
