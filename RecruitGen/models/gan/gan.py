"""
GAN — Synthetic Candidate Profiles for Dataset Augmentation
-----------------------------------------------------------
Files used:
    generator.safetensors       trained generator weights
    discriminator.safetensors   trained critic weights (only needed to resume training)
    config.json                 architecture, job roles, feature bounds, value support, training info
    metrics.json                evaluation report written by models/gan/train.py (shown in the app)

Conditional WGAN-GP on the 9 RecruitGen features, conditioned on the job role (Kaggle resume category):

noise z + role one-hot -> Generator -> 9 features in [-1, 1] -> real units -> snapped to observed values
features + role one-hot -> Critic   -> realness score (Wasserstein estimate)

Every generated row is synthetic and is labelled as such by the callers.
"""

import os
import json

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
GENERATOR_PATH = os.path.join(MODEL_DIR, "generator.safetensors")
CRITIC_PATH = os.path.join(MODEL_DIR, "discriminator.safetensors")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

NOISE_DIM = 32
HIDDEN_DIM = 128


class Generator(nn.Module):
    def __init__(self, num_features=9, num_roles=24, noise_dim=NOISE_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim + num_roles, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, num_features), nn.Tanh(),
        )

    def forward(self, z, role):
        return self.net(torch.cat([z, role], dim=1))


class Critic(nn.Module):
    """WGAN-GP critic: no sigmoid and no batch norm (the gradient penalty is computed per sample)."""

    def __init__(self, num_features=9, num_roles=24, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features + num_roles, hidden_dim * 2), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, role):
        return self.net(torch.cat([x, role], dim=1))


# ---------------- feature scaling ----------------
def scale_features(values, feature_names, bounds):
    """Real units -> [-1, 1] using the fixed feature bounds."""
    lo = np.array([bounds[f][0] for f in feature_names], dtype=np.float32)
    hi = np.array([bounds[f][1] for f in feature_names], dtype=np.float32)
    return ((np.asarray(values, dtype=np.float32) - lo) / (hi - lo)) * 2.0 - 1.0


def unscale_features(values, feature_names, bounds):
    lo = np.array([bounds[f][0] for f in feature_names], dtype=np.float32)
    hi = np.array([bounds[f][1] for f in feature_names], dtype=np.float32)
    return (np.clip(values, -1.0, 1.0) + 1.0) / 2.0 * (hi - lo) + lo


def snap_to_support(values, feature_names, support):
    """Map every generated value to the nearest value observed in the real training data.
    The extractor produces discrete values (scores in steps of 15, counts, education levels), so a raw
    continuous output like 37.2 would never be produced from a real resume."""
    out = np.array(values, dtype=np.float64, copy=True)
    for j, name in enumerate(feature_names):
        allowed = np.asarray(sorted(support[name]), dtype=np.float64)
        if len(allowed) == 1:
            out[:, j] = allowed[0]
            continue
        pos = np.clip(np.searchsorted(allowed, out[:, j]), 1, len(allowed) - 1)
        left, right = allowed[pos - 1], allowed[pos]
        out[:, j] = np.where(np.abs(out[:, j] - left) <= np.abs(right - out[:, j]), left, right)
    return out


# ---------------- checkpoint io ----------------
def save_checkpoint(generator, critic, config, out_dir=MODEL_DIR):
    os.makedirs(out_dir, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in generator.state_dict().items()},
              os.path.join(out_dir, "generator.safetensors"), metadata={"format": "pt", "model": "RecruitGenGAN"})
    if critic is not None:
        save_file({k: v.detach().cpu().contiguous() for k, v in critic.state_dict().items()},
                  os.path.join(out_dir, "discriminator.safetensors"), metadata={"format": "pt"})
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


def checkpoint_available(model_dir=MODEL_DIR):
    return all(os.path.exists(os.path.join(model_dir, n)) for n in ("generator.safetensors", "config.json"))


def load_metrics(model_dir=MODEL_DIR):
    path = os.path.join(model_dir, "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


class CandidateGAN:
    """Loads the trained generator and produces synthetic candidate profiles for a job role."""

    def __init__(self, model_dir=MODEL_DIR, generator=None, config=None):
        if config is None:
            if not checkpoint_available(model_dir):
                raise FileNotFoundError(
                    "GAN checkpoint not found in models/gan/ — train it with GAN_Training.ipynb "
                    "or `python -m models.gan.train`.")
            with open(os.path.join(model_dir, "config.json"), "r") as f:
                config = json.load(f)
        self.config = config
        self.feature_names = config["feature_names"]
        self.role_names = config["role_names"]
        self.bounds = {k: tuple(v) for k, v in config["feature_bounds"].items()}
        self.support = config["feature_support"]
        self.integer_features = set(config["integer_features"])
        self.noise_dim = config["noise_dim"]

        if generator is None:
            generator = Generator(len(self.feature_names), len(self.role_names), self.noise_dim, config["hidden_dim"])
            generator.load_state_dict(load_file(os.path.join(model_dir, "generator.safetensors")), strict=True)
        self.generator = generator.cpu().eval()

    @property
    def num_params(self):
        return sum(p.numel() for p in self.generator.parameters())

    @property
    def training_info(self):
        return self.config.get("training", {})

    def _raw(self, role_index, n, generator, noise_scale=1.0):
        z = torch.randn(n, self.noise_dim, generator=generator) * float(noise_scale)
        role = torch.zeros(n, len(self.role_names))
        role[:, role_index] = 1.0
        with torch.no_grad():
            return self.generator(z, role).numpy()

    def generate_array(self, counts, seed=None, noise_scale=1.0):
        """counts: {role: n}. Returns (features [N, 9] in real units, role list).
        noise_scale < 1 gives more typical profiles, > 1 more diverse ones."""
        gen = torch.Generator().manual_seed(int(seed)) if seed is not None else None
        blocks, roles = [], []
        for role, n in counts.items():
            n = int(n)
            if n <= 0:
                continue
            if role not in self.role_names:
                raise ValueError(f"Unknown job role: {role}")
            blocks.append(self._raw(self.role_names.index(role), n, gen, noise_scale))
            roles += [role] * n
        if not blocks:
            return np.zeros((0, len(self.feature_names))), []
        values = unscale_features(np.concatenate(blocks), self.feature_names, self.bounds)
        return snap_to_support(values, self.feature_names, self.support), roles

    def generate(self, role, n=10, seed=None):
        """n synthetic profiles for one job role, as a list of dicts (9 features + job_role)."""
        return self.generate_many({role: n}, seed)

    def generate_many(self, counts, seed=None, noise_scale=1.0):
        values, roles = self.generate_array(counts, seed, noise_scale)
        rows = []
        for vec, role in zip(values, roles):
            row = {}
            for name, value in zip(self.feature_names, vec):
                row[name] = int(round(value)) if name in self.integer_features else round(float(value), 1)
            row["job_role"] = role
            rows.append(row)
        return rows
