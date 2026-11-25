"""
Comparison baselines for the GAN evaluation (training/evaluation only — not used by the app)
-------------------------------------------------------------------------------------------
CVAEBaseline         same layer sizes as models/cvae (64-32 encoder, 8-d latent, 32-64 decoder, sigmoid output),
                     retrained on the Kaggle train split so GAN and CVAE are compared on identical data.
                     Inputs are scaled to 0-1 (the sigmoid output range). The shipped models/cvae weights are
                     NOT touched: they are conditioned on the 5 RecruitGen roles, not the 24 Kaggle categories.
OversamplingBaseline random duplication of real training rows — the naive augmentation every generator must beat.

Both expose generate_array(counts, seed) like CandidateGAN, so the evaluation code treats them identically.
"""

import copy

import numpy as np
import torch
import torch.nn as nn

from models.gan.dataset import FEATURE_BOUNDS, FEATURE_NAMES
from models.gan.gan import snap_to_support

LATENT_DIM = 8


def _unit(values):
    lo = np.array([FEATURE_BOUNDS[f][0] for f in FEATURE_NAMES], dtype=np.float32)
    hi = np.array([FEATURE_BOUNDS[f][1] for f in FEATURE_NAMES], dtype=np.float32)
    return (np.asarray(values, dtype=np.float32) - lo) / (hi - lo)


def _from_unit(values):
    lo = np.array([FEATURE_BOUNDS[f][0] for f in FEATURE_NAMES], dtype=np.float32)
    hi = np.array([FEATURE_BOUNDS[f][1] for f in FEATURE_NAMES], dtype=np.float32)
    return np.clip(values, 0.0, 1.0) * (hi - lo) + lo


class _CVAE(nn.Module):
    def __init__(self, num_features, num_roles, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(num_features + num_roles, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.z_mean = nn.Linear(32, latent_dim)
        self.z_log_var = nn.Linear(32, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim + num_roles, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU(),
                                     nn.Linear(64, num_features), nn.Sigmoid())

    def decode(self, z, role):
        return self.decoder(torch.cat([z, role], dim=1))

    def forward(self, x, role):
        h = self.encoder(torch.cat([x, role], dim=1))
        mean, log_var = self.z_mean(h), self.z_log_var(h)
        z = mean + torch.exp(0.5 * log_var) * torch.randn_like(mean)
        return self.decode(z, role), mean, log_var


class CVAEBaseline:
    def __init__(self, role_names, support):
        self.role_names = list(role_names)
        self.support = support
        self.model = _CVAE(len(FEATURE_NAMES), len(self.role_names))
        self.history = []

    def _tensors(self, df):
        x = torch.tensor(_unit(df[FEATURE_NAMES].values))
        role = torch.zeros(len(df), len(self.role_names))
        role[torch.arange(len(df)), torch.tensor(df["job_role"].map(self.role_names.index).values)] = 1.0
        return x, role

    @staticmethod
    def _loss(recon, x, mean, log_var, beta):
        rec = ((x - recon) ** 2).sum(dim=1).mean()
        kl = (-0.5 * (1 + log_var - mean ** 2 - log_var.exp()).sum(dim=1)).mean()
        return rec + beta * kl

    def fit(self, train_df, val_df, epochs=150, batch_size=64, lr=1e-3, beta=1.0, patience=15, seed=42):
        """Same recipe as CVAE_Training.ipynb: reconstruction + KL, Adam 1e-3, early stopping on validation loss."""
        torch.manual_seed(seed)
        x, role = self._tensors(train_df)
        xv, rv = self._tensors(val_df)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        best, best_state, wait = float("inf"), None, 0
        for epoch in range(1, epochs + 1):
            self.model.train()
            for idx in torch.randperm(len(x)).split(batch_size):
                recon, mean, log_var = self.model(x[idx], role[idx])
                loss = self._loss(recon, x[idx], mean, log_var, beta)
                opt.zero_grad()
                loss.backward()
                opt.step()
            self.model.eval()
            with torch.no_grad():
                recon, mean, log_var = self.model(xv, rv)
                val_loss = float(self._loss(recon, xv, mean, log_var, beta))
            self.history.append({"epoch": epoch, "val_loss": val_loss})
            if val_loss < best:
                best, best_state, wait = val_loss, copy.deepcopy(self.model.state_dict()), 0
            else:
                wait += 1
                if wait >= patience:
                    break
        self.model.load_state_dict(best_state)
        self.model.eval()
        self.best_val_loss, self.epochs_run = best, len(self.history)
        return self

    def generate_array(self, counts, seed=None):
        gen = torch.Generator().manual_seed(int(seed)) if seed is not None else None
        blocks, roles = [], []
        for role_name, n in counts.items():
            n = int(n)
            if n <= 0:
                continue
            role = torch.zeros(n, len(self.role_names))
            role[:, self.role_names.index(role_name)] = 1.0
            with torch.no_grad():
                blocks.append(self.model.decode(torch.randn(n, LATENT_DIM, generator=gen), role).numpy())
            roles += [role_name] * n
        if not blocks:
            return np.zeros((0, len(FEATURE_NAMES))), []
        return snap_to_support(_from_unit(np.concatenate(blocks)), FEATURE_NAMES, self.support), roles


class OversamplingBaseline:
    def __init__(self, train_df):
        self.by_role = {r: g[FEATURE_NAMES].values for r, g in train_df.groupby("job_role")}

    def generate_array(self, counts, seed=None):
        rng = np.random.default_rng(seed)
        blocks, roles = [], []
        for role, n in counts.items():
            n = int(n)
            if n <= 0:
                continue
            pool = self.by_role[role]
            blocks.append(pool[rng.integers(0, len(pool), n)])
            roles += [role] * n
        if not blocks:
            return np.zeros((0, len(FEATURE_NAMES))), []
        return np.concatenate(blocks).astype(np.float64), roles
