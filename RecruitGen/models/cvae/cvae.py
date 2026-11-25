"""
CVAE — Synthetic Candidate Generation & Role Fit
------------------------------------------------
Files used:
    vae_model.h5      full Conditional VAE (encoder + sampling + decoder)
    vae.weights.h5    trained weights (fallback if the full model cannot be read)
    scaler.json       StandardScaler statistics used during training

Role fit = 40% CVAE role-conditioned reconstruction score + 60% role relevance.
"""

import os
import json

import numpy as np
import keras
from keras import layers, ops

from models import skip_tensorflow_probe

skip_tensorflow_probe()

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "vae_model.h5")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "vae.weights.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.json")

INPUT_DIM = 9
CONDITION_DIM = 5
LATENT_DIM = 8

FEATURE_NAMES = [
    "experience_years",
    "projects_count",
    "programming_score",
    "data_analytics_score",
    "ml_ai_score",
    "cloud_score",
    "communication_score",
    "certification_count",
    "education_level",
]

ROLE_NAMES = [
    "AI/ML Engineer",
    "Cloud Engineer",
    "Data Analyst",
    "General IT",
    "Software Engineer",
]

# How much each feature matters for each role (each row sums to 1.0)
ROLE_WEIGHTS = {
    "AI/ML Engineer": {
        "experience_years": 0.08, "projects_count": 0.12, "programming_score": 0.18,
        "data_analytics_score": 0.08, "ml_ai_score": 0.28, "cloud_score": 0.06,
        "communication_score": 0.05, "certification_count": 0.05, "education_level": 0.10,
    },
    "Cloud Engineer": {
        "experience_years": 0.10, "projects_count": 0.10, "programming_score": 0.12,
        "data_analytics_score": 0.03, "ml_ai_score": 0.05, "cloud_score": 0.35,
        "communication_score": 0.07, "certification_count": 0.10, "education_level": 0.08,
    },
    "Data Analyst": {
        "experience_years": 0.08, "projects_count": 0.08, "programming_score": 0.10,
        "data_analytics_score": 0.35, "ml_ai_score": 0.05, "cloud_score": 0.03,
        "communication_score": 0.15, "certification_count": 0.06, "education_level": 0.10,
    },
    "General IT": {
        "experience_years": 0.12, "projects_count": 0.10, "programming_score": 0.15,
        "data_analytics_score": 0.08, "ml_ai_score": 0.08, "cloud_score": 0.10,
        "communication_score": 0.12, "certification_count": 0.08, "education_level": 0.17,
    },
    "Software Engineer": {
        "experience_years": 0.10, "projects_count": 0.15, "programming_score": 0.30,
        "data_analytics_score": 0.05, "ml_ai_score": 0.08, "cloud_score": 0.07,
        "communication_score": 0.07, "certification_count": 0.04, "education_level": 0.14,
    },
}

# Practical value ranges used to clip generated candidates
FEATURE_RANGES = {
    "experience_years": (0, 15), "projects_count": (0, 15), "programming_score": (0, 100),
    "data_analytics_score": (0, 100), "ml_ai_score": (0, 100), "cloud_score": (0, 100),
    "communication_score": (0, 100), "certification_count": (0, 10), "education_level": (1, 5),
}


@keras.saving.register_keras_serializable(package="RecruitGen")
class Sampling(layers.Layer):
    """Reparameterization trick: z = mean + exp(0.5 * log_var) * epsilon."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = keras.random.normal(shape=ops.shape(z_mean))
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon


def build_cvae():
    candidate_input = keras.Input(shape=(INPUT_DIM,), name="candidate_features")
    role_input = keras.Input(shape=(CONDITION_DIM,), name="target_role")

    x = layers.Concatenate(name="encoder_condition")([candidate_input, role_input])
    x = layers.Dense(64, activation="relu", name="encoder_dense_1")(x)
    x = layers.Dense(32, activation="relu", name="encoder_dense_2")(x)
    z_mean = layers.Dense(LATENT_DIM, name="z_mean")(x)
    z_log_var = layers.Dense(LATENT_DIM, name="z_log_var")(x)
    z = Sampling(name="latent_sampling")([z_mean, z_log_var])

    x = layers.Concatenate(name="decoder_condition")([z, role_input])
    x = layers.Dense(32, activation="relu", name="decoder_dense_1")(x)
    x = layers.Dense(64, activation="relu", name="decoder_dense_2")(x)
    candidate_output = layers.Dense(INPUT_DIM, activation="sigmoid", name="candidate_reconstruction")(x)

    return keras.Model([candidate_input, role_input], candidate_output, name="RecruitGen_CVAE")


class CandidateCVAE:
    """Loads the trained CVAE and exposes role-fit scoring and synthetic candidate generation."""

    def __init__(self):
        self.model = self._load_model()

        with open(SCALER_PATH, "r") as f:
            scaler = json.load(f)
        self.mean = np.array(scaler["mean"], dtype=np.float32)
        self.scale = np.array(scaler["scale"], dtype=np.float32)

        g = self.model.get_layer
        # Deterministic encoder (z_mean) for stable role-fit scores
        self.encoder = keras.Model(self.model.inputs, g("z_mean").output, name="cvae_encoder")

        z_in = keras.Input(shape=(LATENT_DIM,), name="z")
        role_in = keras.Input(shape=(CONDITION_DIM,), name="role")
        x = g("decoder_condition")([z_in, role_in])
        x = g("decoder_dense_1")(x)
        x = g("decoder_dense_2")(x)
        self.decoder = keras.Model([z_in, role_in], g("candidate_reconstruction")(x), name="cvae_decoder")

    @staticmethod
    def _load_model():
        try:
            return keras.models.load_model(MODEL_PATH, compile=False, custom_objects={"Sampling": Sampling})
        except Exception:
            model = build_cvae()
            model.load_weights(WEIGHTS_PATH)
            return model

    @property
    def num_params(self):
        return int(self.model.count_params())

    # ---------------- helpers ----------------
    def _scale(self, candidate):
        values = np.array([float(candidate[name]) for name in FEATURE_NAMES], dtype=np.float32)
        return ((values - self.mean) / self.scale).reshape(1, INPUT_DIM)

    @staticmethod
    def _role_vector(index, n=1):
        vec = np.zeros((n, CONDITION_DIM), dtype=np.float32)
        vec[:, index] = 1.0
        return vec

    @staticmethod
    def _feature_scores(candidate):
        """Convert every feature to a 0-100 score."""
        scores = {
            "experience_years": min(float(candidate["experience_years"]) / 5.0, 1.0) * 100,
            "projects_count": min(float(candidate["projects_count"]) / 5.0, 1.0) * 100,
            "certification_count": min(float(candidate["certification_count"]) / 5.0, 1.0) * 100,
            "education_level": min(float(candidate["education_level"]) / 3.0, 1.0) * 100,
        }
        for name in ["programming_score", "data_analytics_score", "ml_ai_score", "cloud_score", "communication_score"]:
            scores[name] = float(candidate[name])
        return {k: max(0.0, min(100.0, v)) for k, v in scores.items()}

    def role_relevance(self, candidate, role):
        scores = self._feature_scores(candidate)
        return max(0.0, min(100.0, sum(scores[f] * w for f, w in ROLE_WEIGHTS[role].items())))

    # ---------------- public API ----------------
    def role_fit(self, candidate):
        """Score the candidate against every role. Returns a list sorted by fit score (best first)."""
        missing = [f for f in FEATURE_NAMES if f not in candidate]
        if missing:
            raise ValueError("Missing candidate features: " + ", ".join(missing))

        x = self._scale(candidate)
        n = len(ROLE_NAMES)
        xs = np.repeat(x, n, axis=0)
        roles = np.eye(n, dtype=np.float32)
        z = self.encoder.predict([xs, roles], verbose=0)
        recon = self.decoder.predict([z, roles], verbose=0)

        results = []
        for i, role in enumerate(ROLE_NAMES):
            mse = float(np.mean(np.square(xs[i] - recon[i])))
            mae = float(np.mean(np.abs(xs[i] - recon[i])))
            cvae_score = 100.0 / (1.0 + mse)
            relevance = self.role_relevance(candidate, role)
            results.append({
                "role": role,
                "mse": mse,
                "mae": mae,
                "cvae_score": cvae_score,
                "relevance_score": relevance,
                "fit_score": 0.40 * cvae_score + 0.60 * relevance,
            })
        return sorted(results, key=lambda r: r["fit_score"], reverse=True)

    def generate(self, role, n=5, seed=None):
        """Generate n synthetic candidate profiles conditioned on a target role."""
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((n, LATENT_DIM)).astype(np.float32)
        out = self.decoder.predict([z, self._role_vector(ROLE_NAMES.index(role), n)], verbose=0)

        rows = []
        for vec in out:
            # Decoder output is in the normalised feature space -> map back to real units
            values = vec * self.scale + self.mean
            row = {}
            for name, value in zip(FEATURE_NAMES, values):
                lo, hi = FEATURE_RANGES[name]
                value = float(np.clip(value, lo, hi))
                if name in ("projects_count", "certification_count", "education_level"):
                    value = int(round(value))
                elif name == "experience_years":
                    value = round(value, 1)
                else:
                    value = round(value, 1)
                row[name] = value
            row["role_relevance"] = round(self.role_relevance(row, role), 1)
            rows.append(row)
        return rows
