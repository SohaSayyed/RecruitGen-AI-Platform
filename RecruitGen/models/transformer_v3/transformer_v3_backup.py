"""
Transformer V3 — Intelligent Resume Screening
---------------------------------------------
Files used:
    model.safetensors   trained weights (DistilBERT encoder + feature network + fusion classifier)
    config.json         architecture, role names and the fixed 9-feature order
    scaler.json         StandardScaler statistics for the 9 numerical features
    tokenizer/          DistilBERT WordPiece tokenizer

Resume text  -> DistilBERT [CLS] embedding -> text projection ─┐
9 features   -> feature network ───────────────────────────────┴-> fusion -> role probabilities
"""

import os
import json

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import DistilBertConfig, DistilBertModel, PreTrainedTokenizerFast

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(MODEL_DIR, "model.safetensors")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.json")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer", "tokenizer.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RecruitGenTransformerV3(nn.Module):
    def __init__(self, num_features=9, hidden_dim=128, num_roles=5):
        super().__init__()
        # Architecture only — every weight is loaded from model.safetensors (no internet needed)
        self.bert = DistilBertModel(DistilBertConfig())
        bert_hidden = self.bert.config.hidden_size

        self.text_projection = nn.Sequential(
            nn.Linear(bert_hidden, hidden_dim), nn.ReLU(), nn.Dropout(0.2)
        )
        self.feature_network = nn.Sequential(
            nn.Linear(num_features, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.2),
            nn.Linear(64, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(0.2)
        )
        self.classifier = nn.Linear(hidden_dim, num_roles)

    def forward(self, input_ids, attention_mask, numerical_features):
        cls = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        fused = torch.cat([self.text_projection(cls), self.feature_network(numerical_features)], dim=1)
        return self.classifier(self.fusion(fused))


class ResumeScreener:
    def __init__(self):
        with open(CONFIG_PATH, "r") as f:
            self.config = json.load(f)
        with open(SCALER_PATH, "r") as f:
            scaler = json.load(f)
        self.mean = np.array(scaler["mean"], dtype=np.float32)
        self.scale = np.array(scaler["scale"], dtype=np.float32)

        self.role_names = self.config["role_names"]
        self.feature_order = self.config["feature_order"]
        self.max_length = self.config["max_length"]

        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=TOKENIZER_PATH,
            unk_token="[UNK]", sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]", mask_token="[MASK]",
        )

        self.model = RecruitGenTransformerV3(
            num_features=self.config["num_features"],
            hidden_dim=self.config["hidden_dim"],
            num_roles=self.config["num_roles"],
        )
        state_dict = {k: v.float() if v.is_floating_point() else v for k, v in load_file(WEIGHTS_PATH).items()}
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(DEVICE).eval()

    @property
    def num_params(self):
        return sum(p.numel() for p in self.model.parameters())

    def predict(self, resume_text, features):
        """
        features: dict with the 9 features (or a list in feature_order).
        Returns (predicted_role, confidence_percent, {role: probability_percent}).
        """
        if not isinstance(resume_text, str) or not resume_text.strip():
            raise ValueError("Resume text cannot be empty.")
        if isinstance(features, dict):
            features = [features[name] for name in self.feature_order]
        if len(features) != 9:
            raise ValueError("Transformer V3 requires exactly 9 numerical features.")

        scaled = (np.array(features, dtype=np.float32) - self.mean) / self.scale
        numerical = torch.tensor(scaled[None, :], dtype=torch.float32, device=DEVICE)

        encoded = self.tokenizer(
            resume_text, padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        with torch.no_grad():
            logits = self.model(
                input_ids=encoded["input_ids"].to(DEVICE),
                attention_mask=encoded["attention_mask"].to(DEVICE),
                numerical_features=numerical,
            )
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        idx = int(np.argmax(probs))
        probabilities = {role: float(p) * 100 for role, p in zip(self.role_names, probs)}
        return self.role_names[idx], float(probs[idx]) * 100, probabilities
