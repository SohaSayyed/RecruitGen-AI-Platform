
"""
Transformer V3 — Intelligent Resume Screening
------------------------------------------------
Files used:
    model.safetensors   Trained weights hosted on Hugging Face
    config.json         Architecture and role configuration
    scaler.json         StandardScaler statistics
    tokenizer/          DistilBERT tokenizer

Resume text  -> DistilBERT [CLS] embedding -> text projection ─┐
9 features   -> feature network ───────────────────────────────┴-> fusion -> role probabilities
"""

import os
import json

import numpy as np
import torch
import torch.nn as nn

from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from transformers import (
    DistilBertConfig,
    DistilBertModel,
    PreTrainedTokenizerFast,
)


MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

HF_REPO_ID = "SohaSayyed/RecruitGen-Transformer-V3"
HF_MODEL_FILENAME = "model.safetensors"

CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.json")
TOKENIZER_PATH = os.path.join(
    MODEL_DIR, "tokenizer", "tokenizer.json"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class RecruitGenTransformerV3(nn.Module):

    def __init__(self, num_features=9, hidden_dim=128, num_roles=5):
        super().__init__()

        # Initialize DistilBERT architecture.
        # Trained weights are loaded from Hugging Face below.
        self.bert = DistilBertModel(DistilBertConfig())
        bert_hidden = self.bert.config.hidden_size

        self.text_projection = nn.Sequential(
            nn.Linear(bert_hidden, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.feature_network = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.classifier = nn.Linear(hidden_dim, num_roles)

    def forward(self, input_ids, attention_mask, numerical_features):

        cls = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state[:, 0]

        fused = torch.cat(
            [
                self.text_projection(cls),
                self.feature_network(numerical_features),
            ],
            dim=1,
        )

        return self.classifier(self.fusion(fused))


class ResumeScreener:

    def __init__(self):

        # Load model configuration.
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        # Load feature scaler.
        with open(SCALER_PATH, "r", encoding="utf-8") as f:
            scaler = json.load(f)

        self.mean = np.array(scaler["mean"], dtype=np.float32)
        self.scale = np.array(scaler["scale"], dtype=np.float32)

        self.role_names = self.config["role_names"]
        self.feature_order = self.config["feature_order"]
        self.max_length = self.config["max_length"]

        # Load tokenizer.
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=TOKENIZER_PATH,
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]",
        )

        # Initialize model architecture.
        self.model = RecruitGenTransformerV3(
            num_features=self.config["num_features"],
            hidden_dim=self.config["hidden_dim"],
            num_roles=self.config["num_roles"],
        )

        # Download trained weights from Hugging Face.
        weights_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_MODEL_FILENAME,
        )

        # Load trained weights.
        state_dict = {
            key: value.float() if value.is_floating_point() else value
            for key, value in load_file(weights_path).items()
        }

        self.model.load_state_dict(state_dict, strict=True)

        # Move model to CPU/GPU and set evaluation mode.
        self.model.to(DEVICE)
        self.model.eval()

    @property
    def num_params(self):
        return sum(
            parameter.numel()
            for parameter in self.model.parameters()
        )

    def predict(self, resume_text, features):
        """
        features: Dictionary containing 9 features,
                  or a list in feature_order.

        Returns:
            predicted_role,
            confidence_percent,
            role_probability_dictionary
        """

        if not isinstance(resume_text, str) or not resume_text.strip():
            raise ValueError("Resume text cannot be empty.")

        if isinstance(features, dict):
            features = [
                features[name]
                for name in self.feature_order
            ]

        if len(features) != 9:
            raise ValueError(
                "Transformer V3 requires exactly 9 numerical features."
            )

        # Scale numerical features.
        scaled = (
            np.array(features, dtype=np.float32) - self.mean
        ) / self.scale

        numerical = torch.tensor(
            scaled[None, :],
            dtype=torch.float32,
            device=DEVICE,
        )

        # Tokenize resume text.
        encoded = self.tokenizer(
            resume_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Generate predictions.
        with torch.no_grad():

            logits = self.model(
                input_ids=encoded["input_ids"].to(DEVICE),
                attention_mask=encoded["attention_mask"].to(DEVICE),
                numerical_features=numerical,
            )

            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        # Get predicted role and confidence.
        idx = int(np.argmax(probs))

        probabilities = {
            role: float(probability) * 100
            for role, probability in zip(self.role_names, probs)
        }

        return (
            self.role_names[idx],
            float(probs[idx]) * 100,
            probabilities,
        )
