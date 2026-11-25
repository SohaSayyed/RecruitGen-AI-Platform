"""Transformer V3 page — role prediction report (also provides the prediction / features sections
reused by the Resume Analyzer report)."""

from reportlab.platypus import Spacer

from . import theme as t
from .components import BarChart, MetricCards, SectionTitle, bullets, data_table, para, text_block
from .document import build_pdf, content_width

PIPELINE = [
    "**Feature extraction** — 9 numerical signals are read from the resume (experience, projects, skills scores, "
    "certifications, education).",
    "**Transformer encoding** — DistilBERT turns the resume text into a semantic embedding.",
    "**Feature fusion** — the embedding is combined with the scaled numerical features.",
    "**Role prediction** — a softmax layer scores every job role; the highest probability wins.",
]


def prediction_section(role, confidence, probabilities):
    ranked = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
    runner = ranked[1] if len(ranked) > 1 else None
    out = [
        SectionTitle("Role prediction · Transformer V3",
                     "DistilBERT text embedding fused with 9 numerical features, scored across every job role"),
        MetricCards([
            ("Predicted role", role, "Highest probability", t.PRIMARY),
            ("Confidence", f"{confidence:.1f}%", "Softmax probability", t.score_color(confidence)),
            ("Runner-up", runner[0] if runner else "—", f"{runner[1]:.1f}% probability" if runner else "", t.ACCENT_DARK),
            ("Lead margin", f"{confidence - runner[1]:.1f} pts" if runner else "—", "Over the runner-up", t.INK),
        ]),
        Spacer(1, 12),
        para("Probability per job role", "h3"),
        BarChart(ranked),
    ]
    return out


def feature_section(feature_rows, feature_scores, width, title="Candidate features"):
    """feature_rows: [(label, value, meaning)], feature_scores: {label: 0-100}."""
    rows = [(str(i), label, value, meaning) for i, (label, value, meaning) in enumerate(feature_rows, 1)]
    ranked = sorted(feature_scores.items(), key=lambda kv: kv[1], reverse=True)
    return [
        SectionTitle(title, "The 9 numerical signals extracted from the resume and used by every model"),
        data_table(["#", "Feature", "Value", "Meaning"], rows,
                   [width * 0.06, width * 0.27, width * 0.2, width * 0.47], align=["CENTER", "LEFT", "LEFT", "LEFT"]),
        Spacer(1, 12),
        para("Feature strength (normalised 0–100)", "h3"),
        BarChart(ranked, fmt="{:.0f}", highlight=False, color=t.ACCENT_DARK),
    ]


def build_transformer_pdf(result, feature_rows, feature_scores, prepared_by=""):
    role, confidence, probabilities = result["pred"]
    width = content_width()
    story = prediction_section(role, confidence, probabilities)
    story += feature_section(feature_rows, feature_scores, width)
    story += [SectionTitle("How the prediction is made"), *bullets(PIPELINE, numbered=True)]
    if result.get("text"):
        story += [SectionTitle("Resume text analysed", f"{len(result['text']):,} characters"),
                  *text_block(result["text"], 7000)]
    return build_pdf(
        story, title=role, kind="Role Prediction Report", eyebrow="Transformer V3 · Role prediction",
        subtitle=f"Predicted role {role} with {confidence:.1f}% confidence",
        meta={"Prepared by": prepared_by, "Model": "Transformer V3 (DistilBERT)"},
    )
