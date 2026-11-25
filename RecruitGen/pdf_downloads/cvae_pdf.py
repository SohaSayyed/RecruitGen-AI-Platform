"""CVAE / VAE page — role-fit report and synthetic candidate report (the role-fit section is reused by
the Resume Analyzer report)."""

from reportlab.platypus import Spacer

from . import theme as t
from .components import (BarChart, GroupedBars, MetricCards, RankCards, SectionTitle, bullets, callout,
                         data_table, para)
from .document import build_pdf, content_width
from .transformer_pdf import feature_section

FIT_NOTE = ("**Fit score** = 40% CVAE role-conditioned reconstruction + 60% role-specific feature relevance. "
            "A low reconstruction error means the candidate looks like a typical profile for that role.")

# (key from CandidateCVAE.generate, column title, number format)
SYNTHETIC_COLUMNS = [
    ("experience_years", "Exp. (yrs)", "{:.1f}"), ("projects_count", "Projects", "{:.0f}"),
    ("programming_score", "Prog.", "{:.0f}"), ("data_analytics_score", "Data", "{:.0f}"),
    ("ml_ai_score", "ML / AI", "{:.0f}"), ("cloud_score", "Cloud", "{:.0f}"),
    ("communication_score", "Comm.", "{:.0f}"), ("certification_count", "Certs", "{:.0f}"),
    ("education_level", "Edu.", "{:.0f}"), ("role_relevance", "Relevance", "{:.1f}%"),
]


def role_fit_section(fit, width):
    rows = [(str(i), r["role"], f"{r['fit_score']:.2f}%", f"{r['cvae_score']:.2f}%", f"{r['relevance_score']:.2f}%",
             f"{r['mse']:.4f}") for i, r in enumerate(fit, 1)]
    return [
        SectionTitle("Role fit · CVAE", "Conditional VAE reconstruction for every role + role-specific feature relevance"),
        RankCards([(r["role"], r["fit_score"], f"CVAE {r['cvae_score']:.1f}% · Relevance {r['relevance_score']:.1f}%")
                   for r in fit[:3]]),
        Spacer(1, 14),
        para("Score breakdown per role", "h3"),
        GroupedBars([(r["role"], [r["fit_score"], r["cvae_score"], r["relevance_score"]]) for r in fit],
                    [("Fit score", t.PRIMARY), ("CVAE score", t.ACCENT_DARK), ("Relevance", t.LAVENDER)]),
        Spacer(1, 10),
        para("Complete role ranking", "h3"),
        data_table(["Rank", "Role", "Fit score", "CVAE score", "Relevance", "Recon. MSE"], rows,
                   [width * w for w in (0.08, 0.32, 0.15, 0.15, 0.15, 0.15)],
                   align=["CENTER", "LEFT", "RIGHT", "RIGHT", "RIGHT", "RIGHT"]),
        Spacer(1, 8),
        callout(FIT_NOTE, width, "info", "How the score is calculated"),
    ]


def build_role_fit_pdf(fit, feature_rows, feature_scores, prepared_by=""):
    width = content_width()
    best = fit[0]
    story = role_fit_section(fit, width)
    story += feature_section(feature_rows, feature_scores, width, title="Candidate profile used")
    return build_pdf(
        story, title=f"Best fit: {best['role']}", kind="Role Fit Report", eyebrow="CVAE · Role fit analysis",
        subtitle=f"{best['role']} leads with a {best['fit_score']:.1f}% fit score across {len(fit)} roles",
        meta={"Prepared by": prepared_by, "Model": "Conditional VAE"},
    )


def build_synthetic_pdf(role, rows, mean_scores, prepared_by="", seed=None):
    """rows: candidates from CandidateCVAE.generate; mean_scores: {feature label: 0-100} of the average profile."""
    width = content_width()
    n = len(rows)
    relevance = [r.get("role_relevance", 0) for r in rows]
    experience = [r.get("experience_years", 0) for r in rows]
    best = sorted(range(n), key=lambda i: relevance[i], reverse=True)[:3]
    table_rows = [[f"SYN-{i + 1:03d}"] + [fmt.format(r.get(key, 0)) for key, _, fmt in SYNTHETIC_COLUMNS]
                  for i, r in enumerate(rows)]
    story = [
        SectionTitle("Generation summary", f"Latent space sampled and decoded with the condition “{role}”"),
        MetricCards([
            ("Candidates", str(n), "Synthetic profiles", t.PRIMARY),
            ("Target role", role, "CVAE condition", t.ACCENT_DARK),
            ("Avg. relevance", f"{sum(relevance) / max(n, 1):.1f}%", "Role-specific relevance",
             t.score_color(sum(relevance) / max(n, 1))),
            ("Avg. experience", f"{sum(experience) / max(n, 1):.1f} yrs", f"Seed {seed}" if seed is not None else "",
             t.INK),
        ]),
        Spacer(1, 10),
        callout("Top candidates by relevance: " + ", ".join(f"**SYN-{i + 1:03d}** ({relevance[i]:.1f}%)" for i in best),
                width, "good", "Strongest profiles"),
        SectionTitle("Average generated profile", "Mean of every generated candidate, normalised 0–100"),
        BarChart(sorted(mean_scores.items(), key=lambda kv: kv[1], reverse=True), fmt="{:.0f}", highlight=False,
                 color=t.PRIMARY),
        SectionTitle("Synthetic candidates", "Use these profiles for testing, data augmentation or demos"),
        data_table(["Candidate"] + [title for _, title, _ in SYNTHETIC_COLUMNS], table_rows,
                   [width * 0.12] + [width * 0.088] * 10,
                   align=["LEFT"] + ["RIGHT"] * len(SYNTHETIC_COLUMNS), font_size=7.6),
        Spacer(1, 6),
        para("Exp. = experience · Prog. = programming · Comm. = communication · Edu. = education level "
             "(1 Diploma, 2 Bachelor's, 3 Master's, 5 PhD). Scores are 0–100.", "small"),
        SectionTitle("Notes"),
        *bullets([
            "Profiles are **synthetic**: they are sampled from the CVAE latent space, not real people.",
            "Changing the random seed produces a different but reproducible batch.",
            "Relevance is computed with the same role-specific weights used for role-fit scoring.",
        ]),
    ]
    return build_pdf(
        story, title=f"{n} synthetic candidates · {role}", kind="Synthetic Candidates",
        eyebrow="CVAE · Synthetic candidate generation",
        subtitle=f"Conditioned on {role}" + (f" · random seed {seed}" if seed is not None else ""),
        meta={"Prepared by": prepared_by, "Model": "Conditional VAE"},
    )
