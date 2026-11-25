"""Resume Analyzer page — the complete end-to-end screening report (all three models + AI insights)."""

import os

from reportlab.platypus import CondPageBreak, Spacer

from . import theme as t
from .assistant_pdf import insights_section
from .components import MetricCards, ScoreRings, SectionTitle, callout, image_grid, text_block
from .cvae_pdf import role_fit_section
from .document import build_pdf, content_width
from .transformer_pdf import feature_section, prediction_section


def _score(value):
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def build_analysis_pdf(analysis, feature_rows, feature_scores, prepared_by=""):
    width = content_width()
    pred, fit = analysis["prediction"], analysis["role_fit"]
    ins = analysis.get("insights") or {}
    best = fit[0]
    name = ins.get("candidate_name") or os.path.splitext(analysis.get("file_name") or "")[0] or "Candidate"
    overall, ats = _score(ins.get("overall_score")), _score(ins.get("ats_score"))

    rings = [(pred["confidence"], "Prediction confidence", f"{pred['confidence']:.0f}%", t.score_color(pred["confidence"])),
             (best["fit_score"], "Best role fit", f"{best['fit_score']:.0f}%", t.score_color(best["fit_score"]))]
    if overall is not None:
        rings.append((overall, "Resume score", str(overall), t.score_color(overall)))
    if ats is not None:
        rings.append((ats, "ATS score", str(ats), t.score_color(ats)))

    story = [
        SectionTitle("Executive summary", "Results from the full RecruitGen pipeline", space_before=0),
        MetricCards([
            ("Predicted role", pred["role"], "Transformer V3", t.PRIMARY),
            ("Best role fit", best["role"], f"CVAE fit score {best['fit_score']:.1f}%", t.ACCENT_DARK),
            ("Candidate", name, ins.get("headline") or "AI insights", t.INK),
        ]),
        Spacer(1, 12),
        ScoreRings(rings),
    ]
    if ins.get("summary"):
        story += [Spacer(1, 10), callout(ins["summary"], width, "info", "Professional summary")]
    story += [CondPageBreak(200)] + prediction_section(pred["role"], pred["confidence"], pred["probabilities"])
    story += [CondPageBreak(260)] + role_fit_section(fit, width)
    story += [CondPageBreak(260)] + feature_section(feature_rows, feature_scores, width)
    if ins:
        story += [CondPageBreak(260)] + insights_section(ins, width)
    else:
        story += [SectionTitle("AI resume insights"),
                  callout("AI insights were not available when this resume was analysed.", width, "warn")]
    if analysis.get("denoised"):
        story += [CondPageBreak(330), SectionTitle("Autoencoder · image cleaning", "Original upload vs denoised page"),
                  image_grid([(analysis["originals"][0], "Original resume"),
                              (analysis["denoised"][0], "Denoised by Autoencoder")], width, max_h=300)]
    if analysis.get("text"):
        story += [CondPageBreak(160), SectionTitle("Appendix · extracted resume text",
                                                   f"{len(analysis['text']):,} characters read from the resume"),
                  *text_block(analysis["text"], 9000)]
    return build_pdf(
        story, title=name, kind="Resume Screening Report", eyebrow="Resume screening report",
        subtitle=f"{pred['role']} · {pred['confidence']:.1f}% confidence · best fit {best['role']}",
        meta={"Prepared by": prepared_by, "Source": analysis.get("file_name", "")},
    )
