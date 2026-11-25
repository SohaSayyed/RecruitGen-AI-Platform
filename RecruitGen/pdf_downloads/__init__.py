"""
RecruitGen PDF downloads
========================
Every PDF the app offers is built here with ReportLab, using one shared design (branded cover header,
running header/footer, page numbers, KPI cards, charts and styled tables).

    theme.py             colours, embedded fonts, text cleaning, paragraph styles
    document.py          page layout: cover header, running header/footer, "Page X of Y"
    components.py        building blocks: cards, score rings, charts, chips, tables, images, Markdown
    analyzer_pdf.py      Resume Analyzer  -> full screening report
    autoencoder_pdf.py   Autoencoder      -> denoising report + clean resume PDF
    cvae_pdf.py          CVAE / VAE       -> role-fit report + synthetic candidates
    transformer_pdf.py   Transformer V3   -> role prediction report
    assistant_pdf.py     AI Assistant     -> insights, job match, interview kit, chat transcript
    admin_pdf.py         History / Admin  -> screening history, activity, sessions, users, full report

PDFs are generated in memory and streamed to the browser — nothing is written to disk.
"""

from .admin_pdf import (build_activity_pdf, build_admin_report_pdf, build_history_pdf, build_sessions_pdf,
                        build_users_pdf)
from .analyzer_pdf import build_analysis_pdf
from .assistant_pdf import build_chat_pdf, build_insights_pdf, build_interview_pdf, build_job_match_pdf
from .autoencoder_pdf import build_clean_resume_pdf, build_denoise_report_pdf
from .cvae_pdf import build_role_fit_pdf, build_synthetic_pdf
from .document import pdf_filename
from .transformer_pdf import build_transformer_pdf

__all__ = [
    "build_activity_pdf", "build_admin_report_pdf", "build_analysis_pdf", "build_chat_pdf",
    "build_clean_resume_pdf", "build_denoise_report_pdf", "build_history_pdf", "build_insights_pdf",
    "build_interview_pdf", "build_job_match_pdf", "build_role_fit_pdf", "build_sessions_pdf",
    "build_synthetic_pdf", "build_transformer_pdf", "build_users_pdf", "pdf_filename",
]
