"""AI Assistant page — resume insights, job match, printable interview kit and chat transcript
(the insights section is reused by the Resume Analyzer report)."""

from collections import Counter

from reportlab.platypus import KeepTogether, Spacer

from . import theme as t
from .components import (CheckOptions, Chips, MetricCards, RatingRow, ScoreRings, SectionTitle, Tag, bullets,
                         callout, card, columns, markdown_flowables, para, text_block)
from .document import build_pdf, content_width

TYPE_COLORS = {"Technical": t.PRIMARY, "Project": t.ACCENT_DARK, "Behavioural": t.SUCCESS}


def _score(value):
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def insights_section(ins, width):
    overall, ats = _score(ins.get("overall_score")), _score(ins.get("ats_score"))
    rings = ScoreRings([(overall, "Resume score", f"{overall}", t.score_color(overall)),
                        (ats, "ATS score", f"{ats}", t.score_color(ats))])
    intro = []
    if ins.get("headline"):
        intro.append(para(f"**{ins['headline']}**", "lead"))
    if ins.get("summary"):
        intro.append(para(ins["summary"]))
    out = [SectionTitle("AI resume insights", "Quality, ATS readiness and hiring signals"),
           columns([[rings], intro or [para("No summary available.", "small")]], width, [0.36, 0.64])]
    if ins.get("key_skills"):
        out += [para("Key skills", "h3"), Chips(ins["key_skills"])]
    strengths, gaps = ins.get("strengths") or [], ins.get("gaps") or []
    if strengths or gaps:
        out += [Spacer(1, 8), columns([
            [para("Strengths", "h3"), *bullets(strengths, t.SUCCESS)],
            [para("Gaps to address", "h3"), *bullets(gaps, t.WARNING)],
        ], width)]
    if ins.get("suggestions"):
        out += [para("Suggestions to improve the resume", "h3"), *bullets(ins["suggestions"], t.PRIMARY, numbered=True)]
    if ins.get("suitable_roles"):
        out += [para("Suitable roles", "h3"), Chips(ins["suitable_roles"], "good")]
    if ins.get("interview_questions"):
        out += [para("Suggested interview questions", "h3"),
                *bullets(ins["interview_questions"], t.ACCENT_DARK, numbered=True)]
    return out


def build_insights_pdf(ins, prepared_by="", source=""):
    width = content_width()
    name = ins.get("candidate_name") or "Candidate"
    return build_pdf(
        insights_section(ins, width), title=name, kind="AI Resume Insights", eyebrow="AI Assistant · Resume insights",
        subtitle=ins.get("headline", ""), meta={"Prepared by": prepared_by, "Source": source},
    )


def build_job_match_pdf(result, prepared_by="", job_description=""):
    width = content_width()
    score = _score(result.get("match_score"))
    verdict = result.get("verdict") or "—"
    matched, missing = result.get("matched_skills") or [], result.get("missing_skills") or []
    total = len(matched) + len(missing)
    story = [
        SectionTitle("Match overview", "How well the resume covers the job description"),
        columns([
            [ScoreRings([(score, "Match score", f"{score}%", t.score_color(score))], size=78)],
            [para(f"**Verdict: {verdict}**", "lead"), para(result.get("summary", ""))],
        ], width, [0.3, 0.7]),
        Spacer(1, 6),
        MetricCards([
            ("Match score", f"{score}%", "Overall fit", t.score_color(score)),
            ("Verdict", verdict, "AI assessment", t.PRIMARY),
            ("Skills matched", str(len(matched)), f"of {total} required" if total else "", t.SUCCESS),
            ("Skills missing", str(len(missing)), "To develop or highlight", t.DANGER if missing else t.SUCCESS),
        ]),
        SectionTitle("Skills coverage"),
        columns([
            [para("Matched skills", "h3"), Chips(matched, "good") if matched else para("None found.", "small")],
            [para("Missing skills", "h3"), Chips(missing, "bad") if missing else para("Nothing missing.", "small")],
        ], width),
    ]
    if result.get("recommendations"):
        story += [SectionTitle("Recommendations", "What the candidate should add or improve"),
                  *bullets(result["recommendations"], t.PRIMARY, numbered=True)]
    if job_description.strip():
        story += [SectionTitle("Job description"), *text_block(job_description, 6000)]
    return build_pdf(
        story, title=f"{verdict} · {score}%", kind="Job Match Report", eyebrow="AI Assistant · Job match",
        subtitle="Resume compared against the job description", meta={"Prepared by": prepared_by},
    )


def build_interview_pdf(questions, role, level, prepared_by="", candidate=""):
    width = content_width()
    counts = Counter(q.get("type", "Other") for q in questions)
    story = [
        SectionTitle("Interview plan", f"{len(questions)} questions for a {level}-level {role} interview"),
        MetricCards([
            ("Questions", str(len(questions)), f"{level} level", t.PRIMARY),
            ("Technical", str(counts.get("Technical", 0)), "Skills & depth", t.PRIMARY),
            ("Project", str(counts.get("Project", 0)), "Hands-on experience", t.ACCENT_DARK),
            ("Behavioural", str(counts.get("Behavioural", 0)), "Teamwork & ownership", t.SUCCESS),
        ]),
        Spacer(1, 8),
        callout("Rate each answer from 1 (weak) to 5 (excellent) and note evidence. The “What to look for” line "
                "describes a strong answer.", width, "tip", "How to use this kit"),
        SectionTitle("Questions"),
    ]
    for i, q in enumerate(questions, 1):
        qtype = q.get("type") or "Question"
        color = TYPE_COLORS.get(qtype, t.MUTED)
        body = [
            Tag(f"Q{i} · {qtype.upper()}", color),
            Spacer(1, 5),
            para(f"**{q.get('question', '')}**", "card_title"),
        ]
        if q.get("what_to_look_for"):
            body.append(para(f"**What to look for:** {q['what_to_look_for']}", "card_text"))
        body += [Spacer(1, 6), RatingRow()]
        story += [card(body, width, accent=color), Spacer(1, 8)]
    story += [
        SectionTitle("Overall recommendation"),
        KeepTogether([CheckOptions(["Strong hire", "Hire", "Leaning no", "No hire"]), Spacer(1, 10),
                      RatingRow("Overall")]),
    ]
    return build_pdf(
        story, title=f"{role} · {level}", kind="Interview Kit", eyebrow="AI Assistant · Interview questions",
        subtitle=f"Tailored questions{f' for {candidate}' if candidate else ''} with rating and notes space",
        meta={"Prepared by": prepared_by, "Candidate": candidate},
    )


def build_chat_pdf(chat, prepared_by="", resume_name=""):
    """chat: [{'role': 'user' | 'assistant', 'content': markdown}] (assistant replies already cleaned)."""
    width = content_width()
    questions = sum(1 for m in chat if m["role"] == "user")
    story = [
        SectionTitle("Conversation summary"),
        MetricCards([
            ("Messages", str(len(chat)), "In this conversation", t.PRIMARY),
            ("Your questions", str(questions), "Asked to the assistant", t.ACCENT_DARK),
            ("AI answers", str(len(chat) - questions), "From RecruitGen AI", t.SUCCESS),
            ("Resume in context", resume_name or "None", "Used to tailor answers", t.INK),
        ]),
        SectionTitle("Transcript"),
    ]
    for i, m in enumerate(chat, 1):
        user = m["role"] == "user"
        tag = Tag(f"{'You' if user else 'RecruitGen AI'} · {i}", t.PRIMARY if user else t.ACCENT_DARK,
                  filled=user, line=True)
        content = [para(m["content"])] if user else markdown_flowables(m["content"], width)
        content = content or [para("—", "small")]
        story += [Spacer(1, 4), KeepTogether([tag, Spacer(1, 5), content[0]]), *content[1:], Spacer(1, 8)]
    return build_pdf(
        story, title="AI Assistant Conversation", kind="Chat Transcript", eyebrow="AI Assistant · Chat",
        subtitle=f"{len(chat)} messages" + (f" · resume: {resume_name}" if resume_name else ""),
        meta={"Prepared by": prepared_by},
    )
