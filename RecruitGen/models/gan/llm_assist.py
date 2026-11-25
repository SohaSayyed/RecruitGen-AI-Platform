"""
LLM assistance for the hybrid GAN — structured profile suggestions + validated descriptions
------------------------------------------------------------------------------------------
Uses the chat endpoint already configured for the app (.streamlit/secrets.toml [ai] or AI_* environment variables,
read through core.resume_intelligence._settings) — no key is stored here.

The LLM never supplies the 9 numbers directly. It returns structured JSON (title, experience, skills, projects,
certifications, education, soft skills). Python turns that into resume text and runs the app's own feature extractor;
a suggestion is kept only if the extracted features agree with what the LLM claimed (validate_profile).

Reliability: disk cache (identical requests are never sent twice), token-per-minute throttling from the rate-limit
headers, Retry-After on HTTP 429, retries on 5xx / timeouts, a repair attempt for invalid JSON, and a clear status for
every call so callers can fall back to GAN-only output.
"""

import hashlib
import json
import os
import re
import time

import requests

from models.gan.dataset import FEATURE_BOUNDS, PROJECT_DIR
from models.transformer_v3 import feature_extractor as fx

CACHE_DIR = os.path.join(PROJECT_DIR, "data", "llm_cache")

ROLES = ["AI/ML Engineer", "Cloud Engineer", "Data Analyst", "General IT", "Software Engineer"]
ROLE_BRIEF = {
    "AI/ML Engineer": "builds, trains and deploys machine-learning / deep-learning models",
    "Cloud Engineer": "designs, automates and operates cloud infrastructure and deployments",
    "Data Analyst": "analyses business data and builds reports, dashboards and statistical analyses",
    "General IT": "provides IT support, system / network administration and IT operations",
    "Software Engineer": "designs, builds and tests software applications and services",
}
SPECIALISATIONS = {
    "AI/ML Engineer": ["NLP", "computer vision", "MLOps / model deployment", "recommender systems",
                       "LLM / generative AI applications", "time-series forecasting"],
    "Cloud Engineer": ["AWS infrastructure", "Azure platform", "GCP data platform", "Kubernetes / containers",
                       "infrastructure as code", "cloud security & networking"],
    "Data Analyst": ["BI dashboards", "marketing analytics", "financial reporting", "product analytics",
                     "healthcare / operations analytics", "SQL data warehousing"],
    "General IT": ["help desk / IT support", "network administration", "Windows / Linux system administration",
                   "IT operations & asset management", "end-user computing", "IT security support"],
    "Software Engineer": ["Java backend", "React frontend", "full stack JavaScript", "Python services",
                          "C# / .NET applications", "mobile / embedded C++"],
}
# (label, min years, max years, min projects, max projects)
SENIORITY = [("junior", 0, 2, 0, 3), ("mid-level", 2, 5, 1, 5), ("senior", 5, 10, 2, 7), ("lead / principal", 10, 18, 2, 9)]
DEGREE_FORMATS = ("Diploma in ..., B.Tech / B.E. / B.Sc / BCA / Bachelor of ..., M.Tech / M.Sc / MCA / MBA / Master of ..., "
                  "PhD in ...")
# The feature that must show real evidence for each role (score >= 30 = at least two matching keywords)
ROLE_EVIDENCE = {"AI/ML Engineer": "ml_ai_score", "Cloud Engineer": "cloud_score",
                 "Data Analyst": "data_analytics_score", "Software Engineer": "programming_score"}
EDUCATION_LEVEL = {"Diploma": 1, "Bachelor's": 2, "Master's": 3, "PhD": 5}
SOFT_SKILLS = ["communication", "leadership", "teamwork", "collaboration", "presentation", "problem solving",
               "stakeholder management"]

SYSTEM = ("You generate SYNTHETIC, fictional candidate profiles for machine-learning dataset augmentation. "
          "Never use real people's names, real employers or contact details. Answer with valid JSON only.")


# ============================================================
# Client
# ============================================================

def _settings():
    from core.resume_intelligence import _settings as app_settings
    return app_settings()


def _seconds(value):
    """'3.059s' / '1m26.4s' / '250ms' -> seconds."""
    if not value:
        return 0.0
    total = 0.0
    for num, unit in re.findall(r"([\d.]+)\s*(ms|h|m|s)", str(value)):
        total += float(num) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[unit]
    return total or (float(value) if re.fullmatch(r"[\d.]+", str(value)) else 0.0)


def _parse_json(text):
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    try:
        return json.loads(text)
    except ValueError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except ValueError:
                return None
    return None


class LLMClient:
    """status of every call: ok · cached · unavailable · offline · rate_limited · invalid_json · error"""

    def __init__(self, cache_dir=CACHE_DIR, min_tokens=2500, max_wait=45):
        self.settings = _settings()
        self.cache_dir = cache_dir
        self.min_tokens = min_tokens        # wait for the token window to refill below this
        self.max_wait = max_wait            # never block longer than this per call
        self.remaining_tokens = None
        self.tokens_reset_at = 0.0
        self.calls = {"network": 0, "cached": 0, "failed": 0}
        self.key_index = 0                  # primary key first; switch to the backup key if it fails
        os.makedirs(cache_dir, exist_ok=True)

    def available(self):
        s = self.settings
        return bool(s["api_keys"] and s["base_url"] and s["model"])

    def _next_key(self):
        if self.key_index + 1 < len(self.settings["api_keys"]):
            self.key_index += 1
            self.remaining_tokens = None
            return True
        return False

    def _cache_path(self, payload):
        key = json.dumps({k: payload[k] for k in ("model", "messages", "temperature", "max_tokens")}, sort_keys=True)
        return os.path.join(self.cache_dir, hashlib.sha256(key.encode()).hexdigest()[:32] + ".json")

    def _throttle(self):
        if self.remaining_tokens is not None and self.remaining_tokens < self.min_tokens:
            wait = self.tokens_reset_at - time.time()
            if 0 < wait <= self.max_wait:
                time.sleep(wait)

    def _post(self, payload):
        s = self.settings
        key = s["api_keys"][self.key_index]
        r = requests.post(f"{s['base_url']}/chat/completions", headers={"Authorization": f"Bearer {key}"},
                          json=payload, timeout=120)
        if "x-ratelimit-remaining-tokens" in r.headers:
            self.remaining_tokens = int(float(r.headers["x-ratelimit-remaining-tokens"]))
            self.tokens_reset_at = time.time() + _seconds(r.headers.get("x-ratelimit-reset-tokens"))
        return r

    def chat_json(self, user, max_tokens=4000, temperature=0.9, system=SYSTEM, use_cache=True, retries=4):
        """Returns (data or None, status)."""
        if not self.available():
            return None, "unavailable"
        payload = {"model": self.settings["model"], "temperature": temperature, "max_tokens": max_tokens,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                   "response_format": {"type": "json_object"}}
        if "gpt-oss" in payload["model"]:
            payload["reasoning_effort"] = "low"      # fewer hidden reasoning tokens -> fewer rate-limit waits
        path = self._cache_path(payload)
        if use_cache and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                self.calls["cached"] += 1
                return json.load(f), "cached"

        status = "error"
        for attempt in range(retries):
            self._throttle()
            try:
                r = self._post(payload)
            except (requests.ConnectionError, requests.Timeout):
                status = "offline"
                time.sleep(2 * (attempt + 1))
                continue
            except requests.RequestException:
                status = "error"
                break
            if r.status_code == 200:
                content = r.json()["choices"][0]["message"].get("content", "")
                data = _parse_json(content)
                if data is None:
                    status = "invalid_json"
                    payload["messages"] = payload["messages"][:2] + [
                        {"role": "assistant", "content": content[:2000]},
                        {"role": "user", "content": "That was not valid JSON. Return the same answer as one valid JSON object only."}]
                    continue
                self.calls["network"] += 1
                if use_cache:
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(data, f)
                return data, "ok"
            if r.status_code in (401, 403):
                status = "error"
                if self._next_key():
                    continue
                break
            if r.status_code == 429:
                status = "rate_limited"
                if self._next_key():
                    continue
                time.sleep(min(self.max_wait, max(2.0, _seconds(r.headers.get("retry-after")) or 5.0 * (attempt + 1))))
                continue
            if r.status_code == 400:
                if payload.pop("reasoning_effort", None) is not None or payload.pop("response_format", None) is not None:
                    continue
            if r.status_code in (500, 502, 503, 504):
                status = "error"
                time.sleep(2 * (attempt + 1))
                continue
            status = "error"
            break
        self.calls["failed"] += 1
        return None, status


# ============================================================
# Profile -> resume text -> 9 features (app extractor)
# ============================================================

def _clean_list(items, limit):
    out = []
    for item in items or []:
        if isinstance(item, dict):
            item = item.get("name") or item.get("title") or ""
        item = re.sub(r"\s+", " ", str(item)).strip(" -•*\t")
        if 2 <= len(item) <= 120 and item.lower() not in {o.lower() for o in out}:
            out.append(item)
    return out[:limit]


def profile_text(p):
    """Structured profile -> resume text in the layout the extractor parses (sections, numbered projects, bullets).
    The free-text summary is NOT included, so prose cannot change the features."""
    lines = [f"{p['title']} with {p['experience_years']:g} years of experience."]
    lines.append("Skills: " + ", ".join(p["skills"]) + ".")
    if p["projects"]:
        lines.append("Projects:")
        lines += [f"{i}. {name}" for i, name in enumerate(p["projects"], 1)]
    if p["certifications"]:
        lines.append("Certifications:")
        lines += [f"- {c}" for c in p["certifications"]]
    lines.append(f"Education: {p['degree']}.")
    if p["soft_skills"]:
        lines.append("Soft skills: " + ", ".join(p["soft_skills"]) + ".")
    return "\n".join(lines)


def normalise_profile(raw):
    """Coerce the LLM JSON into the profile schema. Returns (profile, error)."""
    if not isinstance(raw, dict):
        return None, "not an object"
    try:
        exp = round(float(raw.get("experience_years")), 1)
    except (TypeError, ValueError):
        return None, "experience_years is not a number"
    education = str(raw.get("education", "")).strip()
    education = next((k for k in EDUCATION_LEVEL if k.lower().rstrip("'s") in education.lower()), education)
    p = {
        "title": re.sub(r"\s+", " ", str(raw.get("title", ""))).strip()[:80],
        "experience_years": exp,
        "skills": _clean_list(raw.get("skills"), 25),
        "projects": _clean_list(raw.get("projects"), 15),
        "certifications": _clean_list(raw.get("certifications"), 10),
        "education": education,
        "degree": re.sub(r"\s+", " ", str(raw.get("degree", ""))).strip()[:80],
        "soft_skills": [s for s in _clean_list(raw.get("soft_skills"), 7) if s.lower() in SOFT_SKILLS],
        "summary": re.sub(r"\s+", " ", str(raw.get("summary", ""))).strip()[:500],
    }
    if not p["title"]:
        return None, "missing title"
    if not 0 <= exp <= FEATURE_BOUNDS["experience_years"][1]:
        return None, "experience out of range"
    if len(p["skills"]) < 3:
        return None, "fewer than 3 skills"
    if p["education"] not in EDUCATION_LEVEL:
        return None, "invalid education"
    if not p["degree"]:
        return None, "missing degree"
    return p, None


def validate_profile(p, role=None):
    """Extract the 9 features from the profile text and check they agree with the profile's own claims.
    Returns (features or None, reason)."""
    features = fx.extract_features(profile_text(p))
    if abs(features["experience_years"] - p["experience_years"]) > 0.05:
        return None, "experience not recoverable from text"
    if features["projects_count"] != len(p["projects"]):
        return None, "project count mismatch"
    if features["certification_count"] != len(p["certifications"]):
        return None, "certification count mismatch"
    if features["education_level"] != EDUCATION_LEVEL[p["education"]]:
        return None, "education level mismatch"
    for name, (lo, hi) in FEATURE_BOUNDS.items():
        if not lo <= features[name] <= hi:
            return None, f"{name} out of range"
    if role in ROLE_EVIDENCE and features[ROLE_EVIDENCE[role]] < 30:
        return None, f"too little {ROLE_EVIDENCE[role].replace('_score', '')} evidence for {role}"
    return {k: float(v) for k, v in features.items()}, "ok"


# ============================================================
# Prompts
# ============================================================

def _keyword_hint():
    return ("Feature extractor keyword groups (skills are scored by how many of these appear): "
            f"programming {fx.PROGRAMMING_KEYWORDS[:14]}; data analytics {fx.DATA_ANALYTICS_KEYWORDS[:12]}; "
            f"ML/AI {fx.ML_AI_KEYWORDS[:14]}; cloud {fx.CLOUD_KEYWORDS[:14]}.")


def seed_prompt(role, n, batch, real_hints=None):
    """Prompt for n diverse profiles; batch index rotates specialisation and seniority mix for diversity."""
    spec = SPECIALISATIONS[role][batch % len(SPECIALISATIONS[role])]
    bands = [SENIORITY[(batch + i) % len(SENIORITY)] for i in range(n)]
    band_text = "; ".join(f"profile {i + 1}: {b[0]}, {b[1]}-{b[2]} years, {b[3]}-{b[4]} projects"
                          for i, b in enumerate(bands))
    hints = f"\nPatterns observed in real resumes: {real_hints}" if real_hints else ""
    return f"""Create {n} distinct SYNTHETIC candidate profiles for the role "{role}" (someone who {ROLE_BRIEF[role]}).
Focus area for this batch: {spec}. Seniority per profile: {band_text}.{hints}
{_keyword_hint()}

Realism rules: experience and project count must fit the seniority band (use the whole range, not always the same
number); certifications vary from 0 to 5 and are not tied to seniority; education must fit the role; skills must be
concrete tools/technologies typical for the focus area, not a copy of the keyword list.

Return {{"profiles": [...]}} where every profile has:
"title": job title (string), "experience_years": number, "skills": 4-16 concrete skills,
"projects": short project names (no numbers in names), "certifications": 0-5 certification names (may be empty),
"education": one of "Diploma", "Bachelor's", "Master's", "PhD",
"degree": degree name written as one of: {DEGREE_FORMATS},
"soft_skills": 0-4 items from {SOFT_SKILLS}, "summary": one or two sentences without numbers.
Make the {n} profiles clearly different from each other (skills, specialisation, seniority, education)."""


def enrichment_prompt(role, items):
    """items: [(id, features)] — ask for profile content that reproduces the GAN's numbers exactly."""
    rows = []
    for pid, f in items:
        rows.append({
            "id": pid, "experience_years": f["experience_years"], "projects": int(f["projects_count"]),
            "certifications": int(f["certification_count"]),
            "education": next(k for k, v in EDUCATION_LEVEL.items() if v == int(f["education_level"])) if int(f["education_level"]) in EDUCATION_LEVEL.values() else "Bachelor's",
            "programming_keywords": int(round(f["programming_score"] / 15)),
            "data_analytics_keywords": int(round(f["data_analytics_score"] / 15)),
            "ml_ai_keywords": int(round(f["ml_ai_score"] / 15)),
            "cloud_keywords": int(round(f["cloud_score"] / 15)),
            "soft_skills": max(0, int(round((f["communication_score"] - 30) / 10))) if f["communication_score"] != 50 else 0,
        })
    return f"""Write SYNTHETIC profile content for {len(rows)} "{role}" candidates whose numbers were produced by a
generative model. Do not change any number. For each item, choose skills so that the number of matching keywords per
group equals the given count (keyword groups below; "sql" counts for both programming and data analytics, so avoid it
unless both need it), list exactly the given number of projects and certifications, and use a degree matching the
education level.
{_keyword_hint()}
Items: {json.dumps(rows)}

Return {{"profiles": [...]}} in the same order, each with: "id", "title", "experience_years" (same number),
"skills" (list), "projects" (exactly the given count, short names without numbers), "certifications" (exactly the given
count), "education" (same), "degree", "soft_skills" (exactly the given count, from {SOFT_SKILLS}),
"summary" (one or two sentences without numbers)."""
