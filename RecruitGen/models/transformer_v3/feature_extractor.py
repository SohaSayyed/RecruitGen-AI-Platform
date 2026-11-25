# ============================================================
# RecruitGen V3 - Resume Feature Extraction Module
# ============================================================
#
# PURPOSE:
# Convert textual resume/profile information into the
# 9 numerical features expected by Transformer V3.
#
# PIPELINE:
#
# Resume Text
#     ↓
# Feature Extraction
#     ↓
# 9 Candidate Features
#     ↓
# Transformer V3
#     ↓
# Predicted Role + Confidence
#
# ============================================================


import re


# ============================================================
# FEATURE ORDER
# ============================================================
#
# IMPORTANT:
# This order MUST match the Transformer V3 training order.
#
# ============================================================

FEATURE_ORDER = [
    "experience_years",
    "projects_count",
    "programming_score",
    "data_analytics_score",
    "ml_ai_score",
    "cloud_score",
    "communication_score",
    "certification_count",
    "education_level"
]


# ============================================================
# KEYWORD GROUPS
# ============================================================

PROGRAMMING_KEYWORDS = [
    "python",
    "java",
    "c++",
    "c#",
    "javascript",
    "typescript",
    "html",
    "css",
    "react",
    "node.js",
    "nodejs",
    "sql",
    "mysql",
    "postgresql",
    "mongodb",
    "git",
    "github",
    "data structures",
    "algorithms",
    "oop",
    "object oriented programming"
]


DATA_ANALYTICS_KEYWORDS = [
    "data analysis",
    "data analytics",
    "data analyst",
    "pandas",
    "numpy",
    "excel",
    "power bi",
    "tableau",
    "matplotlib",
    "seaborn",
    "statistics",
    "statistical analysis",
    "sql",
    "dashboard",
    "visualization",
    "business intelligence",
    "kpi",
    "reporting"
]


ML_AI_KEYWORDS = [
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "ai",
    "ml",
    "tensorflow",
    "keras",
    "pytorch",
    "scikit-learn",
    "sklearn",
    "cnn",
    "convolutional neural network",
    "lstm",
    "nlp",
    "natural language processing",
    "computer vision",
    "opencv",
    "mediapipe",
    "transformer",
    "bert",
    "rag",
    "generative ai",
    "neural network",
    "classification",
    "regression",
    "model training",
    "model deployment"
]


CLOUD_KEYWORDS = [
    "aws",
    "amazon web services",
    "azure",
    "microsoft azure",
    "gcp",
    "google cloud",
    "ec2",
    "s3",
    "lambda",
    "api gateway",
    "dynamodb",
    "rds",
    "iam",
    "vpc",
    "cloudformation",
    "terraform",
    "docker",
    "kubernetes",
    "cloud computing",
    "serverless"
]


COMMUNICATION_KEYWORDS = [
    "communication",
    "leadership",
    "teamwork",
    "collaboration",
    "presentation",
    "presentations",
    "public speaking",
    "interpersonal",
    "client communication",
    "stakeholder",
    "team player",
    "problem solving",
    "problem-solving"
]


# ============================================================
# CERTIFICATION PROVIDERS
# ============================================================

CERTIFICATION_KEYWORDS = [
    "nptel",
    "coursera",
    "udemy",
    "aws certified",
    "microsoft certified",
    "google certified",
    "cisco",
    "forage"
]


# ============================================================
# HELPER
# ============================================================

def _count_keyword_matches(text, keywords):
    """
    Count unique keyword matches.

    Each keyword contributes at most one match.
    """

    text_lower = text.lower()

    count = 0

    for keyword in keywords:

        if keyword.lower() in text_lower:
            count += 1

    return count


# ============================================================
# SCORE CALCULATION
# ============================================================

def _calculate_score(match_count, max_score=100):
    """
    Existing RecruitGen scoring behavior.

    Each matched keyword contributes 15 points.

    Example:

    1 match  -> 15
    2 matches -> 30
    3 matches -> 45
    ...
    7 matches -> 100 maximum
    """

    score = match_count * 15

    score = min(
        score,
        max_score
    )

    return float(score)


# ============================================================
# EXPERIENCE
# ============================================================

def extract_experience(text):
    """
    Extract experience in years.

    Examples:

        1.5 years of experience
        2 years experience
        6 months experience
        1 year experience

    Returns:
        float
    """

    text_lower = text.lower()

    # --------------------------------------------------------
    # Years
    # --------------------------------------------------------

    year_patterns = [
        r'(\d+(?:\.\d+)?)\s*\+?\s*years?\s+(?:of\s+)?experience',
        r'experience\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*\+?\s*years?',
        r'(\d+(?:\.\d+)?)\s*\+?\s*years?\s+in\s+'
    ]

    for pattern in year_patterns:

        match = re.search(
            pattern,
            text_lower
        )

        if match:

            return float(
                match.group(1)
            )

    # --------------------------------------------------------
    # Months
    # --------------------------------------------------------

    month_patterns = [
        r'(\d+)\s*months?\s+(?:of\s+)?experience',
        r'experience\s*[:\-]?\s*(\d+)\s*months?'
    ]

    for pattern in month_patterns:

        match = re.search(
            pattern,
            text_lower
        )

        if match:

            months = float(
                match.group(1)
            )

            return round(
                months / 12,
                2
            )

    # --------------------------------------------------------
    # Internship duration
    # --------------------------------------------------------

    internship_match = re.search(
        r'(\d+)\s*(?:month|months)\s+(?:internship|intern)',
        text_lower
    )

    if internship_match:

        months = float(
            internship_match.group(1)
        )

        return round(
            months / 12,
            2
        )

    # --------------------------------------------------------
    # Fresher
    # --------------------------------------------------------

    fresher_words = [
        "fresher",
        "fresh graduate",
        "recent graduate",
        "entry level",
        "entry-level"
    ]

    if any(
        word in text_lower
        for word in fresher_words
    ):

        return 0.0

    return 0.0


# ============================================================
# PROJECT COUNT
# ============================================================

def extract_projects_count(text):
    """
    Robust project-count detection.

    Handles:

        Projects:
        1. Project A
        2. Project B
        3. Project C
        4. Project D

    Handles:

        Projects: 1. Project A 2. Project B 3. Project C

    Handles:

        Projects:
        - Project A
        - Project B

    Handles:

        Project 1: ...
        Project 2: ...
    """

    text_lower = text.lower()

    # ========================================================
    # 1. EXPLICIT PROJECT COUNT
    # ========================================================

    explicit_patterns = [
        r'\b(\d+)\s+projects?\b',
        r'\b(\d+)\s+major\s+projects?\b',
        r'\b(\d+)\s+academic\s+projects?\b',
        r'\b(\d+)\s+personal\s+projects?\b'
    ]

    for pattern in explicit_patterns:

        match = re.search(
            pattern,
            text_lower
        )

        if match:

            return min(
                int(match.group(1)),
                100
            )

    # ========================================================
    # 2. PROJECT 1:, PROJECT 2:, ETC.
    # ========================================================

    numbered_projects = re.findall(
        r'\bproject\s+(\d+)\s*[:\-]',
        text_lower
    )

    if numbered_projects:

        return min(
            len(set(numbered_projects)),
            100
        )

    # ========================================================
    # 3. FIND PROJECTS SECTION
    # ========================================================

    project_match = re.search(
        r'\bprojects?\s*:',
        text_lower
    )

    if project_match:

        project_section = text[
            project_match.end():
        ]

        section_lower = project_section.lower()

        # ====================================================
        # 4. STOP AT NEXT SECTION
        # ====================================================

        stop_patterns = [
            r'\bcertifications?\s*:',
            r'\bcertificates?\s*:',
            r'\beducation\s*:',
            r'\bexperience\s*:',
            r'\bskills\s*:',
            r'\btechnical skills\s*:',
            r'\bachievements?\s*:',
            r'\bawards?\s*:',
            r'\bpublications?\s*:',
            r'\binternships?\s*:'
        ]

        stop_positions = []

        for pattern in stop_patterns:

            match = re.search(
                pattern,
                section_lower
            )

            if match:

                stop_positions.append(
                    match.start()
                )

        if stop_positions:

            project_section = project_section[
                :min(stop_positions)
            ]

        # ====================================================
        # 5. NUMBERED LIST
        # ====================================================
        #
        # Examples:
        #
        # 1. Project A
        # 2. Project B
        # 3. Project C
        #
        # Also:
        #
        # 1) Project A
        # 2) Project B
        #
        # ====================================================

        numbered_matches = re.findall(
            r'(?:^|\n|\s)(\d+)\s*[\.\)\-–—:]\s+',
            project_section
        )

        if numbered_matches:

            unique_numbers = []

            for number in numbered_matches:

                if number not in unique_numbers:

                    unique_numbers.append(
                        number
                    )

            return min(
                len(unique_numbers),
                100
            )

        # ====================================================
        # 6. BULLET LIST
        # ====================================================

        bullet_matches = re.findall(
            r'(?:^|\n)\s*[\-•*]\s+([^\n]+)',
            project_section
        )

        if bullet_matches:

            return min(
                len(bullet_matches),
                100
            )

        # ====================================================
        # 7. INLINE NUMBERED LIST
        # ====================================================

        inline_numbered = re.findall(
            r'(?:^|\s)(\d+)\s*[\.\)]\s+[A-Za-z]',
            project_section
        )

        if inline_numbered:

            return min(
                len(set(inline_numbered)),
                100
            )

        # ====================================================
        # 8. SEMICOLON / PIPE SEPARATED
        # ====================================================

        separated_projects = re.split(
            r'\s*[;|]\s*',
            project_section
        )

        meaningful_parts = [
            part.strip()
            for part in separated_projects
            if len(part.strip()) > 5
        ]

        if len(meaningful_parts) > 1:

            return min(
                len(meaningful_parts),
                100
            )

    # ========================================================
    # 9. FALLBACK
    # ========================================================

    if re.search(
        r'\bproject\b',
        text_lower
    ):

        return 1

    return 0


# ============================================================
# PROGRAMMING SCORE
# ============================================================

def extract_programming_score(text):

    matches = _count_keyword_matches(
        text,
        PROGRAMMING_KEYWORDS
    )

    return _calculate_score(
        matches
    )


# ============================================================
# DATA ANALYTICS SCORE
# ============================================================

def extract_data_analytics_score(text):

    matches = _count_keyword_matches(
        text,
        DATA_ANALYTICS_KEYWORDS
    )

    return _calculate_score(
        matches
    )


# ============================================================
# ML / AI SCORE
# ============================================================

def extract_ml_ai_score(text):

    matches = _count_keyword_matches(
        text,
        ML_AI_KEYWORDS
    )

    return _calculate_score(
        matches
    )


# ============================================================
# CLOUD SCORE
# ============================================================

def extract_cloud_score(text):

    matches = _count_keyword_matches(
        text,
        CLOUD_KEYWORDS
    )

    return _calculate_score(
        matches
    )


# ============================================================
# COMMUNICATION SCORE
# ============================================================

def extract_communication_score(text):
    """
    Estimate communication evidence.

    Existing behavior:
        No evidence -> 50
        Evidence -> 30 + 10 per keyword
        Maximum -> 100
    """

    matches = _count_keyword_matches(
        text,
        COMMUNICATION_KEYWORDS
    )

    if matches == 0:

        return 50.0

    score = 30 + (
        matches * 10
    )

    return float(
        min(
            score,
            100
        )
    )


# ============================================================
# CERTIFICATION COUNT
# ============================================================

def extract_certification_count(text):
    """
    Robust certification-count detection.

    IMPORTANT:
    This function prioritizes the actual Certifications:
    section and does not simply count every occurrence of
    AWS, NPTEL, Cisco, etc. in the entire resume.

    Handles:

        Certifications:
        AWS Cloud Practitioner
        NPTEL Machine Learning
        Cisco AI

    Handles:

        Certifications: AWS Cloud Practitioner,
        NPTEL Machine Learning, Cisco AI

    Handles:

        Certifications:
        1. AWS
        2. NPTEL
        3. Cisco

    Handles:

        Certifications:
        - AWS
        - NPTEL
        - Cisco
    """

    text_lower = text.lower()

    # ========================================================
    # 1. EXPLICIT CERTIFICATION COUNT
    # ========================================================

    explicit_patterns = [
        r'\b(\d+)\s+certifications?\b',
        r'\b(\d+)\s+certificates?\b'
    ]

    for pattern in explicit_patterns:

        match = re.search(
            pattern,
            text_lower
        )

        if match:

            return min(
                int(match.group(1)),
                50
            )

    # ========================================================
    # 2. CERTIFICATIONS: 3
    # ========================================================

    explicit_section_count = re.search(
        r'\b(?:certifications?|certificates?)\s*[:\-]\s*(\d+)\b',
        text_lower
    )

    if explicit_section_count:

        return min(
            int(explicit_section_count.group(1)),
            50
        )

    # ========================================================
    # 3. FIND CERTIFICATION SECTION
    # ========================================================

    certification_match = re.search(
        r'\b(?:certifications?|certificates?)\s*:',
        text_lower
    )

    if certification_match:

        certification_section = text[
            certification_match.end():
        ]

        section_lower = certification_section.lower()

        # ====================================================
        # 4. STOP AT NEXT MAJOR SECTION
        # ====================================================

        stop_patterns = [
            r'\beducation\s*:',
            r'\bexperience\s*:',
            r'\bskills\s*:',
            r'\btechnical skills\s*:',
            r'\bprojects?\s*:',
            r'\bachievements?\s*:',
            r'\bawards?\s*:',
            r'\bpublications?\s*:',
            r'\binternships?\s*:'
        ]

        stop_positions = []

        for pattern in stop_patterns:

            match = re.search(
                pattern,
                section_lower
            )

            if match:

                stop_positions.append(
                    match.start()
                )

        if stop_positions:

            certification_section = certification_section[
                :min(stop_positions)
            ]

        # ====================================================
        # 5. REMOVE TRAILING EDUCATION
        #
        # Example:
        #
        # Certifications: AWS, NPTEL, Cisco.
        # B.Tech in Software Engineering.
        #
        # ====================================================

        certification_section = re.split(
            r'\.\s+(?=[A-Z][^.\n]{0,80}\b(?:B\.?\s*Tech|B\.?\s*E\.?|M\.?\s*Tech|M\.?\s*E\.?|MBA|MCA|Bachelor|Master|Education)\b)',
            certification_section,
            maxsplit=1,
            flags=re.IGNORECASE
        )[0]

        # ====================================================
        # 6. NUMBERED CERTIFICATION LIST
        # ====================================================

        numbered_matches = re.findall(
            r'(?:^|\n|\s)(\d+)\s*[\.\)\-:]\s+',
            certification_section
        )

        if numbered_matches:

            return min(
                len(set(numbered_matches)),
                50
            )

        # ====================================================
        # 7. BULLET CERTIFICATION LIST
        # ====================================================

        bullet_matches = re.findall(
            r'(?:^|\n)\s*[\-•*]\s+([^\n]+)',
            certification_section
        )

        if bullet_matches:

            return min(
                len(bullet_matches),
                50
            )

        # ====================================================
        # 8. COMMA / SEMICOLON / PIPE LIST
        # ====================================================

        separated_items = re.split(
            r'\s*[,;|]\s*',
            certification_section
        )

        meaningful_items = []

        for item in separated_items:

            item = item.strip(
                " \t\n\r.-"
            )

            if len(item) >= 3:

                meaningful_items.append(
                    item
                )

        if len(meaningful_items) >= 2:

            return min(
                len(meaningful_items),
                50
            )

        # ====================================================
        # 9. NEWLINE CERTIFICATION LIST
        # ====================================================

        lines = []

        for line in certification_section.splitlines():

            line = line.strip()

            if not line:

                continue

            if re.match(
                r'^(education|experience|skills|projects?)\s*:?',
                line,
                re.IGNORECASE
            ):

                continue

            lines.append(
                line
            )

        if len(lines) > 0:

            return min(
                len(lines),
                50
            )

        # ====================================================
        # 10. SINGLE CERTIFICATION
        # ====================================================

        cleaned = certification_section.strip(
            " \t\n\r.-,"
        )

        if cleaned:

            return 1

    # ========================================================
    # 11. FALLBACK
    #
    # Only count explicit certification wording.
    # Generic AWS/NPTEL/Cisco skill mentions are NOT treated
    # as separate certifications.
    # ========================================================

    fallback_patterns = [
        r'\bnptel\s+(?:certification|certificate|certified)\b',
        r'\bcoursera\s+(?:certification|certificate|certified)\b',
        r'\budemy\s+(?:certification|certificate|certified)\b',
        r'\baws\s+certified\b',
        r'\bmicrosoft\s+certified\b',
        r'\bgoogle\s+certified\b',
        r'\bcisco\s+certified\b',
        r'\bforage\s+(?:certification|certificate|certified)\b'
    ]

    fallback_count = 0

    for pattern in fallback_patterns:

        if re.search(
            pattern,
            text_lower
        ):

            fallback_count += 1

    return min(
        fallback_count,
        50
    )


# ============================================================
# EDUCATION LEVEL
# ============================================================

def extract_education_level(text):
    """
    Determine highest education level.

    Encoding used by RecruitGen V3:

        1 = Diploma / 12th / High School
        2 = Bachelor's
        3 = Master's
        4 = Advanced degree
        5 = PhD / Doctorate

    IMPORTANT:
    B.Tech / B.E. must return 2.
    M.Tech / MBA / MCA must return 3.
    PhD must return 5.
    """

    text_lower = text.lower()

    # ========================================================
    # 5 - PhD / Doctorate
    # ========================================================

    phd_patterns = [
        r'\bph\.?\s*d\.?\b',
        r'\bdoctorate\b',
        r'\bdoctoral\b'
    ]

    for pattern in phd_patterns:

        if re.search(
            pattern,
            text_lower
        ):

            return 5

    # ========================================================
    # 3 - Master's
    # ========================================================

    masters_patterns = [
        r"\bmaster'?s\b",
        r'\bmasters\b',
        r'\bmaster\s+of\b',
        r'\bm\.?\s*tech\b',
        r'\bm\.?\s*e\.?\b',
        r'\bm\.?\s*b\.?\s*a\.?\b',
        r'\bm\.?\s*c\.?\s*a\.?\b',
        r'\bm\.?\s*sc\.?\b'
    ]

    for pattern in masters_patterns:

        if re.search(
            pattern,
            text_lower
        ):

            return 3

    # ========================================================
    # 2 - Bachelor's
    # ========================================================

    bachelor_patterns = [
        r'\bbachelor\b',
        r'\bb\.?\s*tech\b',
        r'\bb\.?\s*e\.?\b',
        r'\bb\.?\s*sc\.?\b',
        r'\bb\.?\s*c\.?\s*a\.?\b',
        r'\bb\.?\s*b\.?\s*a\.?\b',
        r'\bundergraduate\b'
    ]

    for pattern in bachelor_patterns:

        if re.search(
            pattern,
            text_lower
        ):

            return 2

    # ========================================================
    # 1 - Diploma / 12th / School
    # ========================================================

    basic_education_patterns = [
        r'\bdiploma\b',
        r'\b12th\b',
        r'\bhigher secondary\b',
        r'\bhigh school\b'
    ]

    for pattern in basic_education_patterns:

        if re.search(
            pattern,
            text_lower
        ):

            return 1

    # ========================================================
    # DEFAULT
    # ========================================================

    return 1


# ============================================================
# MAIN FEATURE EXTRACTION
# ============================================================

def extract_features(resume_text):
    """
    Convert resume text into the exact 9 features expected
    by Transformer V3.

    Returns:
        dict
    """

    if not resume_text:

        raise ValueError(
            "Resume text cannot be empty."
        )

    text = resume_text.strip()

    if len(text) < 20:

        raise ValueError(
            "Please enter a more complete resume or candidate profile."
        )

    features = {

        "experience_years":
            extract_experience(text),

        "projects_count":
            extract_projects_count(text),

        "programming_score":
            extract_programming_score(text),

        "data_analytics_score":
            extract_data_analytics_score(text),

        "ml_ai_score":
            extract_ml_ai_score(text),

        "cloud_score":
            extract_cloud_score(text),

        "communication_score":
            extract_communication_score(text),

        "certification_count":
            extract_certification_count(text),

        "education_level":
            extract_education_level(text)
    }

    return features


# ============================================================
# DICTIONARY -> MODEL LIST
# ============================================================

def features_to_list(features):
    """
    Convert feature dictionary to the exact order required
    by Transformer V3.
    """

    return [
        float(features["experience_years"]),
        float(features["projects_count"]),
        float(features["programming_score"]),
        float(features["data_analytics_score"]),
        float(features["ml_ai_score"]),
        float(features["cloud_score"]),
        float(features["communication_score"]),
        float(features["certification_count"]),
        float(features["education_level"])
    ]


# ============================================================
# MODULE TEST
# ============================================================

if __name__ == "__main__":

    sample_resume = """
    Machine Learning Engineer with 1.5 years of experience.

    Skills:
    Python, TensorFlow, PyTorch, Scikit-learn,
    Pandas, SQL, AWS, EC2, S3.

    Projects:
    1. ML prediction system
    2. Resume screening
    3. Data analytics dashboard
    4. Cloud deployment

    Certifications:
    AWS Cloud Practitioner,
    NPTEL Machine Learning,
    Cisco AI.

    B.Tech in Software Engineering.

    Strong communication, teamwork,
    collaboration and presentation skills.
    """

    extracted = extract_features(
        sample_resume
    )

    print("=" * 65)
    print("RECRUITGEN V3 FEATURE EXTRACTION TEST")
    print("=" * 65)

    for key in FEATURE_ORDER:

        print(
            f"{key:25s}: {extracted[key]}"
        )

    print("=" * 65)

    model_features = features_to_list(
        extracted
    )

    print(
        "MODEL LIST:",
        model_features
    )

    print("=" * 65)