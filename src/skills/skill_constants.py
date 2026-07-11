from __future__ import annotations

import re
from typing import Dict, List, Set


NOISE_TERMS: Set[str] = {
    "skill", "skills", "technical", "technical skills",
    "language", "languages", "tool", "tools",
    "technology", "technologies", "framework", "frameworks",
    "library", "libraries", "experience", "knowledge",
    "understanding", "familiarity", "proficiency", "ability",
    "candidate", "developer", "engineer", "engineering",
    "machine learning engineer", "software engineer",
    "data scientist", "ai engineer", "ml engineer",
    "role", "job", "position", "responsibility", "responsibilities",
    "requirement", "requirements", "qualification", "qualifications",
    "project", "projects", "work", "team", "company", "business",
    "product", "application", "applications", "system", "systems",
    "model", "models", "feature", "features", "training", "learning",
    "learn", "machine", "development", "problem", "solution", "solutions",
    "computer science engineering", "summary", "objective", "profile",
    "education", "certification", "certifications", "responsibilities",
    "information", "background", "management", "design",
}

ROLE_PATTERNS: List[str] = [
    r"\b(?:an?|the)\s+(?:senior\s+|junior\s+|lead\s+|principal\s+)?"
    r"(?:machine learning|software|data|ai|backend|frontend|full[- ]stack|"
    r"deep learning|nlp|ml|devops|cloud)\s+(?:engineer|developer|scientist|analyst|researcher)\b",

    r"\b(?:machine learning|software|data|ai|backend|frontend|full[- ]stack|"
    r"deep learning|nlp|ml|devops|cloud)\s+(?:engineer|developer|scientist|analyst|researcher)\b",
]

SECTION_EVIDENCE_WEIGHT: Dict[str, float] = {
    "Work Experience": 1.0,
    "Projects": 0.85,
    "Technical Skills": 0.70,
    "Certifications": 0.60,
    "Summary": 0.50,
    "Education": 0.40,
    "Unknown": 0.30,
}

REQUIREMENT_WEIGHTS: Dict[str, float] = {
    "mandatory": 1.0,
    "preferred": 0.6,
    "optional": 0.3,
}

MATCH_SCORES: Dict[str, float] = {
    "exact_match": 1.0,
    "alias_match": 1.0,
    "related_match": 0.35,
    "missing": 0.0,
}

FINAL_SCORE_WEIGHTS: Dict[str, float] = {
    "mandatory_skill_coverage": 0.50,
    "preferred_skill_coverage": 0.20,
    "evidence_strength": 0.15,
    "project_experience_relevance": 0.10,
    "education_or_certification_relevance": 0.05,
}

MANDATORY_KEYWORDS: List[str] = [
    "required", "must have", "must-have", "should have", "should-have",
    "mandatory", "minimum requirement", "minimum requirements",
    "strong experience in", "strong experience with",
    "proficiency in", "proficiency with", "solid experience",
    "hands-on experience", "extensive experience",
    "working knowledge of", "expertise in",
]

PREFERRED_KEYWORDS: List[str] = [
    "preferred", "nice to have", "nice-to-have", "good to have",
    "good-to-have", "bonus", "plus", "desirable", "advantage",
    "ideally", "beneficial",
]

OPTIONAL_KEYWORDS: List[str] = [
    "exposure to", "familiarity with", "basic knowledge of",
    "basic understanding of", "awareness of", "some experience with",
    "introductory knowledge",
]

SECTION_PATTERNS: Dict[str, List[str]] = {
    "Work Experience": [
        r"(?:work|professional|employment)\s*experience",
        r"experience", r"work history", r"career history",
    ],
    "Projects": [
        r"projects?", r"personal projects?", r"academic projects?",
        r"key projects?",
    ],
    "Technical Skills": [
        r"technical\s+skills?", r"skills?\s+(?:&|and)?\s*(?:competencies|expertise)?",
        r"core\s+competencies", r"technologies", r"tech\s+stack",
        r"programming\s+skills?",
    ],
    "Certifications": [
        r"certifications?", r"certificates?", r"credentials?",
        r"professional\s+certifications?",
    ],
    "Summary": [
        r"(?:professional\s+)?summary", r"objective", r"career\s+objective",
        r"profile", r"about\s+me",
    ],
    "Education": [
        r"education(?:al)?\s*(?:background|qualifications?)?",
        r"academic\s+(?:background|qualifications?|history)?",
        r"degrees?", r"qualifications?",
    ],
}


def build_section_regexes(patterns: Dict[str, List[str]]):
    standalone = {}
    inline = {}
    for section, pats in patterns.items():
        combined = "|".join(f"(?:{p})" for p in pats)
        standalone[section] = re.compile(
            rf"^(?:{combined})\s*[:\-\u2013]?\s*$",
            re.IGNORECASE | re.MULTILINE,
        )
        inline[section] = re.compile(
            rf"^(?:{combined})\s*[:\-\u2013]",
            re.IGNORECASE | re.MULTILINE,
        )
    return standalone, inline


STANDALONE_SECTION_RE, INLINE_SECTION_RE = build_section_regexes(SECTION_PATTERNS)
