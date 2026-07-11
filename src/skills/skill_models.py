from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional

from src.skills.skill_constants import SECTION_EVIDENCE_WEIGHT


@dataclass
class ExtractedSkill:
    skill: str
    category: str
    match_type: str          # taxonomy_exact | alias_exact | semantic
    source_text: str
    source_section: str = "Unknown"
    confidence: float = 1.0
    evidence_weight: float = SECTION_EVIDENCE_WEIGHT["Unknown"]
    occurrence_count: int = 1
    sections_found: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "skill": self.skill,
            "category": self.category,
            "match_type": self.match_type,
            "source_text": self.source_text,
            "source_section": self.source_section,
            "confidence": self.confidence,
            "evidence_weight": self.evidence_weight,
            "occurrence_count": self.occurrence_count,
            "sections_found": self.sections_found,
        }


@dataclass
class JDSkill:
    skill: str
    category: str
    requirement_type: str    # mandatory | preferred | optional
    importance_weight: float
    source_text: str

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SkillMatchResult:
    jd_skill: str
    requirement_type: str
    status: str              # exact_match | alias_match | related_match | missing
    matched_resume_skill: Optional[str]
    related_resume_skills: List[str]
    resume_evidence: List[Dict]
    explanation: str

    def to_dict(self) -> Dict:
        return asdict(self)
