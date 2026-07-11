from __future__ import annotations

from typing import Dict, List, Optional

from src.skills.skill_extractor import HybridSkillExtractor, get_extractor
from src.skills.skill_matcher import JDRequirementClassifier, SkillMatcher, WeightedScorer

_extractor: Optional[HybridSkillExtractor] = None
_classifier: Optional[JDRequirementClassifier] = None
_matcher: Optional[SkillMatcher] = None
_scorer: Optional[WeightedScorer] = None


def _ensure_loaded() -> None:
    global _extractor, _classifier, _matcher, _scorer
    if _extractor is None:
        _extractor = get_extractor(semantic_threshold=0.82)
        _classifier = JDRequirementClassifier()
        _matcher = SkillMatcher(_extractor)
        _scorer = WeightedScorer()


def extract_skills(text: str) -> List[Dict]:
    _ensure_loaded()
    return [s.to_dict() for s in _extractor.extract_skills(text)]


def find_missing_skills(resume_text: str, jd_text: str) -> Dict:
    _ensure_loaded()

    resume_skills = _extractor.extract_skills(resume_text, use_semantic_fallback=True, section_aware=True)
    jd_raw_skills = _extractor.extract_skills(jd_text, use_semantic_fallback=False, section_aware=False)
    jd_skills = _classifier.classify(jd_text, jd_raw_skills)
    match_results = _matcher.match(resume_skills, jd_skills)
    summary = _scorer.score(match_results, resume_skills)

    matched = [r for r in match_results if r.status in ("exact_match", "alias_match")]
    partially_related = [r for r in match_results if r.status == "related_match"]

    missing_mandatory = [
        r.jd_skill for r in match_results
        if r.requirement_type == "mandatory" and r.status in ("missing", "related_match")
    ]
    missing_preferred = [
        r.jd_skill for r in match_results
        if r.requirement_type == "preferred" and r.status in ("missing", "related_match")
    ]

    resume_skill_names = {rs.skill for rs in resume_skills}
    jd_skill_names = {js.skill for js in jd_skills}

    return {
        "resume_skills": [rs.to_dict() for rs in resume_skills],
        "jd_required_skills": [js.to_dict() for js in jd_skills],
        "matched_skills": [r.to_dict() for r in matched],
        "partially_related_skills": [r.to_dict() for r in partially_related],
        "missing_mandatory_skills": sorted(missing_mandatory),
        "missing_preferred_skills": sorted(missing_preferred),
        "additional_resume_skills": sorted(resume_skill_names - jd_skill_names),
        "missing_skills": sorted(set(missing_mandatory + missing_preferred)),
        "skill_analysis": [r.to_dict() for r in match_results],
        "summary": summary,
    }
