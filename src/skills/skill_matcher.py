from __future__ import annotations

import re
from typing import Dict, List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.skills.skill_taxonomy import SKILL_TAXONOMY
from src.skills.skill_constants import (
    REQUIREMENT_WEIGHTS,
    MATCH_SCORES,
    FINAL_SCORE_WEIGHTS,
    MANDATORY_KEYWORDS,
    PREFERRED_KEYWORDS,
    OPTIONAL_KEYWORDS,
)
from src.skills.skill_models import ExtractedSkill, JDSkill, SkillMatchResult
from src.skills.skill_extractor import HybridSkillExtractor
from src.skills.skill_utils import normalize_for_lookup


class JDRequirementClassifier:
    WINDOW_CHARS = 160

    def classify(self, jd_text: str, jd_skills: List[ExtractedSkill]) -> List[JDSkill]:
        text_lower = jd_text.lower()
        result: List[JDSkill] = []

        for skill in jd_skills:
            req_type = self._classify_single(text_lower, skill.skill)
            result.append(JDSkill(
                skill=skill.skill,
                category=skill.category,
                requirement_type=req_type,
                importance_weight=REQUIREMENT_WEIGHTS[req_type],
                source_text=skill.source_text,
            ))

        return result

    def _classify_single(self, text_lower: str, skill_name: str) -> str:
        skill_lower = skill_name.lower()
        positions = [m.start() for m in re.finditer(re.escape(skill_lower), text_lower)]

        if not positions:
            for alias in SKILL_TAXONOMY.get(skill_name, {}).get("aliases", []):
                for m in re.finditer(re.escape(alias.lower()), text_lower):
                    positions.append(m.start())
        if not positions:
            return "mandatory"

        best = "mandatory"
        order = ["mandatory", "preferred", "optional"]

        for pos in positions:
            window = text_lower[max(0, pos - self.WINDOW_CHARS): pos + self.WINDOW_CHARS]

            for kw in OPTIONAL_KEYWORDS:
                if kw in window:
                    best = "optional"
                    break
            if best == "optional":
                continue

            for kw in PREFERRED_KEYWORDS:
                if kw in window and order.index(best) > order.index("preferred"):
                    best = "preferred"
                    break

        return best


class SkillMatcher:
    RELATED_SEMANTIC_THRESHOLD = 0.88

    def __init__(self, extractor: HybridSkillExtractor):
        self.extractor = extractor

    def match(
        self,
        resume_skills: List[ExtractedSkill],
        jd_skills: List[JDSkill],
    ) -> List[SkillMatchResult]:
        resume_by_canonical = {s.skill.lower(): s for s in resume_skills}

        resume_by_alias: Dict[str, ExtractedSkill] = {}
        for rs in resume_skills:
            resume_by_alias[normalize_for_lookup(rs.skill)] = rs
            for alias in SKILL_TAXONOMY.get(rs.skill, {}).get("aliases", []):
                resume_by_alias[normalize_for_lookup(alias)] = rs

        return [
            self._match_one(jd_skill, resume_skills, resume_by_canonical, resume_by_alias)
            for jd_skill in jd_skills
        ]

    def _match_one(
        self,
        jd_skill: JDSkill,
        resume_skills: List[ExtractedSkill],
        resume_by_canonical: Dict[str, ExtractedSkill],
        resume_by_alias: Dict[str, ExtractedSkill],
    ) -> SkillMatchResult:
        jd_lower = jd_skill.skill.lower()
        jd_normalized = normalize_for_lookup(jd_skill.skill)

        # Level 1: exact canonical name
        if jd_lower in resume_by_canonical:
            matched = resume_by_canonical[jd_lower]
            return SkillMatchResult(
                jd_skill=jd_skill.skill,
                requirement_type=jd_skill.requirement_type,
                status="exact_match",
                matched_resume_skill=matched.skill,
                related_resume_skills=[],
                resume_evidence=[matched.to_dict()],
                explanation=f"Exact match: '{matched.skill}' found in resume.",
            )

        # Level 2: alias match (resume alias → JD skill, or JD alias → resume skill)
        if jd_normalized in resume_by_alias:
            matched = resume_by_alias[jd_normalized]
            return SkillMatchResult(
                jd_skill=jd_skill.skill,
                requirement_type=jd_skill.requirement_type,
                status="alias_match",
                matched_resume_skill=matched.skill,
                related_resume_skills=[],
                resume_evidence=[matched.to_dict()],
                explanation=f"Alias match: '{matched.skill}' maps to '{jd_skill.skill}'.",
            )

        for alias in SKILL_TAXONOMY.get(jd_skill.skill, {}).get("aliases", []):
            alias_norm = normalize_for_lookup(alias)
            if alias_norm in resume_by_alias:
                matched = resume_by_alias[alias_norm]
                return SkillMatchResult(
                    jd_skill=jd_skill.skill,
                    requirement_type=jd_skill.requirement_type,
                    status="alias_match",
                    matched_resume_skill=matched.skill,
                    related_resume_skills=[],
                    resume_evidence=[matched.to_dict()],
                    explanation=f"Alias match via '{alias}': '{matched.skill}' satisfies '{jd_skill.skill}'.",
                )

        # Level 3: related skills from taxonomy (informational, not a full match)
        related = set(SKILL_TAXONOMY.get(jd_skill.skill, {}).get("related_skills", []))
        broader = SKILL_TAXONOMY.get(jd_skill.skill, {}).get("broader_skill", "")
        narrower = set(SKILL_TAXONOMY.get(jd_skill.skill, {}).get("narrower_skills", []))
        all_related = related | narrower
        if broader:
            all_related.add(broader)

        related_in_resume = [rs.skill for rs in resume_skills if rs.skill in all_related]

        # Level 4: semantic similarity (also informational, not a full match)
        semantic_related: List[str] = []
        if resume_skills:
            jd_emb = self.extractor.embedding_model.encode(
                [f"{jd_skill.skill}, a technical skill in {jd_skill.category}"],
                convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False,
            )
            resume_descs = [
                f"{rs.skill}, a technical skill in {rs.category}" for rs in resume_skills
            ]
            res_embs = self.extractor.embedding_model.encode(
                resume_descs,
                convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False,
            )
            sims = cosine_similarity(jd_emb, res_embs)[0]
            for idx, score in enumerate(sims):
                rs = resume_skills[idx]
                same_cat = rs.category == jd_skill.category
                compatible = self._compatible_categories(rs.category, jd_skill.category)
                if (same_cat or compatible) and score >= self.RELATED_SEMANTIC_THRESHOLD:
                    if rs.skill not in related_in_resume:
                        semantic_related.append(rs.skill)

        all_related_found = list(dict.fromkeys(related_in_resume + semantic_related))

        if all_related_found:
            return SkillMatchResult(
                jd_skill=jd_skill.skill,
                requirement_type=jd_skill.requirement_type,
                status="related_match",
                matched_resume_skill=None,
                related_resume_skills=all_related_found,
                resume_evidence=[],
                explanation=(
                    f"Related concepts found ({', '.join(all_related_found)}), "
                    f"but '{jd_skill.skill}' itself was not explicitly in the resume."
                ),
            )

        return SkillMatchResult(
            jd_skill=jd_skill.skill,
            requirement_type=jd_skill.requirement_type,
            status="missing",
            matched_resume_skill=None,
            related_resume_skills=[],
            resume_evidence=[],
            explanation=f"'{jd_skill.skill}' was not found in the resume.",
        )

    @staticmethod
    def _compatible_categories(cat_a: str, cat_b: str) -> bool:
        groups = [
            {"Deep Learning Framework", "AI/ML", "ML Framework", "NLP Framework",
             "Computer Vision Library", "NLP Library"},
            {"Backend Framework", "Backend Runtime", "Backend"},
            {"Database", "Query Language"},
            {"Frontend Framework", "Frontend Library", "Frontend"},
            {"Cloud", "DevOps"},
            {"Data Science Library", "Data Science", "Data Visualization", "Scientific Computing"},
            {"Programming Language", "Scripting"},
        ]
        return any(cat_a in group and cat_b in group for group in groups)


class WeightedScorer:
    def score(
        self,
        match_results: List[SkillMatchResult],
        resume_skills: List[ExtractedSkill],
    ) -> Dict:
        mandatory = [r for r in match_results if r.requirement_type == "mandatory"]
        preferred = [r for r in match_results if r.requirement_type == "preferred"]

        def coverage(results: List[SkillMatchResult]) -> float:
            if not results:
                return 0.0
            return sum(MATCH_SCORES.get(r.status, 0.0) for r in results) / len(results)

        mandatory_cov = coverage(mandatory)
        preferred_cov = coverage(preferred)

        matched_weights = [
            e["evidence_weight"]
            for r in match_results
            if r.status in ("exact_match", "alias_match")
            for e in r.resume_evidence
        ]
        evidence_strength = float(np.mean(matched_weights)) if matched_weights else 0.0

        project_sections = {"Work Experience", "Projects"}
        project_skills = [rs for rs in resume_skills if any(s in project_sections for s in rs.sections_found)]
        project_relevance = min(1.0, len(project_skills) / max(1, len(resume_skills)))

        edu_sections = {"Education", "Certifications"}
        edu_skills = [rs for rs in resume_skills if any(s in edu_sections for s in rs.sections_found)]
        edu_relevance = min(1.0, len(edu_skills) / max(1, len(resume_skills)))

        overall = (
            FINAL_SCORE_WEIGHTS["mandatory_skill_coverage"] * mandatory_cov
            + FINAL_SCORE_WEIGHTS["preferred_skill_coverage"] * preferred_cov
            + FINAL_SCORE_WEIGHTS["evidence_strength"] * evidence_strength
            + FINAL_SCORE_WEIGHTS["project_experience_relevance"] * project_relevance
            + FINAL_SCORE_WEIGHTS["education_or_certification_relevance"] * edu_relevance
        )

        exact_count = sum(1 for r in match_results if r.status == "exact_match")
        alias_count = sum(1 for r in match_results if r.status == "alias_match")
        related_count = sum(1 for r in match_results if r.status == "related_match")

        return {
            "mandatory_skill_coverage": round(mandatory_cov, 4),
            "preferred_skill_coverage": round(preferred_cov, 4),
            "evidence_strength": round(evidence_strength, 4),
            "project_experience_relevance": round(project_relevance, 4),
            "education_or_certification_relevance": round(edu_relevance, 4),
            "overall_skill_score": round(overall, 4),
            "exact_match_count": exact_count + alias_count,
            "alias_match_count": alias_count,
            "related_match_count": related_count,
            "missing_mandatory_count": sum(
                1 for r in mandatory if r.status in ("missing", "related_match")
            ),
            "missing_preferred_count": sum(
                1 for r in preferred if r.status in ("missing", "related_match")
            ),
        }
