from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from spacy.matcher import PhraseMatcher

from src.skills.skill_taxonomy import SKILL_TAXONOMY
from src.skills.skill_constants import (
    SECTION_EVIDENCE_WEIGHT,
    STANDALONE_SECTION_RE,
    INLINE_SECTION_RE,
)
from src.skills.skill_models import ExtractedSkill
from src.skills.skill_utils import (
    normalize_text,
    normalize_for_lookup,
    split_compound_phrase,
    is_noise_phrase,
)


class SectionParser:
    def parse(self, text: str) -> List[Tuple[str, str]]:
        text = normalize_text(text)
        boundaries: List[Tuple[int, str]] = []

        for line_match in re.finditer(r"^[^\n]{2,60}$", text, re.MULTILINE):
            line = line_match.group().strip()
            for section_name, pattern in STANDALONE_SECTION_RE.items():
                if pattern.match(line):
                    boundaries.append((line_match.start(), section_name))
                    break

        for section_name, pattern in INLINE_SECTION_RE.items():
            for m in pattern.finditer(text):
                if not any(abs(m.start() - b[0]) < 5 for b in boundaries):
                    boundaries.append((m.start(), section_name))

        if not boundaries:
            return [("Unknown", text)]

        boundaries.sort(key=lambda item: item[0])

        sections: List[Tuple[str, str]] = []
        for i, (start, name) in enumerate(boundaries):
            end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
            chunk = text[start:end].strip()
            chunk_lines = chunk.split("\n")[1:]  # drop the heading line itself
            sections.append((name, "\n".join(chunk_lines).strip()))

        if boundaries[0][0] > 0:
            pre_text = text[: boundaries[0][0]].strip()
            if pre_text:
                sections.insert(0, ("Unknown", pre_text))

        return sections

    def get_section_for_text(self, sections: List[Tuple[str, str]], target: str) -> str:
        for section_name, section_text in sections:
            if target.lower() in section_text.lower():
                return section_name
        return "Unknown"


_EXTRACTOR_CACHE: Dict[str, "HybridSkillExtractor"] = {}


def get_extractor(
    spacy_model: str = "en_core_web_sm",
    embedding_model_name: str = "all-MiniLM-L6-v2",
    semantic_threshold: float = 0.82,
) -> "HybridSkillExtractor":
    cache_key = f"{spacy_model}|{embedding_model_name}|{semantic_threshold}"
    if cache_key not in _EXTRACTOR_CACHE:
        _EXTRACTOR_CACHE[cache_key] = HybridSkillExtractor(
            spacy_model=spacy_model,
            embedding_model_name=embedding_model_name,
            semantic_threshold=semantic_threshold,
        )
    return _EXTRACTOR_CACHE[cache_key]


class HybridSkillExtractor:
    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        semantic_threshold: float = 0.82,
    ):
        self.nlp = spacy.load(spacy_model)
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.semantic_threshold = semantic_threshold

        self.alias_to_canonical: Dict[str, str] = {}
        self.canonical_skills: List[str] = []
        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self._section_parser = SectionParser()

        self._build_taxonomy()
        self._prepare_embeddings()

    def _build_taxonomy(self) -> None:
        patterns = []
        for canonical, metadata in SKILL_TAXONOMY.items():
            self.canonical_skills.append(canonical)
            all_names = {canonical, *metadata.get("aliases", [])}
            for name in all_names:
                normalized = normalize_for_lookup(name)
                if normalized:
                    self.alias_to_canonical[normalized] = canonical
                    patterns.append(self.nlp.make_doc(name))
        self.matcher.add("TECH_SKILL", patterns)

    def _prepare_embeddings(self) -> None:
        descriptions = [
            f"{skill}, a technical skill in {SKILL_TAXONOMY[skill]['category']}"
            for skill in self.canonical_skills
        ]
        self.taxonomy_embeddings = self.embedding_model.encode(
            descriptions,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def _make_skill(
        self,
        canonical: str,
        source_text: str,
        match_type: str,
        confidence: float,
        section: str = "Unknown",
    ) -> ExtractedSkill:
        weight = SECTION_EVIDENCE_WEIGHT.get(section, SECTION_EVIDENCE_WEIGHT["Unknown"])
        return ExtractedSkill(
            skill=canonical,
            category=SKILL_TAXONOMY[canonical]["category"],
            match_type=match_type,
            source_text=source_text.strip(),
            source_section=section,
            confidence=round(float(confidence), 3),
            evidence_weight=weight,
            occurrence_count=1,
            sections_found=[section],
        )

    def _exact_match(self, text: str, section: str) -> List[ExtractedSkill]:
        doc = self.nlp(text)
        results: List[ExtractedSkill] = []
        for _, start, end in self.matcher(doc):
            span = doc[start:end]
            normalized = normalize_for_lookup(span.text)
            canonical = self.alias_to_canonical.get(normalized)
            if not canonical:
                continue
            is_exact = normalize_for_lookup(canonical) == normalized
            mtype = "taxonomy_exact" if is_exact else "alias_exact"
            results.append(self._make_skill(canonical, span.text, mtype, 1.0, section))
        return results

    def _candidate_phrases(self, text: str) -> List[str]:
        doc = self.nlp(text)
        candidates: Set[str] = set()

        for ent in doc.ents:
            candidates.update(split_compound_phrase(ent.text))

        for chunk in doc.noun_chunks:
            cleaned = re.sub(
                r"^(?:experience|knowledge|proficiency|familiarity|"
                r"understanding|expertise)\s+(?:with|in|of)\s+",
                "",
                chunk.text.strip(),
                flags=re.IGNORECASE,
            )
            candidates.update(split_compound_phrase(cleaned))

        return sorted(
            {c.strip(" .,:;-") for c in candidates if not is_noise_phrase(c.strip(" .,:;-"))},
            key=lambda x: (-len(x), x.lower()),
        )

    def _semantic_match(self, candidate: str, section: str) -> Optional[ExtractedSkill]:
        if is_noise_phrase(candidate):
            return None

        normalized = normalize_for_lookup(candidate)

        if normalized in self.alias_to_canonical:
            canonical = self.alias_to_canonical[normalized]
            return self._make_skill(canonical, candidate, "alias_exact", 1.0, section)

        is_one_word = len(normalized.split()) == 1
        if is_one_word and len(normalized) <= 3:
            return None

        emb = self.embedding_model.encode(
            [f"{candidate}, a technical software skill"],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        sims = cosine_similarity(emb, self.taxonomy_embeddings)[0]
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])

        threshold = max(self.semantic_threshold, 0.85) if is_one_word else self.semantic_threshold
        if best_score < threshold:
            return None

        canonical = self.canonical_skills[best_idx]
        return self._make_skill(canonical, candidate, "semantic", best_score, section)

    @staticmethod
    def _merge_duplicates(skills: List[ExtractedSkill]) -> List[ExtractedSkill]:
        priority = {"taxonomy_exact": 3, "alias_exact": 2, "semantic": 1}
        unique: Dict[str, ExtractedSkill] = {}

        for skill in skills:
            key = skill.skill.lower()
            if key not in unique:
                unique[key] = skill
                continue
            existing = unique[key]
            existing.occurrence_count += 1
            if skill.source_section not in existing.sections_found:
                existing.sections_found.append(skill.source_section)
            new_rank = (priority.get(skill.match_type, 0), skill.evidence_weight, skill.confidence)
            old_rank = (priority.get(existing.match_type, 0), existing.evidence_weight, existing.confidence)
            if new_rank > old_rank:
                skill.occurrence_count = existing.occurrence_count
                skill.sections_found = existing.sections_found
                unique[key] = skill

        return sorted(unique.values(), key=lambda s: (s.category.lower(), s.skill.lower()))

    def extract_skills(
        self,
        text: str,
        use_semantic_fallback: bool = True,
        section_aware: bool = True,
    ) -> List[ExtractedSkill]:
        text = normalize_text(text)
        if not text:
            return []

        sections = self._section_parser.parse(text) if section_aware else [("Unknown", text)]

        results: List[ExtractedSkill] = []
        for section_name, chunk in sections:
            if chunk.strip():
                results.extend(self._exact_match(chunk, section_name))

        if use_semantic_fallback:
            found_sources = {normalize_for_lookup(s.source_text) for s in results}
            found_canonicals = {s.skill.lower() for s in results}

            for candidate in self._candidate_phrases(text):
                if normalize_for_lookup(candidate) in found_sources:
                    continue
                section_name = "Unknown"
                for sname, schunk in sections:
                    if candidate.lower() in schunk.lower():
                        section_name = sname
                        break
                sem = self._semantic_match(candidate, section_name)
                if sem and sem.skill.lower() not in found_canonicals:
                    results.append(sem)
                    found_canonicals.add(sem.skill.lower())

        return self._merge_duplicates(results)
