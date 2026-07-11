from __future__ import annotations

import re
from typing import List

from src.skills.skill_constants import NOISE_TERMS, ROLE_PATTERNS


def normalize_text(text: str) -> str:
    if not text:
        return ""
    replacements = {
        "\u2013": "-", "\u2014": "-", "\u2022": "\n",
        "\u00b7": "\n", "\u2192": " ", "\u00a0": " ",
        "\u2019": "'", "\u201c": '"', "\u201d": '"',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_for_lookup(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("&", " and ").replace("_", " ")
    text = re.sub(r"[^a-z0-9+#./\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,-/")


def split_compound_phrase(text: str) -> List[str]:
    if not text:
        return []
    text = re.sub(r"\s+(?:and|or)\s+", " | ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*[,;|]\s*", " | ", text)
    if not re.search(r"https?://", text, flags=re.IGNORECASE):
        text = re.sub(r"\s*/\s*", " | ", text)
    return [
        part.strip(" .:-")
        for part in text.split("|")
        if part.strip(" .:-")
    ]


def is_noise_phrase(phrase: str) -> bool:
    normalized = normalize_for_lookup(phrase)
    if not normalized:
        return True
    if normalized in NOISE_TERMS:
        return True
    if len(normalized) < 2 or len(normalized) > 70:
        return True
    if normalized.isdigit():
        return True
    if not re.search(r"[a-zA-Z+#]", normalized):
        return True
    if "\n" in phrase:
        return True
    for pattern in ROLE_PATTERNS:
        if re.fullmatch(pattern, normalized, flags=re.IGNORECASE):
            return True
    return False
