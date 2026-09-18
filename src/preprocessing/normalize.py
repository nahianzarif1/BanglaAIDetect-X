"""
src/preprocessing/normalize.py

Deterministic normalization: same raw input -> always the same output.
No randomness anywhere in this file.
"""

import re
import unicodedata

# Unicode range for Bangla script
_BANGLA_RANGE = re.compile(r"[\u0980-\u09FF]")
_LATIN_RANGE = re.compile(r"[A-Za-z]")

# Characters we normalize to a single canonical form
_PUNCT_MAP = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "…": "...", "–": "-", "—": "-",
}


def normalize_unicode(text: str) -> str:
    """NFC-normalize and strip control/zero-width characters."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", text)
    # remove zero-width joiner/non-joiner artifacts and other control chars
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\t")
    return text


def normalize_punctuation(text: str) -> str:
    for bad, good in _PUNCT_MAP.items():
        text = text.replace(bad, good)
    return text


def collapse_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_script(text: str) -> str:
    """
    Returns one of: 'bangla', 'banglish', 'mixed', 'other'.
    Heuristic: compare Bangla-script char count vs. Latin-alphabet char count.
    'banglish' = Romanized Bangla written mostly in Latin letters but is
    Bangla in meaning; this heuristic only flags scripts, not semantics,
    so treat 'banglish' here as "mostly Latin letters" and validate further
    upstream if needed.
    """
    bangla_chars = len(_BANGLA_RANGE.findall(text))
    latin_chars = len(_LATIN_RANGE.findall(text))
    total = bangla_chars + latin_chars
    if total == 0:
        return "other"
    bangla_ratio = bangla_chars / total
    if bangla_ratio >= 0.9:
        return "bangla"
    if bangla_ratio <= 0.1:
        return "banglish"
    return "mixed"


def normalize_text(text: str) -> str:
    """Full deterministic normalization pipeline used before storage."""
    text = normalize_unicode(text)
    text = normalize_punctuation(text)
    text = collapse_whitespace(text)
    return text


if __name__ == "__main__":
    sample = "  এটি   একটি   উদাহরণ...   বাক্য়!!  "
    cleaned = normalize_text(sample)
    print(repr(cleaned))
    print("script:", detect_script(cleaned))
