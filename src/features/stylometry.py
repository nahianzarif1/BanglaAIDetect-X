"""
Bangla stylometric / statistical features for AI vs human detection.

These are content-light on purpose: Wikipedia and ChatGPT can share a topic
(ঢাকা, মুক্তিযুদ্ধ, কৃষি) while differing in rhythm, hedging, discourse
markers, number density, and sentence burstiness.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

_SENT_SPLIT = re.compile(r"[।!?]+")
_WORD = re.compile(r"[^\s।!?.,;:\"'()\[\]{}]+")
_DIGIT = re.compile(r"[0-9০-৯]")
_YEAR = re.compile(r"(?:১[৬-৯]|২[০-১]|16|17|18|19|20)\d{2}")

AI_MARKERS = [
    "এছাড়াও", "এছাড়াও", "তাছাড়া", "তাছাড়া", "অন্যদিকে", "সর্বোপরি",
    "উল্লেখযোগ্যভাবে", "উল্লেখযোগ্য", "গুরুত্বপূর্ণ ভূমিকা", "অপরিহার্য",
    "সুতরাং", "অতএব", "এ প্রসঙ্গে", "পরিশেষে", "সামগ্রিকভাবে",
    "হলো একটি", "হল একটি", "হিসেবে পরিচিত", "দিন দিন বৃদ্ধি",
    "গুরুত্বপূর্ণ অবদান", "ইতিবাচক প্রভাব", "নেতিবাচক প্রভাব",
    "সঠিকভাবে মোকাবিলা", "পরিকল্পিত উদ্যোগ", "সচেতনতা বৃদ্ধি",
    "অগ্রাধিকারমূলক", "টেকসই উন্নয়ন", "টেকসই উন্নয়ন",
    "মূল চালিকাশক্তি", "অপার সম্ভাবনা", "গুরুত্বপূর্ণ বিষয়",
    "গুরুত্বপূর্ণ বিষয়", "এছাড়া", "এছাড়া",
    # Additional AI markers
    "এটি হলো", "এটি হল", "এটি একটি", "এটি এক",
    "যা প্রয়োজন", "যা দরকার", "যা আবশ্যক",
    "মূলত", "মূলতই", "মূলত:",
    "সাধারণত", "সাধারণতই", "সাধারণত:",
    "অনেক ক্ষেত্রে", "অনেক ক্ষেত্রেই", "বেশিরভাগ",
    "লক্ষ্যণীয়", "উল্লেখযোগ্য", "বিশেষভাবে",
    "এর ফলে", "এর মাধ্যমে", "এর মাধ্যমেই",
    "প্রতিনিয়ত", "প্রতিটি", "প্রত্যেকটি",
    "বিভিন্ন", "বিভিন্নভাবে", "বিভিন্ন ধরনের",
]

HUMAN_MARKERS = [
    "যদিও", "তবে", "কিন্তু", "আর ", "মনে হয়", "মনে হয়",
    "প্রায়", "প্রায়", "অনুযায়ী", "অনুযায়ী", "সালে", "সালের",
    "সূত্র", "মতে", "বলা হয়", "বলা হয়", "ধরা হয়", "ধরা হয়",
    "এখনও", "এখনো", "কখনো কখনো", "অনেক সময়", "অনেক সময়",
    # Additional human markers - more conversational and idiomatic
    "ভাবছিলাম", "গিয়েছিলাম", "এসেছিলাম", "বললাম",
    "দেখলাম", "পেলাম", "পারলাম", "হল",
    "করলাম", "খেলাম", "চললাম", "এলাম",
    "জানালেন", "বলেন", "মনে করেন", "দেখেন",
    "ফোন করে", "বাসায়", "বাইরে", "ভেতরে",
    "সকালে", "সন্ধ্যায়", "রাতে", "দুপুরে",
    "আজ", "কাল", "গতকাল", "আগামী",
    "এই", "ওই", "সেই", "কোন",
    "বলে", "জানায়", "করে", "নেয়",
    "হয়", "যায়", "দেয়", "পারে",
]

FEATURE_NAMES = [
    "n_tokens",
    "n_sentences",
    "avg_sent_len",
    "std_sent_len",
    "burstiness",
    "type_token_ratio",
    "hapax_ratio",
    "avg_word_len",
    "std_word_len",
    "digit_ratio",
    "year_count_norm",
    "punct_ratio",
    "paren_ratio",
    "comma_ratio",
    "ai_marker_density",
    "human_marker_density",
    "marker_balance",
    "repeated_bigram_ratio",
    "start_formulaic",
    "end_formulaic",
    "parallel_comma_lists",
    "unique_punct_types",
]


def _sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _words(text: str) -> list[str]:
    return _WORD.findall(text)


def _count_markers(text: str, markers: list[str]) -> int:
    return sum(text.count(m) for m in markers)


def extract_style_features(text: str) -> dict:
    text = text or ""
    sents = _sentences(text)
    words = _words(text)
    n_tok = max(len(words), 1)
    n_sent = max(len(sents), 1)
    sent_lens = np.array([len(_words(s)) for s in sents], dtype=float)
    word_lens = np.array([len(w) for w in words], dtype=float) if words else np.array([0.0])

    avg_sl = float(sent_lens.mean())
    std_sl = float(sent_lens.std()) if len(sent_lens) > 1 else 0.0
    burstiness = std_sl / avg_sl if avg_sl > 0 else 0.0

    counts = Counter(words)
    types = len(counts)
    hapax = sum(1 for _, c in counts.items() if c == 1)

    bigrams = list(zip(words, words[1:]))
    if bigrams:
        bc = Counter(bigrams)
        repeated_bigram_ratio = sum(v - 1 for v in bc.values() if v > 1) / len(bigrams)
    else:
        repeated_bigram_ratio = 0.0

    ai_m = _count_markers(text, AI_MARKERS)
    hu_m = _count_markers(text, HUMAN_MARKERS)

    first = sents[0] if sents else ""
    last = sents[-1] if sents else ""
    start_formulaic = 1.0 if re.search(r"হলো|হল একটি|হিসেবে পরিচিত|গুরুত্বপূর্ণ", first) else 0.0
    end_formulaic = 1.0 if re.search(r"সর্বোপরি|সুতরাং|অতএব|পরিশেষে|সামগ্রিকভাবে|অপরিহার্য", last) else 0.0

    parallel_comma_lists = len(re.findall(r"[^।]{0,40}, [^।]{0,40} এবং ", text))

    punct_chars = [ch for ch in text if ch in "।,;:!?()\"'-"]
    unique_punct = len(set(punct_chars))

    return {
        "n_tokens": float(len(words)),
        "n_sentences": float(len(sents)),
        "avg_sent_len": avg_sl,
        "std_sent_len": std_sl,
        "burstiness": burstiness,
        "type_token_ratio": types / n_tok,
        "hapax_ratio": hapax / n_tok,
        "avg_word_len": float(word_lens.mean()),
        "std_word_len": float(word_lens.std()) if len(word_lens) > 1 else 0.0,
        "digit_ratio": len(_DIGIT.findall(text)) / max(len(text), 1),
        "year_count_norm": len(_YEAR.findall(text)) / n_sent,
        "punct_ratio": len(punct_chars) / max(len(text), 1),
        "paren_ratio": (text.count("(") + text.count(")")) / n_tok,
        "comma_ratio": text.count(",") / n_sent,
        "ai_marker_density": ai_m / n_sent,
        "human_marker_density": hu_m / n_sent,
        "marker_balance": (ai_m - hu_m) / n_sent,
        "repeated_bigram_ratio": repeated_bigram_ratio,
        "start_formulaic": start_formulaic,
        "end_formulaic": end_formulaic,
        "parallel_comma_lists": float(parallel_comma_lists) / n_sent,
        "unique_punct_types": float(unique_punct),
    }


def style_vector(text: str) -> np.ndarray:
    feats = extract_style_features(text)
    return np.array([feats[name] for name in FEATURE_NAMES], dtype=float)


def style_explanations(text: str) -> list[dict]:
    """Human-readable cues for the demo UI (not the model's only signal)."""
    f = extract_style_features(text)
    cues = []

    if f["ai_marker_density"] >= 0.35:
        cues.append({
            "side": "ai",
            "title": "AI-style discourse markers",
            "detail": "Frequent connectors such as এছাড়াও, অন্যদিকে, সর্বোপরি — common in ChatGPT-like Bangla.",
        })
    if f["start_formulaic"] or f["end_formulaic"]:
        cues.append({
            "side": "ai",
            "title": "Template opening / closing",
            "detail": "Definition-style start (হলো একটি…) or wrap-up (সর্বোপরি / সুতরাং) typical of LLM essays.",
        })
    if f["burstiness"] < 0.35 and f["n_sentences"] >= 4:
        cues.append({
            "side": "ai",
            "title": "Uniform sentence rhythm",
            "detail": "Sentence lengths are unusually even. Human Wikipedia/news writing is burstier.",
        })
    if f["parallel_comma_lists"] > 0.15:
        cues.append({
            "side": "ai",
            "title": "Parallel lists",
            "detail": "Repeated “X, Y এবং Z” scaffolding is over-used by instruction-tuned models.",
        })

    if f["digit_ratio"] >= 0.012 or f["year_count_norm"] >= 0.25:
        cues.append({
            "side": "human",
            "title": "Specific numbers / years",
            "detail": "Concrete dates and figures are denser in encyclopedic human text than in generic LLM summaries.",
        })
    if f["human_marker_density"] >= 0.25:
        cues.append({
            "side": "human",
            "title": "Human discourse (যদিও / তবে / অনুযায়ী)",
            "detail": "Contrast and attribution markers show up more in edited human prose.",
        })
    if f["burstiness"] >= 0.45:
        cues.append({
            "side": "human",
            "title": "Burstiness",
            "detail": "Mix of short and long sentences — typical of Wikipedia and news desks, not of one-shot LLM drafts.",
        })
    if f["paren_ratio"] >= 0.02:
        cues.append({
            "side": "human",
            "title": "Parenthetical asides",
            "detail": "Editors pack extra facts in parentheses; chat models flatten them.",
        })

    if not cues:
        cues.append({
            "side": "neutral",
            "title": "Weak style signal",
            "detail": "Stylometry is inconclusive on this excerpt. Treat the score as low-confidence.",
        })
    return cues


class StylometryTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.vstack([style_vector(t) for t in X])

    def get_feature_names_out(self, input_features=None):
        return np.array(FEATURE_NAMES)
