"""
src/inference.py

Turns Bangla text into a structured, decision-support prediction.
English / Banglish are rejected — they were never evaluated.
"""

import os
import yaml
import joblib

from src.preprocessing.normalize import (
    normalize_text, detect_script, is_supported_bangla, script_counts,
)
from src.features.stylometry import extract_style_features, style_explanations


class InputValidationError(Exception):
    pass


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _confidence_band(ai_probability: float, uncertain_margin: float) -> str:
    dist = abs(ai_probability - 0.5)
    if dist < uncertain_margin:
        return "low"
    if dist < uncertain_margin * 2:
        return "medium"
    return "high"


class BanglaAIDetector:
    """Fusion detector (TF-IDF + stylometry). BanglaBERT can replace the
    joblib pipeline later without changing this return schema."""

    def __init__(self, cfg: dict = None):
        self.cfg = cfg or load_config()
        models_dir = self.cfg["paths"]["models"]
        candidates = [
            os.path.join(models_dir, "fusion_detector.joblib"),
            os.path.join(models_dir, "baseline_logreg.joblib"),
        ]
        path = next((p for p in candidates if os.path.exists(p)), None)
        if path is None:
            raise FileNotFoundError(
                "Trained detector not found. Run `python -m src.models.baseline` first."
            )
        self.pipe = joblib.load(path)
        self.min_tokens = self.cfg["dataset"]["min_tokens"]
        self.min_bangla_ratio = self.cfg["inference"].get("min_bangla_ratio", 0.75)

    def _validate(self, text: str) -> str:
        if text is None or not isinstance(text, str) or not text.strip():
            raise InputValidationError("ইনপুট খালি। বাংলা অনুচ্ছেদ লিখুন।")
        try:
            text.encode("utf-8")
        except UnicodeEncodeError as e:
            raise InputValidationError(f"ইনপুট বৈধ UTF-8 নয়: {e}")

        clean = normalize_text(text)
        if not is_supported_bangla(clean, self.min_bangla_ratio):
            script = detect_script(clean)
            raise InputValidationError(
                "এই সংস্করণ শুধু বাংলা লিপির লেখা যাচাই করে। "
                f"ইংরেজি বা বাংলিশ সমর্থিত নয় (detected: {script})।"
            )
        if len(clean.split()) < self.min_tokens:
            raise InputValidationError(
                f"কমপক্ষে {self.min_tokens}টি শব্দ দিন — ছোট উদ্ধৃতিতে স্কোর নির্ভরযোগ্য নয়।"
            )
        return clean

    def predict(self, text: str) -> dict:
        try:
            clean = self._validate(text)
        except InputValidationError as e:
            return {"error": str(e)}

        try:
            script = detect_script(clean)
            counts = script_counts(clean)
            ai_probability = float(self.pipe.predict_proba([clean])[0, 1])
            human_probability = 1.0 - ai_probability
            assert 0 <= ai_probability <= 1
            assert abs(ai_probability + human_probability - 1) < 1e-6

            threshold = self.cfg["inference"]["decision_threshold"]
            margin = self.cfg["inference"].get("uncertain_margin", 0.12)
            confidence = _confidence_band(ai_probability, margin)
            if confidence == "low":
                label = "uncertain"
            else:
                label = "ai" if ai_probability >= threshold else "human"

            style = extract_style_features(clean)
            cues = style_explanations(clean)
            tokens = len(clean.split())

            reliability_note = (
                "Decision-support estimate for Bangla text only — not an academic-integrity verdict. "
                "Unlike Turnitin, this lab model has no billion-document index and no paid LLM watermark API."
            )
            if tokens < 80:
                reliability_note += " Short passages are easier to mis-score; paste a full paragraph."

            return {
                "label": label,
                "ai_probability": round(ai_probability, 4),
                "human_probability": round(human_probability, 4),
                "confidence": confidence,
                "detected_script": script,
                "bangla_ratio": round(counts["bangla_ratio"], 3),
                "token_count": tokens,
                "style": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in style.items()},
                "cues": cues,
                "note": reliability_note,
            }
        except Exception as e:
            return {"error": f"Inference failed: {e}"}


if __name__ == "__main__":
    detector = BanglaAIDetector()
    wiki = (
        "ঢাকা বাংলাদেশের রাজধানী ও বৃহত্তম শহর। ২০২২ সালের আদমশুমারি অনুযায়ী সিটি কর্পোরেশন "
        "এলাকার জনসংখ্যা প্রায় এক কোটি। শহরটি বুড়িগঙ্গা নদীর তীরে গড়ে উঠেছে। মুঘল আমলে এটি "
        "প্রাদেশিক কেন্দ্র ছিল, যদিও আধুনিক নগরায়ণ স্বাধীনতার পর তীব্র হয়। যানজট ও জলাবদ্ধতা "
        "নিত্য সমস্যা। পুরান ঢাকার গলি ও নতুন ঢাকার পরিকল্পিত এলাকা পাশাপাশি টিকে আছে। "
        "মেট্রোরেল চালু হলেও সারা মহানগরীর চাপ কমেনি বলে নগর পরিকল্পকরা মনে করেন।"
    )
    gpt = (
        "ঢাকা হলো বাংলাদেশের একটি গুরুত্বপূর্ণ শহর। এটি সমাজ, অর্থনীতি এবং দৈনন্দিন জীবনে "
        "গুরুত্বপূর্ণ ভূমিকা পালন করে। এছাড়াও শিক্ষা, সংস্কৃতি এবং প্রশাসনে এর ইতিবাচক প্রভাব দেখা যায়। "
        "অন্যদিকে নগরায়ণ সম্পর্কিত চ্যালেঞ্জগুলো সঠিকভাবে মোকাবিলা করা প্রয়োজন। "
        "সর্বোপরি ঢাকা বিষয়ে সচেতনতা বৃদ্ধি এবং টেকসই উন্নয়ন নিশ্চিত করা অপরিহার্য।"
    )
    print("WIKI-LIKE:", detector.predict(wiki))
    print("CHATGPT-LIKE:", detector.predict(gpt))
