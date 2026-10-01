"""
src/inference.py

Single entry point for turning raw text into a structured prediction.
Wraps model loading + tokenization + inference in try/except so a bad
input never crashes the caller (e.g. the Streamlit app).
"""

import os
import yaml
import joblib

from src.preprocessing.normalize import normalize_text, detect_script


class InputValidationError(Exception):
    pass


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class BanglaAIDetector:
    """Loads the Level-1 TF-IDF + LogReg baseline. Swap in BanglaBERT the
    same way once src/models/banglabert.py has been trained and saved."""

    def __init__(self, cfg: dict = None):
        self.cfg = cfg or load_config()
        vec_path = os.path.join(self.cfg["paths"]["models"], "tfidf_vectorizer.joblib")
        clf_path = os.path.join(self.cfg["paths"]["models"], "baseline_logreg.joblib")

        if not (os.path.exists(vec_path) and os.path.exists(clf_path)):
            raise FileNotFoundError(
                "Trained baseline not found. Run `python -m src.models.baseline` first."
            )
        self.vectorizer = joblib.load(vec_path)
        self.clf = joblib.load(clf_path)
        self.min_tokens = self.cfg["dataset"]["min_tokens"]

    def _validate(self, text: str) -> None:
        if text is None or not isinstance(text, str) or not text.strip():
            raise InputValidationError("Input text is empty.")
        try:
            text.encode("utf-8")
        except UnicodeEncodeError as e:
            raise InputValidationError(f"Input is not valid UTF-8: {e}")
        if len(text.split()) < self.min_tokens:
            raise InputValidationError(
                f"Input has fewer than {self.min_tokens} tokens — result would be unreliable."
            )

    def predict(self, text: str) -> dict:
        try:
            self._validate(text)
        except InputValidationError as e:
            return {"error": str(e)}

        try:
            clean = normalize_text(text)
            script = detect_script(clean)
            vec = self.vectorizer.transform([clean])
            ai_probability = float(self.clf.predict_proba(vec)[0, 1])
            human_probability = 1.0 - ai_probability

            assert 0 <= ai_probability <= 1
            assert abs(ai_probability + human_probability - 1) < 1e-6

            threshold = self.cfg["inference"]["decision_threshold"]
            label = "ai" if ai_probability >= threshold else "human"

            return {
                "label": label,
                "ai_probability": round(ai_probability, 4),
                "human_probability": round(human_probability, 4),
                "detected_script": script,
                "note": "Decision-support estimate, not an infallible determination.",
            }
        except Exception as e:  # never let inference crash the caller silently
            return {"error": f"Inference failed: {e}"}


if __name__ == "__main__":
    detector = BanglaAIDetector()
    example = ("এখানে অন্তত পঞ্চাশটি শব্দসম্বলিত একটি নমুনা বাংলা অনুচ্ছেদ বসাতে হবে যাতে মডেলটি "
                "ন্যূনতম দৈর্ঘ্যের শর্ত পূরণ করে এবং সঠিকভাবে পূর্বাভাস দিতে পারে। "
                "এই অনুচ্ছেদটি শুধু পরীক্ষার উদ্দেশ্যে তৈরি করা হয়েছে এবং এতে বাস্তব কোনো তথ্য নেই। "
                "মডেলের ইনপুট বৈধতা যাচাইয়ের জন্য প্রয়োজনীয় ন্যূনতম শব্দসংখ্যা নিশ্চিত করতে এই বাক্যগুলো যোগ করা হয়েছে।")
    print(detector.predict(example))
