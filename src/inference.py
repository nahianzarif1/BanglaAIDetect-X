"""Validated Bangla inference for classical, transformer, and ensemble models."""
import math
import numpy as np
from src.config import load_config
from src.models.registry import load_registry, load_predictor
from src.preprocessing.normalize import normalize_text, detect_script, is_supported_bangla, script_counts
from src.features.stylometry import extract_style_features, style_explanations


class InputValidationError(ValueError):
    pass


def _confidence_band(ai_probability, uncertain_margin, threshold=.5):
    distance = abs(ai_probability-threshold)
    if distance < uncertain_margin:
        return "low"
    if distance < uncertain_margin*2:
        return "medium"
    return "high"


class BanglaAIDetector:
    def __init__(self, cfg=None, model_id=None):
        self.cfg = cfg or load_config()
        registry = load_registry(self.cfg)
        self.model_id = model_id or registry["default_model"]
        self.pipe = load_predictor(self.model_id, self.cfg)
        self.metadata = registry["models"][self.model_id]
        self.threshold = self.metadata.get("threshold", self.cfg["inference"]["decision_threshold"])
        self.min_tokens = self.cfg["dataset"]["min_tokens"]
        self.min_bangla_ratio = self.cfg["inference"].get("min_bangla_ratio", .75)

    def _validate(self, text):
        if not isinstance(text, str) or not text.strip():
            raise InputValidationError("ইনপুট খালি। বাংলা অনুচ্ছেদ লিখুন।")
        if len(text) > self.cfg["inference"].get("max_characters", 100000):
            raise InputValidationError("Text exceeds the 100,000-character limit. Split it into passages.")
        try:
            text.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise InputValidationError("Input contains invalid Unicode.") from exc
        clean = normalize_text(text)
        if not is_supported_bangla(clean, self.min_bangla_ratio):
            raise InputValidationError("এই সংস্করণ শুধু বাংলা লিপির লেখা যাচাই করে। ইংরেজি বা বাংলিশ সমর্থিত নয়।")
        if len(clean.split()) < self.min_tokens:
            raise InputValidationError(f"কমপক্ষে {self.min_tokens}টি শব্দ দিন। ছোট লেখায় স্কোর নির্ভরযোগ্য নয়।")
        return clean

    def _result(self, clean, probability, include_style=True):
        probability = float(probability)
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Model returned an invalid probability. Retrain the artifact.")
        margin = self.cfg["inference"].get("uncertain_margin", .08)
        separation = _confidence_band(probability, margin, self.threshold)
        label = "uncertain" if separation == "low" else ("ai" if probability >= self.threshold else "human")
        result = {"label": label, "ai_probability": round(probability, 6), "human_probability": round(1-probability, 6),
            "model": self.model_id, "model_name": self.metadata["name"], "model_kind": self.metadata["kind"],
            "threshold": self.threshold, "confidence": separation, "score_separation": separation,
            "detected_script": detect_script(clean), "bangla_ratio": round(script_counts(clean)["bangla_ratio"], 3),
            "token_count": len(clean.split()), "style": {}, "cues": [],
            "note": "This score is a model estimate of authorship, not the percentage of words written by AI. "
                    "Score separation measures distance from the validation-selected threshold, not statistical certainty. "
                    "Human review and independent evidence are required for academic-integrity decisions."}
        if include_style:
            result["style"] = {key: float(value) for key, value in extract_style_features(clean).items()}
            result["cues"] = style_explanations(clean)
        if len(clean.split()) < 80:
            result["note"] += " Short passages have limited evidence."
        if hasattr(self.pipe, "truncation_count"):
            truncated = self.pipe.truncation_count(clean)
            result["truncated_tokens"] = truncated
            if truncated:
                result["note"] += f" This transformer used the first {self.pipe.metadata['max_length']} subword tokens; {truncated} tokens were truncated."
        if self.metadata["kind"] == "ensemble":
            result["note"] += " Transformer members, when present, use prefix truncation."
        return result

    def predict(self, text):
        return self.predict_many([text], include_style=True)[0]

    def predict_many(self, texts, include_style=False):
        texts = list(texts)
        if len(texts) > 1000:
            raise InputValidationError("Batch limit is 1,000 passages.")
        results, clean_texts, positions = [None]*len(texts), [], []
        for index, text in enumerate(texts):
            try:
                clean_texts.append(self._validate(text))
                positions.append(index)
            except InputValidationError as exc:
                results[index] = {"error": str(exc), "model": self.model_id}
        if clean_texts:
            probabilities = self.pipe.predict_proba(clean_texts)[:, 1]
            for index, clean, p in zip(positions, clean_texts, probabilities, strict=True):
                results[index] = self._result(clean, p, include_style)
        return results


if __name__ == "__main__":
    from app.samples import DEMOS
    detector = BanglaAIDetector()
    print("Loaded", detector.model_id)
    for sample in DEMOS:
        print(sample["title"], detector.predict(sample["text"]))
