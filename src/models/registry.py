"""Load only locally trained artifacts explicitly declared by an experiment."""
import json
import gc
from pathlib import Path
import joblib
import numpy as np
from src.config import load_config


def load_registry(cfg=None):
    cfg = cfg or load_config()
    path = Path(cfg["paths"]["models"]) / "registry.json"
    if not path.exists():
        return {"schema_version": 1, "models": {}, "default_model": None}
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("schema_version") != 1 or not isinstance(registry.get("models"), dict):
        raise ValueError("Unsupported model registry. Run the training pipeline again.")
    return registry


def available_models(cfg=None):
    cfg = cfg or load_config()
    registry = load_registry(cfg)
    result = {}
    root = Path(cfg["paths"]["models"]).resolve()
    for model_id, metadata in registry["models"].items():
        artifact = (root / metadata["artifact"]).resolve()
        if not artifact.is_relative_to(root):
            raise ValueError("Model artifact escapes the configured model directory.")
        result[model_id] = {**metadata, "available": artifact.exists()}
        if metadata["kind"].startswith("transformer_"):
            complete = (artifact/"detector.json").exists() and (artifact/"config.json").exists() and any(
                (artifact/name).exists() for name in ["model.safetensors", "pytorch_model.bin"])
            if metadata["kind"] == "transformer_embeddings":
                complete = complete and (artifact/"head.joblib").exists()
            result[model_id]["available"] = complete
    for model_id, metadata in result.items():
        if metadata["kind"] == "ensemble":
            weights = metadata.get("weights", {})
            metadata["available"] = metadata["available"] and bool(weights) and all(
                name in result and result[name]["available"] and result[name]["kind"] != "ensemble"
                for name, weight in weights.items() if weight > 1e-8)
    return result


class EnsemblePredictor:
    def __init__(self, metadata, cfg):
        weights = list(metadata["weights"].values())
        if not weights or not np.isfinite(weights).all() or min(weights) < 0 or not np.isclose(sum(weights), 1):
            raise ValueError("Ensemble weights must be finite, nonnegative, and sum to one.")
        registry = load_registry(cfg)
        if any(name not in registry["models"] or registry["models"][name]["kind"] == "ensemble" for name in metadata["weights"]):
            raise ValueError("Ensemble members must be independently trained models.")
        self.cfg = cfg
        self.weights = {name: weight for name, weight in metadata["weights"].items() if weight > 1e-8}
        # Keep the small classical models; load large encoders one at a time.
        self.members = {name: load_predictor(name, cfg) for name in self.weights if registry["models"][name]["kind"] == "sklearn"}

    def _member_probability(self, name, texts):
        if name in self.members:
            return self.members[name].predict_proba(texts)[:, 1]
        member = load_predictor(name, self.cfg)
        probability = member.predict_proba(texts)[:, 1]
        del member
        gc.collect()
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        return probability

    def predict_proba(self, texts):
        texts = list(texts)
        probabilities = sum(weight * self._member_probability(name, texts) for name, weight in self.weights.items())
        return np.column_stack([1-probabilities, probabilities])

    def compare(self, texts):
        texts = list(texts)
        return {name: self._member_probability(name, texts) for name in self.weights}


def load_predictor(model_id=None, cfg=None):
    cfg = cfg or load_config()
    registry = load_registry(cfg)
    model_id = model_id or registry["default_model"]
    if model_id not in registry["models"]:
        raise FileNotFoundError(f"No trained artifact for {model_id or 'any model'}. Run python run_pipeline.py.")
    metadata = registry["models"][model_id]
    path = (Path(cfg["paths"]["models"]) / metadata["artifact"]).resolve()
    if not path.is_relative_to(Path(cfg["paths"]["models"]).resolve()) or not path.exists():
        raise FileNotFoundError(f"Missing model artifact for {model_id}. Retrain this model.")
    if metadata["kind"] == "sklearn":
        return joblib.load(path)
    if metadata["kind"] == "ensemble":
        return EnsemblePredictor(metadata, cfg)
    if metadata["kind"] in {"transformer_embeddings", "transformer_finetuned"}:
        from src.models.transformer import TransformerPredictor
        return TransformerPredictor(path)
    raise ValueError(f"Unsupported model artifact kind: {metadata['kind']}")
