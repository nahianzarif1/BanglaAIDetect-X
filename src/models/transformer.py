"""Supervised frozen-encoder and full fine-tuning paths for Bangla encoders.

Imports are lazy so the classical app works without the optional dependencies.
Only artifacts trained by this pipeline can be used for authorship predictions.
"""
import json
import math
import os
import random
import shutil
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.models.catalog import TRANSFORMERS
from src.preprocessing.normalize import normalize_text
from src.research import binary_metrics, file_hash, write_json


def dependencies():
    try:
        import torch
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Transformer dependencies unavailable. Install requirements-transformers.txt.") from exc
    return torch, AutoModel, AutoModelForSequenceClassification, AutoTokenizer


def device_for(torch, requested="auto"):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def prepare_texts(texts, normalizer_kind):
    if normalizer_kind == "buet":
        try:
            from normalizer import normalize
        except ImportError as exc:
            raise RuntimeError("BanglaBERT requires BUET's official normalizer. Install requirements-transformers.txt.") from exc
        return [normalize(normalize_text(t)) for t in texts]
    return [normalize_text(t) for t in texts]


def encode_batches(texts, tokenizer, model, device, max_length, batch_size, normalizer_kind, embeddings=True):
    torch, *_ = dependencies()
    texts = prepare_texts(texts, normalizer_kind)
    model.eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(texts[start:start+batch_size], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            batch = {key: value.to(device) for key, value in batch.items()}
            prediction = model(**batch)
            if embeddings:
                hidden = prediction.last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                value = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            else:
                value = torch.softmax(prediction.logits, dim=1)
            output.append(value.cpu().numpy())
            if len(texts) > 500 and (start // batch_size) % 50 == 0:
                print(f"Encoder progress: {min(start+batch_size, len(texts))}/{len(texts)} documents on {device}", flush=True)
    return np.concatenate(output) if output else np.empty((0, 2))


class TransformerPredictor:
    def __init__(self, path, device="auto"):
        self.path = Path(path)
        self.metadata = json.loads((self.path / "detector.json").read_text(encoding="utf-8"))
        torch, AutoModel, AutoClassifier, AutoTokenizer = dependencies()
        self.device = device_for(torch, device)
        # These BERT/ELECTRA/RoBERTa tokenizers do not use Mistral's regex.
        # Explicit False avoids a false warning for local checkpoints in 4.57.x.
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True, fix_mistral_regex=False)
        factory = AutoModel if self.metadata["mode"] == "embeddings" else AutoClassifier
        self.model = factory.from_pretrained(self.path, local_files_only=True).to(self.device)
        self.head = joblib.load(self.path / "head.joblib") if self.metadata["mode"] == "embeddings" else None

    def predict_proba(self, texts):
        texts = list(texts)
        if not texts:
            return np.empty((0, 2))
        vectors = encode_batches(texts, self.tokenizer, self.model, self.device,
            self.metadata["max_length"], self.metadata["batch_size"], self.metadata["normalizer"], self.head is not None)
        if self.head is not None:
            return self.head.predict_proba(vectors)
        return vectors

    def truncation_count(self, text):
        tokens = self.tokenizer(prepare_texts([text], self.metadata["normalizer"])[0], truncation=False)["input_ids"]
        return max(0, len(tokens) - self.metadata["max_length"])


def save_frozen_encoder(model, output, checkpoint, cache):
    """Reuse an unchanged downloaded encoder via a hard link, avoiding a second
    physical weight copy. Removing the download cache leaves the artifact intact.
    Local user checkpoints are copied normally and are never moved or altered.
    """
    from transformers.utils.hub import cached_file
    for filename in ["model.safetensors", "pytorch_model.bin"]:
        candidate = cached_file(checkpoint, filename, cache_dir=cache, local_files_only=True,
            _raise_exceptions_for_missing_entries=False)
        if candidate is not None:
            candidate = Path(candidate).resolve()
            if candidate.is_relative_to(Path(cache).resolve()):
                try:
                    os.link(candidate, Path(output)/filename)
                except OSError:
                    break
                model.config.save_pretrained(output)
                return "hard-linked unchanged pretrained weights; supervised head stored separately"
    model.save_pretrained(output, safe_serialization=True)
    return "self-contained encoder weight copy"


def train_transformer(model_id, train, validation, cfg, output, mode="embeddings"):
    torch, AutoModel, AutoClassifier, AutoTokenizer = dependencies()
    if mode not in {"embeddings", "finetune"}:
        raise ValueError("Transformer mode must be embeddings or finetune.")
    spec = TRANSFORMERS[model_id]
    options = cfg["transformers"]
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["seed"])
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = device_for(torch, options.get("device", "auto"))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    minimum_gb = spec.get("minimum_free_gb", 1.5)
    if mode == "embeddings":
        minimum_gb = minimum_gb/2 + .2
    required = minimum_gb * 1024**3
    if shutil.disk_usage(output).free < required:
        raise RuntimeError(f"Insufficient storage for {model_id}. Free at least {minimum_gb:.1f} GB before starting this checkpoint.")
    cache = str(Path(cfg["paths"]["models"]).parent / ".cache/huggingface")
    tokenizer = AutoTokenizer.from_pretrained(spec["checkpoint"], cache_dir=cache, fix_mistral_regex=False)
    factory = AutoModel if mode == "embeddings" else AutoClassifier
    kwargs = {"num_labels": 2, "id2label": {0: "human", 1: "ai"}, "label2id": {"human": 0, "ai": 1}} if mode == "finetune" else {}
    model = factory.from_pretrained(spec["checkpoint"], cache_dir=cache, **kwargs).to(device)
    metadata = {**spec, "model_id": model_id, "mode": mode, "normalizer": spec["normalizer"],
        "max_length": options["max_length"], "batch_size": options["batch_size"], "seed": cfg["seed"],
        "checkpoint_revision": getattr(model.config, "_commit_hash", None), "training_rows": len(train),
        "device": str(device), "long_text_policy": "prefix truncation in both training and inference"}
    metadata["split_hashes"] = {name: file_hash(cfg["paths"][name]) for name in ["train", "validation", "test"] if Path(cfg["paths"][name]).exists()}
    if mode == "embeddings":
        metadata["batch_size"] = options.get("embedding_batch_size", 16)
        print(f"{model_id}: extracting frozen contextual embeddings on {device} for {len(train)} training documents", flush=True)
        vectors = encode_batches(train.text, tokenizer, model, device, metadata["max_length"], metadata["batch_size"], spec["normalizer"])
        head = make_pipeline(StandardScaler(), LogisticRegression(C=options.get("head_C", .1),
            class_weight="balanced", max_iter=2000, random_state=cfg["seed"]))
        head.fit(vectors, train.label.eq("ai").astype(int))
        joblib.dump(head, output / "head.joblib")
        metadata["weight_storage"] = save_frozen_encoder(model, output, spec["checkpoint"], cache)
        metadata["training_method"] = "frozen pretrained encoder, mean pooling, supervised logistic head"
    else:
        _fine_tune(model, tokenizer, train, validation, cfg, metadata, output, device)
        metadata["training_method"] = "supervised end-to-end sequence classification"
    tokenizer.save_pretrained(output)
    write_json(output / "detector.json", metadata)
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    return TransformerPredictor(output, str(device)), metadata


def _fine_tune(model, tokenizer, train, validation, cfg, metadata, output, device):
    torch, *_ = dependencies()
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding, get_linear_schedule_with_warmup
    options = cfg["transformers"]
    texts = prepare_texts(train.text, metadata["normalizer"])
    encoded = tokenizer(texts, truncation=True, max_length=metadata["max_length"])
    labels = train.label.eq("ai").astype(int).tolist()
    records = [{**{key: value[i] for key, value in encoded.items()}, "labels": labels[i]} for i in range(len(labels))]
    loader = DataLoader(records, batch_size=metadata["batch_size"], shuffle=True,
        generator=torch.Generator().manual_seed(cfg["seed"]), collate_fn=DataCollatorWithPadding(tokenizer))
    accumulation = options.get("gradient_accumulation_steps", 1)
    epochs = options["epochs"]
    steps = math.ceil(len(loader)/accumulation) * epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(options["learning_rate"]), weight_decay=options["weight_decay"])
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * options["warmup_ratio"]), steps)
    counts = np.bincount(labels, minlength=2)
    class_weights = torch.tensor(len(labels)/(2*counts), dtype=torch.float32, device=device)
    best, history = -1., []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        epoch_loss = 0.
        for index, batch in enumerate(loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            targets = batch.pop("labels")
            logits = model(**batch).logits
            loss = torch.nn.functional.cross_entropy(logits, targets, weight=class_weights)
            epoch_loss += loss.item()
            # Scale the incomplete final accumulation block by its actual size.
            block_start = (index // accumulation) * accumulation
            block_size = min(accumulation, len(loader)-block_start)
            (loss/block_size).backward()
            if (index+1) % accumulation == 0 or index+1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        probabilities = encode_batches(validation.text, tokenizer, model, device, metadata["max_length"], metadata["batch_size"], metadata["normalizer"], False)[:, 1]
        metrics = binary_metrics(validation.label.eq("ai").astype(int), probabilities)
        history.append({"epoch": epoch+1, "train_loss": epoch_loss/len(loader), "validation_macro_f1": metrics["macro_f1"]})
        print(f"Epoch {epoch+1}/{epochs}: validation macro-F1={metrics['macro_f1']:.4f}", flush=True)
        if metrics["macro_f1"] > best:
            best = metrics["macro_f1"]
            model.save_pretrained(output, safe_serialization=True)
    metadata["history"] = history
    metadata["best_validation_macro_f1"] = best
