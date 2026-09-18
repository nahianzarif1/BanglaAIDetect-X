"""
src/models/banglabert.py

Fine-tunes csebuetnlp/banglabert as a binary human-vs-AI classifier.
Run the TF-IDF baseline (src/models/baseline.py) successfully first.

Usage:
    python -m src.models.banglabert
"""

import os
import random
import yaml
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback,
)


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_datasets(cfg: dict, tokenizer):
    train_df = pd.read_csv(cfg["paths"]["train"])
    val_df = pd.read_csv(cfg["paths"]["validation"])
    test_df = pd.read_csv(cfg["paths"]["test"])

    def to_ds(df: pd.DataFrame) -> Dataset:
        df = df.copy()
        df["labels"] = (df[cfg["dataset"]["label_col"]] == "ai").astype(int)
        ds = Dataset.from_pandas(df[[cfg["dataset"]["text_col"], "labels"]])

        def tokenize(batch):
            return tokenizer(
                batch[cfg["dataset"]["text_col"]],
                truncation=True,
                max_length=cfg["banglabert"]["max_length"],
                padding="max_length",
            )

        return ds.map(tokenize, batched=True)

    return to_ds(train_df), to_ds(val_df), to_ds(test_df)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    preds = (probs >= 0.5).astype(int)
    return {
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "roc_auc": roc_auc_score(labels, probs) if len(set(labels)) > 1 else float("nan"),
    }


def main():
    cfg = load_config()
    set_seed(cfg["seed"])

    model_name = cfg["banglabert"]["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    train_ds, val_ds, test_ds = load_datasets(cfg, tokenizer)

    out_dir = os.path.join(cfg["paths"]["models"], "banglabert_detector")
    os.makedirs(out_dir, exist_ok=True)

    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=cfg["banglabert"]["batch_size"],
        per_device_eval_batch_size=cfg["banglabert"]["batch_size"],
        learning_rate=cfg["banglabert"]["learning_rate"],
        num_train_epochs=cfg["banglabert"]["epochs"],
        warmup_ratio=cfg["banglabert"]["warmup_ratio"],
        weight_decay=cfg["banglabert"]["weight_decay"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        seed=cfg["seed"],
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    test_metrics = trainer.evaluate(test_ds)
    print("Test metrics:", test_metrics)

    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Model saved to {out_dir}")


if __name__ == "__main__":
    main()
