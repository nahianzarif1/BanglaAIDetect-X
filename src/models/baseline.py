"""
Level-1/2 fusion baseline:
  word TF-IDF + character TF-IDF + Bangla stylometry → Logistic Regression.

Word TF-IDF alone overfit the 20-row toy set and then guessed ~50% on
real ChatGPT / Wikipedia. Stylometry (burstiness, discourse markers,
numbers, formulaic openings) is what transfers to unseen topics.
"""

import os
import json
import yaml
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, classification_report, brier_score_loss,
)

from src.features.stylometry import StylometryTransformer


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_splits(cfg: dict):
    possible_paths = {
        "train": [cfg["paths"]["train"], "data/splits/train.csv"],
        "validation": [cfg["paths"]["validation"], "data/splits/validation.csv"],
        "test": [cfg["paths"]["test"], "data/splits/test.csv"],
    }

    splits = {}
    for split_name, path_options in possible_paths.items():
        for path in path_options:
            if os.path.exists(path):
                splits[split_name] = pd.read_csv(path)
                break
        if split_name not in splits:
            raise FileNotFoundError(
                f"{split_name}.csv not found. Run python -m src.data.splitter first."
            )

    return splits["train"], splits["validation"], splits["test"]


def to_xy(df: pd.DataFrame, cfg: dict):
    x = df[cfg["dataset"]["text_col"]].fillna("")
    y = (df[cfg["dataset"]["label_col"]] == "ai").astype(int)
    return x, y


def build_pipeline(cfg: dict) -> Pipeline:
    bcfg = cfg["baseline"]
    union = FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word",
            ngram_range=tuple(bcfg["tfidf_ngram_range"]),
            max_features=bcfg["tfidf_max_features"],
            min_df=bcfg.get("tfidf_min_df", 2),
            sublinear_tf=True,
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=tuple(bcfg.get("char_ngram_range", [3, 5])),
            max_features=bcfg.get("char_max_features", 15000),
            min_df=bcfg.get("tfidf_min_df", 2),
            sublinear_tf=True,
        )),
        ("style", Pipeline([
            ("extract", StylometryTransformer()),
            ("scale", StandardScaler()),
        ])),
    ])
    clf = LogisticRegression(
        C=bcfg["logreg_C"],
        max_iter=bcfg["logreg_max_iter"],
        class_weight="balanced",
        solver="liblinear",
        random_state=cfg["seed"],
    )
    return Pipeline([("features", union), ("clf", clf)])


def train_baseline(train_df: pd.DataFrame, cfg: dict):
    x_train, y_train = to_xy(train_df, cfg)
    pipe = build_pipeline(cfg)
    pipe.fit(x_train, y_train)
    return pipe


def evaluate(pipe, df: pd.DataFrame, cfg: dict) -> tuple:
    x, y_true = to_xy(df, cfg)
    y_prob = pipe.predict_proba(x)[:, 1]
    y_pred = (y_prob >= cfg["inference"]["decision_threshold"]).astype(int)

    # Handle single-class cases
    if y_true.nunique() == 1:
        # If only one class in the set, provide simple metrics
        metrics = {
            "n": int(len(df)),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "roc_auc": float("nan"),
            "pr_auc": float("nan"),
            "brier": float(brier_score_loss(y_true, y_prob)),
            "accuracy": float((y_pred == y_true).mean()),
        }
        report = f"Single class in set (only {'ai' if y_true.iloc[0] == 1 else 'human'}). Metrics limited."
    else:
        metrics = {
            "n": int(len(df)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_prob)),
            "pr_auc": float(average_precision_score(y_true, y_prob)),
            "brier": float(brier_score_loss(y_true, y_prob)),
            "accuracy": float((y_pred == y_true).mean()),
        }
        report = classification_report(y_true, y_pred, target_names=["human", "ai"], zero_division=0)
    return metrics, report


def main():
    cfg = load_config()
    train_df, val_df, test_df = load_splits(cfg)

    pipe = train_baseline(train_df, cfg)

    val_metrics, val_report = evaluate(pipe, val_df, cfg)
    test_metrics, test_report = evaluate(pipe, test_df, cfg)

    print("=== Validation ===")
    print(val_report)
    print(val_metrics)
    print("=== Test ===")
    print(test_report)
    print(test_metrics)

    os.makedirs(cfg["paths"]["models"], exist_ok=True)
    joblib.dump(pipe, os.path.join(cfg["paths"]["models"], "fusion_detector.joblib"))
    # Keep old filenames as aliases so older docs still work
    joblib.dump(pipe, os.path.join(cfg["paths"]["models"], "baseline_logreg.joblib"))
    joblib.dump(pipe, os.path.join(cfg["paths"]["models"], "tfidf_vectorizer.joblib"))

    os.makedirs(cfg["paths"]["reports"], exist_ok=True)
    with open(os.path.join(cfg["paths"]["reports"], "baseline_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": "tfidf_word+char + stylometry + logreg",
                "validation": val_metrics,
                "test": test_metrics,
                "note": "Accuracy on a tiny synthetic set is not the headline metric. Use F1, ROC-AUC, and Brier. Lab corpus ≠ Turnitin-scale data.",
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("Saved fusion pipeline to models/fusion_detector.joblib")
    print("Metrics -> results/reports/baseline_metrics.json")


if __name__ == "__main__":
    main()
