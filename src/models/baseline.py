"""
src/models/baseline.py

Level-1 baseline: TF-IDF + Logistic Regression.
Build and pass this FIRST, before touching BanglaBERT — it proves
data -> train -> eval -> predict works end-to-end.

Usage:
    python -m src.models.baseline
"""

import os
import json
import yaml
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, classification_report,
)


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_splits(cfg: dict):
    # Check for both possible locations of split files
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
    y = (df[cfg["dataset"]["label_col"]] == "ai").astype(int)  # 1 = ai, 0 = human
    return x, y


def train_baseline(train_df: pd.DataFrame, cfg: dict):
    x_train, y_train = to_xy(train_df, cfg)

    vectorizer = TfidfVectorizer(
        max_features=cfg["baseline"]["tfidf_max_features"],
        ngram_range=tuple(cfg["baseline"]["tfidf_ngram_range"]),
    )
    x_train_vec = vectorizer.fit_transform(x_train)

    clf = LogisticRegression(
        C=cfg["baseline"]["logreg_C"],
        max_iter=cfg["baseline"]["logreg_max_iter"],
        random_state=cfg["seed"],
    )
    clf.fit(x_train_vec, y_train)
    return vectorizer, clf


def evaluate(vectorizer, clf, df: pd.DataFrame, cfg: dict) -> dict:
    x, y_true = to_xy(df, cfg)
    x_vec = vectorizer.transform(x)
    y_prob = clf.predict_proba(x_vec)[:, 1]
    y_pred = (y_prob >= cfg["inference"]["decision_threshold"]).astype(int)

    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if y_true.nunique() > 1 else float("nan"),
        "pr_auc": average_precision_score(y_true, y_prob) if y_true.nunique() > 1 else float("nan"),
    }
    report = classification_report(y_true, y_pred, target_names=["human", "ai"], zero_division=0)
    return metrics, report


def main():
    cfg = load_config()
    train_df, val_df, test_df = load_splits(cfg)

    vectorizer, clf = train_baseline(train_df, cfg)

    val_metrics, val_report = evaluate(vectorizer, clf, val_df, cfg)
    test_metrics, test_report = evaluate(vectorizer, clf, test_df, cfg)

    print("=== Validation ===")
    print(val_report)
    print(val_metrics)
    print("=== Test ===")
    print(test_report)
    print(test_metrics)

    os.makedirs(cfg["paths"]["models"], exist_ok=True)
    joblib.dump(vectorizer, os.path.join(cfg["paths"]["models"], "tfidf_vectorizer.joblib"))
    joblib.dump(clf, os.path.join(cfg["paths"]["models"], "baseline_logreg.joblib"))

    os.makedirs(cfg["paths"]["reports"], exist_ok=True)
    with open(os.path.join(cfg["paths"]["reports"], "baseline_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"validation": val_metrics, "test": test_metrics}, f, indent=2, ensure_ascii=False)

    print("Saved model + vectorizer to models/, metrics to results/reports/baseline_metrics.json")


if __name__ == "__main__":
    main()
