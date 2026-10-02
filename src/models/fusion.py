"""
src/models/fusion.py

Fusion detector that combines TF-IDF features with stylometric features
for improved accuracy and robustness.
"""

import os
import json
import yaml
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, classification_report, brier_score_loss,
)
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.features.stylometry import StylometryTransformer


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


def create_fusion_pipeline(cfg: dict):
    """Create a fusion pipeline combining TF-IDF and stylometry features."""
    
    # Text features (TF-IDF)
    text_features = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=cfg["fusion"]["tfidf_max_features"],
            ngram_range=tuple(cfg["fusion"]["tfidf_ngram_range"]),
            min_df=cfg["fusion"].get("tfidf_min_df", 2),
        ))
    ])
    
    # Character features (TF-IDF)
    char_features = Pipeline([
        ('char_tfidf', TfidfVectorizer(
            analyzer='char',
            ngram_range=tuple(cfg["fusion"].get("char_ngram_range", [3, 5])),
            max_features=cfg["fusion"].get("char_max_features", 12000),
            min_df=cfg["fusion"].get("tfidf_min_df", 2),
        ))
    ])
    
    # Stylometry features
    style_features = Pipeline([
        ('stylometry', StylometryTransformer()),
        ('scaler', StandardScaler())
    ])
    
    # Combine features
    combined_features = FeatureUnion([
        ('word_tfidf', text_features),
        ('char_tfidf', char_features),
        ('style', style_features)
    ])
    
    # Full pipeline
    pipeline = Pipeline([
        ('features', combined_features),
        ('classifier', LogisticRegression(
            C=cfg["fusion"]["logreg_C"],
            max_iter=cfg["fusion"]["logreg_max_iter"],
            random_state=cfg["seed"],
            class_weight='balanced',  # Handle class imbalance
            solver='liblinear',  # Better for smaller datasets
        ))
    ])
    
    return pipeline


def train_fusion(train_df: pd.DataFrame, cfg: dict):
    x_train, y_train = to_xy(train_df, cfg)
    
    pipeline = create_fusion_pipeline(cfg)
    pipeline.fit(x_train, y_train)
    
    # Calibrate the classifier for better probability estimates
    calibrated_pipeline = CalibratedClassifierCV(
        pipeline, 
        method='isotonic', 
        cv=5
    )
    calibrated_pipeline.fit(x_train, y_train)
    
    return calibrated_pipeline


def evaluate(pipeline, df: pd.DataFrame, cfg: dict) -> dict:
    x, y_true = to_xy(df, cfg)
    y_prob = pipeline.predict_proba(x)[:, 1]
    y_pred = (y_prob >= cfg["inference"]["decision_threshold"]).astype(int)
    
    # Handle single-class cases
    if y_true.nunique() == 1:
        # If only one class in the set, provide simple metrics
        metrics = {
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "roc_auc": float("nan"),
            "pr_auc": float("nan"),
            "brier": brier_score_loss(y_true, y_prob),
            "accuracy": (y_pred == y_true).mean(),
            "n": len(df),
        }
        report = f"Single class in set (only {'ai' if y_true.iloc[0] == 1 else 'human'}). Metrics limited."
    else:
        metrics = {
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_prob),
            "pr_auc": average_precision_score(y_true, y_prob),
            "brier": brier_score_loss(y_true, y_prob),
            "accuracy": (y_pred == y_true).mean(),
            "n": len(df),
        }
        report = classification_report(y_true, y_pred, target_names=["human", "ai"], zero_division=0)
    return metrics, report


def main():
    cfg = load_config()
    train_df, val_df, test_df = load_splits(cfg)

    print("Training fusion detector (TF-IDF + Stylometry)...")
    pipeline = train_fusion(train_df, cfg)

    val_metrics, val_report = evaluate(pipeline, val_df, cfg)
    test_metrics, test_report = evaluate(pipeline, test_df, cfg)

    print("=== Validation ===")
    print(val_report)
    print(val_metrics)
    print("=== Test ===")
    print(test_report)
    print(test_metrics)

    os.makedirs(cfg["paths"]["models"], exist_ok=True)
    joblib.dump(pipeline, os.path.join(cfg["paths"]["models"], "fusion_detector.joblib"))
    # Also save as baseline for compatibility
    joblib.dump(pipeline, os.path.join(cfg["paths"]["models"], "baseline_logreg.joblib"))

    os.makedirs(cfg["paths"]["reports"], exist_ok=True)
    with open(os.path.join(cfg["paths"]["reports"], "fusion_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "validation": val_metrics,
            "test": test_metrics,
            "note": "Fusion detector with TF-IDF + 22 stylometric features, calibrated with isotonic regression."
        }, f, indent=2, ensure_ascii=False)

    print("Saved fusion model to models/fusion_detector.joblib")
    print("Saved compatibility copy to models/baseline_logreg.joblib")
    print("Saved metrics to results/reports/fusion_metrics.json")


if __name__ == "__main__":
    main()