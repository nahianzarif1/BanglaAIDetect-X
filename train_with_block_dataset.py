"""
train_with_block_dataset.py

Train the model using the block-enhanced dataset (individual + block samples).
"""

import os
import sys
import yaml
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, classification_report,
)
from sklearn.model_selection import train_test_split

sys.path.append(os.path.join(os.path.dirname(__file__), "."))
from src.features.stylometry import StylometryTransformer

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

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

def main():
    cfg = load_config()
    
    # Load block-enhanced dataset with user AI blocks
    df = pd.read_csv('data/processed/dataset_with_user_blocks.csv')
    print(f"Loaded dataset with user AI blocks: {len(df)} rows")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    # Split data
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=cfg["seed"], stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=cfg["seed"], stratify=temp_df['label'])
    
    print(f"\nTrain: {len(train_df)} rows")
    print(f"Validation: {len(val_df)} rows")
    print(f"Test: {len(test_df)} rows")
    
    # Prepare features
    x_train = train_df['text'].fillna("")
    y_train = (train_df['label'] == 'ai').astype(int)
    
    x_val = val_df['text'].fillna("")
    y_val = (val_df['label'] == 'ai').astype(int)
    
    x_test = test_df['text'].fillna("")
    y_test = (test_df['label'] == 'ai').astype(int)
    
    # Train model
    print("\nTraining model...")
    pipe = build_pipeline(cfg)
    pipe.fit(x_train, y_train)
    
    # Evaluate
    print("\n=== Validation ===")
    y_val_prob = pipe.predict_proba(x_val)[:, 1]
    y_val_pred = (y_val_prob >= 0.5).astype(int)
    print(classification_report(y_val, y_val_pred, target_names=["human", "ai"]))
    val_metrics = {
        "precision": precision_score(y_val, y_val_pred, zero_division=0),
        "recall": recall_score(y_val, y_val_pred, zero_division=0),
        "f1": f1_score(y_val, y_val_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_val, y_val_prob),
        "pr_auc": average_precision_score(y_val, y_val_prob),
        "accuracy": (y_val_pred == y_val).mean(),
        "n": len(val_df),
    }
    print(val_metrics)
    
    print("\n=== Test ===")
    y_test_prob = pipe.predict_proba(x_test)[:, 1]
    y_test_pred = (y_test_prob >= 0.5).astype(int)
    print(classification_report(y_test, y_test_pred, target_names=["human", "ai"]))
    test_metrics = {
        "precision": precision_score(y_test, y_test_pred, zero_division=0),
        "recall": recall_score(y_test, y_test_pred, zero_division=0),
        "f1": f1_score(y_test, y_test_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_test_prob),
        "pr_auc": average_precision_score(y_test, y_test_prob),
        "accuracy": (y_test_pred == y_test).mean(),
        "n": len(test_df),
    }
    print(test_metrics)
    
    # Save model
    os.makedirs(cfg["paths"]["models"], exist_ok=True)
    joblib.dump(pipe, os.path.join(cfg["paths"]["models"], "fusion_detector.joblib"))
    joblib.dump(pipe, os.path.join(cfg["paths"]["models"], "baseline_logreg.joblib"))
    
    # Save metrics
    os.makedirs(cfg["paths"]["reports"], exist_ok=True)
    import json
    with open(os.path.join(cfg["paths"]["reports"], "block_enhanced_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model": "tfidf_word+char + stylometry + logreg (block-enhanced dataset)",
            "validation": val_metrics,
            "test": test_metrics,
            "note": "Trained on block-enhanced dataset with 6,992 samples (3,496 human + 3,496 AI, including 1,000 block samples)"
        }, f, indent=2, ensure_ascii=False)
    
    print("\n✓ Model saved to models/fusion_detector.joblib")
    print("✓ Metrics saved to results/reports/block_enhanced_metrics.json")

if __name__ == "__main__":
    main()