"""
src/data/cleaner.py

Reads data/processed/collected_raw.csv (produced by collector.py),
applies deterministic normalization, drops/flags below-threshold rows,
and writes the final data/processed/dataset.csv (the single source of
truth that splitter.py reads from).
"""

import os
import sys
import yaml
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.preprocessing.normalize import normalize_text, detect_script  # noqa: E402


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def clean_dataframe(df: pd.DataFrame, min_tokens: int) -> tuple:
    df = df.copy()
    df["text"] = df["text"].fillna("").apply(normalize_text)
    df["language_type"] = df["text"].apply(detect_script)

    token_counts = df["text"].str.split().apply(len)
    keep_mask = token_counts >= min_tokens
    if "language_type" in df.columns:
        keep_mask = keep_mask & df["language_type"].isin(["bangla"])
    dropped = df[~keep_mask]
    kept = df[keep_mask].reset_index(drop=True)
    return kept, dropped


def main():
    cfg = load_config()
    # Check for both possible locations of the collected raw file
    possible_paths = [
        os.path.join(os.path.dirname(cfg["paths"]["processed"]), "collected_raw.csv"),
        "data/processed/collected_raw.csv",
    ]
    
    raw_merged_path = None
    for path in possible_paths:
        if os.path.exists(path):
            raw_merged_path = path
            break
    
    if raw_merged_path is None:
        raise FileNotFoundError(
            "collected_raw.csv not found. Run python -m src.data.collector or python create_sample_dataset.py first."
        )
    
    df = pd.read_csv(raw_merged_path)

    kept, dropped = clean_dataframe(df, cfg["dataset"]["min_tokens"])

    os.makedirs(os.path.dirname(cfg["paths"]["processed"]), exist_ok=True)
    kept.to_csv(cfg["paths"]["processed"], index=False)

    os.makedirs(cfg["paths"]["reports"], exist_ok=True)
    dropped_path = os.path.join(cfg["paths"]["reports"], "dropped_short_samples.csv")
    dropped.to_csv(dropped_path, index=False)

    print(f"Kept {len(kept)} rows -> {cfg['paths']['processed']}")
    print(f"Dropped {len(dropped)} below-threshold rows -> {dropped_path}")


if __name__ == "__main__":
    main()
