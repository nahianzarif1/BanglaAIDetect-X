"""
src/data/splitter.py

Splits data/processed/dataset.csv into train/validation/test by topic_id
(never by row) so no topic leaks across splits. Deterministic: fixed seed
from config.yaml. Never hand-edit the resulting split files — regenerate
them by re-running this script.
"""

import os
import yaml
import numpy as np
import pandas as pd


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def split_by_topic(df: pd.DataFrame, ratios: dict, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    topics = df["topic_id"].dropna().unique()
    topics = np.array(sorted(topics))  # sort first so shuffle is reproducible
    rng.shuffle(topics)

    n = len(topics)
    n_train = int(round(n * ratios["train"]))
    n_val = int(round(n * ratios["validation"]))
    # test gets the remainder so the three counts always sum to n
    train_topics = set(topics[:n_train])
    val_topics = set(topics[n_train:n_train + n_val])
    test_topics = set(topics[n_train + n_val:])

    def assign(topic_id):
        if topic_id in train_topics:
            return "train"
        if topic_id in val_topics:
            return "validation"
        if topic_id in test_topics:
            return "test"
        return None

    df = df.copy()
    df["split"] = df["topic_id"].apply(assign)
    return df


def verify_no_leakage(df: pd.DataFrame) -> None:
    leak = df.groupby("topic_id")["split"].nunique()
    bad = leak[leak > 1]
    assert bad.empty, f"Leakage detected for topic_id(s): {list(bad.index)}"


def main():
    cfg = load_config()
    # Check for both possible locations of the processed file
    possible_paths = [
        cfg["paths"]["processed"],
        "data/processed/dataset.csv",
    ]
    
    processed_path = None
    for path in possible_paths:
        if os.path.exists(path):
            processed_path = path
            break
    
    if processed_path is None:
        raise FileNotFoundError(
            "dataset.csv not found. Run python -m src.data.cleaner first."
        )
    
    df = pd.read_csv(processed_path)

    df = split_by_topic(df, cfg["dataset"]["split_ratio"], cfg["seed"])
    verify_no_leakage(df)

    os.makedirs(os.path.dirname(cfg["paths"]["train"]), exist_ok=True)
    for split_name, out_path in [
        ("train", cfg["paths"]["train"]),
        ("validation", cfg["paths"]["validation"]),
        ("test", cfg["paths"]["test"]),
    ]:
        subset = df[df["split"] == split_name]
        subset.to_csv(out_path, index=False)
        print(f"{split_name}: {len(subset)} rows -> {out_path}")


if __name__ == "__main__":
    main()
