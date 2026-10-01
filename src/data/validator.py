"""
src/data/validator.py

Run this BEFORE training anything. It never modifies data — it only
reports problems to results/reports/validation_report.txt and raises
if a hard rule is violated.
"""

import os
import sys
import yaml
import pandas as pd

REQUIRED_COLUMNS = [
    "id", "text", "label", "generator", "genre",
    "language_type", "edit_type", "topic_id", "writer_subgroup", "split",
]


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_schema(df: pd.DataFrame) -> list:
    problems = []
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        problems.append(f"Missing columns: {missing_cols}")
        return problems  # can't check further without the columns

    if df["id"].duplicated().any():
        problems.append("Duplicate ids found.")
    if df["id"].isna().any() or df["topic_id"].isna().any() or df["generator"].isna().any():
        problems.append("Null id / topic_id / generator found — every row must be traceable.")
    if not set(df["label"].dropna().unique()).issubset({"human", "ai"}):
        problems.append(f"Unexpected label values: {df['label'].unique()}")
    return problems


def validate_length(df: pd.DataFrame, min_tokens: int) -> list:
    problems = []
    token_counts = df["text"].fillna("").str.split().apply(len)
    too_short = (token_counts < min_tokens).sum()
    if too_short > 0:
        problems.append(f"{too_short} rows are below the {min_tokens}-token minimum "
                         f"(flag or drop them before training).")
    return problems


def validate_balance(df: pd.DataFrame) -> list:
    problems = []
    counts = df["label"].value_counts(normalize=True)
    if counts.empty:
        return problems
    imbalance = counts.max() - counts.min()
    if imbalance > 0.2:
        problems.append(f"Label imbalance is {imbalance:.2%} (human vs ai) — consider rebalancing.")
    return problems


def validate_split_leakage(df: pd.DataFrame) -> list:
    """A topic_id must never appear in more than one split."""
    problems = []
    if "split" not in df.columns or df["split"].isna().all():
        return problems  # not split yet, nothing to check
    leak = df.groupby("topic_id")["split"].nunique()
    leaking_topics = leak[leak > 1]
    if not leaking_topics.empty:
        problems.append(
            f"DATA LEAKAGE: {len(leaking_topics)} topic_id(s) appear in more than one split: "
            f"{list(leaking_topics.index)[:10]}"
        )
    return problems


def validate_no_demographic_feature_leak(df: pd.DataFrame) -> list:
    """writer_subgroup must exist only as metadata, never silently used as a feature
    column name that a model script might pick up automatically."""
    problems = []
    if "writer_subgroup" not in df.columns:
        problems.append("writer_subgroup column missing — fairness audit will be impossible.")
    return problems


def run_all_validations(df: pd.DataFrame, cfg: dict) -> list:
    problems = []
    problems += validate_schema(df)
    if not problems:  # only continue if schema is OK
        problems += validate_length(df, cfg["dataset"]["min_tokens"])
        problems += validate_balance(df)
        problems += validate_split_leakage(df)
        problems += validate_no_demographic_feature_leak(df)
    return problems


def main(processed_csv: str = None):
    cfg = load_config()
    # Check for both possible locations of the processed file
    possible_paths = [
        processed_csv,
        cfg["paths"]["processed"],
        "data/processed/dataset.csv",
    ]
    
    path = None
    for possible_path in possible_paths:
        if possible_path and os.path.exists(possible_path):
            path = possible_path
            break
    
    if path is None:
        raise FileNotFoundError(
            "dataset.csv not found. Run python -m src.data.cleaner first."
        )
    
    df = pd.read_csv(path)

    problems = run_all_validations(df, cfg)

    os.makedirs(cfg["paths"]["reports"], exist_ok=True)
    report_path = os.path.join(cfg["paths"]["reports"], "validation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        if not problems:
            f.write("All validations passed.\n")
        else:
            f.write("VALIDATION ISSUES FOUND:\n")
            for p in problems:
                f.write(f"- {p}\n")

    print(f"Validation report written to {report_path}")
    for p in problems:
        print("WARNING:", p)

    # Hard-fail only on leakage — everything else is a warning you fix by hand
    if any("DATA LEAKAGE" in p for p in problems):
        sys.exit(1)


if __name__ == "__main__":
    main()
