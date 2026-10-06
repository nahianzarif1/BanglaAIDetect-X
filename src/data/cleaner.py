"""Normalize and remove invalid, duplicate, or label-conflicting documents."""
from pathlib import Path
import pandas as pd
from src.config import load_config
from src.data.grouping import text_fingerprint
from src.preprocessing.normalize import normalize_text, is_supported_bangla
from src.research import write_json


def clean_dataframe(df, min_tokens, min_bangla_ratio=.75):
    if not {"id", "text", "label"}.issubset(df):
        raise ValueError("Dataset must contain id, text, and label.")
    df = df.copy()
    if df.label.isna().any() or not set(df.label).issubset({"human", "ai"}):
        raise ValueError("All labels must be human or ai.")
    if df.id.isna().any() or df.id.duplicated().any() or df.id.astype(str).str.strip().eq("").any():
        raise ValueError("Every document needs a unique, nonempty ID.")
    df["text"] = df.text.fillna("").astype(str).map(normalize_text)
    df["text_hash"] = df.text.map(text_fingerprint)
    conflicts = set(df.groupby("text_hash").label.nunique().loc[lambda x: x > 1].index)
    df["drop_reason"] = ""
    df.loc[df.text.str.split().map(len) < min_tokens, "drop_reason"] = "too_short"
    df.loc[~df.text.map(lambda t: is_supported_bangla(t, min_bangla_ratio)), "drop_reason"] = "unsupported_script"
    df.loc[df.duplicated("text_hash"), "drop_reason"] = "duplicate_text"
    df.loc[df.text_hash.isin(conflicts), "drop_reason"] = "conflicting_labels"
    dropped = df.loc[df.drop_reason.ne("")].copy()
    kept = df.loc[df.drop_reason.eq("")].drop(columns="drop_reason").reset_index(drop=True)
    if kept.empty:
        raise ValueError("No eligible Bangla documents remain after cleaning.")
    return kept, dropped


def main(cfg=None):
    cfg = cfg or load_config()
    raw = Path(cfg["paths"]["processed"]).parent / "collected_raw.csv"
    kept, dropped = clean_dataframe(pd.read_csv(raw), cfg["dataset"]["min_tokens"], cfg["inference"]["min_bangla_ratio"])
    kept.to_csv(cfg["paths"]["processed"], index=False)
    reports = Path(cfg["paths"]["reports"])
    reports.mkdir(parents=True, exist_ok=True)
    dropped.to_csv(reports / "dropped_samples.csv", index=False)
    write_json(reports / "cleaning_report.json", {"input_rows": len(kept) + len(dropped), "kept": len(kept),
        "dropped": len(dropped), "drop_reasons": dropped.drop_reason.value_counts().to_dict()})
    print(f"Cleaning: kept {len(kept)}; dropped {len(dropped)}.")
    return kept


if __name__ == "__main__":
    main()
