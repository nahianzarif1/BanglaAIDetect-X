"""Fail training on schema errors, split leakage, or unusable partitions."""
from pathlib import Path
import pandas as pd
from src.config import load_config
from src.data.splitter import verify_no_leakage
from src.preprocessing.normalize import is_supported_bangla
from src.data.grouping import text_fingerprint
from src.research import write_json

REQUIRED_COLUMNS = ["id", "text", "label", "generator", "genre", "topic_id", "writer_subgroup"]


def validate_schema(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df]
    if missing:
        return [f"Missing columns: {missing}"]
    problems = []
    if df.id.isna().any() or df.id.duplicated().any():
        problems.append("Missing or duplicate IDs.")
    if df.label.isna().any() or not set(df.label).issubset({"human", "ai"}):
        problems.append("Labels must be human or ai.")
    if df.text.isna().any() or df.text.astype(str).str.strip().eq("").any():
        problems.append("Empty text found.")
    return problems


def validate_split_leakage(df):
    if "split" not in df:
        return []
    try:
        copy = df.copy()
        copy["text_hash"] = copy.text.map(text_fingerprint)
        verify_no_leakage(copy)
        return []
    except ValueError as exc:
        return [str(exc)]


def run_all_validations(df, cfg):
    problems = validate_schema(df)
    if problems:
        return problems
    if (df.text.str.split().map(len) < cfg["dataset"]["min_tokens"]).any():
        problems.append("Below-minimum text found.")
    if not df.text.map(lambda t: is_supported_bangla(t, cfg["inference"]["min_bangla_ratio"])).all():
        problems.append("Unsupported script found.")
    problems += validate_split_leakage(df)
    if "split" in df:
        if not set(df.split).issubset({"train", "validation", "test"}) or df.split.isna().any():
            problems.append("Invalid split values.")
        for name in ["train", "validation", "test"]:
            subset = df.loc[df.split == name]
            if subset.empty or subset.label.nunique() != 2:
                problems.append(f"{name} requires nonempty human and ai classes.")
    return problems


def main(processed_csv=None, cfg=None):
    cfg = cfg or load_config()
    if processed_csv:
        df = pd.read_csv(processed_csv)
    else:
        frames = []
        for name in ["train", "validation", "test"]:
            frame = pd.read_csv(cfg["paths"][name])
            if "split" not in frame or not frame.split.eq(name).all():
                raise ValueError(f"Split column disagrees with {name} file.")
            frames.append(frame)
        df = pd.concat(frames, ignore_index=True)
    problems = run_all_validations(df, cfg)
    report = {"rows": len(df), "problems": problems, "passed": not problems,
        "labels": df.label.value_counts().to_dict(), "null_rates": df.isna().mean().to_dict()}
    write_json(Path(cfg["paths"]["reports"]) / "validation_report.json", report)
    if problems:
        raise ValueError("Dataset validation failed: " + "; ".join(problems))
    print(f"Validated {len(df)} documents; no exact-text or known-group leakage.")
    return report


if __name__ == "__main__":
    main()
