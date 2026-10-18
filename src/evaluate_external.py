"""Evaluate a locked detector on new labeled data without changing its threshold."""
import argparse
from pathlib import Path
import pandas as pd
from load_custom_dataset import convert_dataset
from src.config import load_config
from src.inference import BanglaAIDetector
from src.data.grouping import text_fingerprint
from src.research import binary_metrics, file_hash, subgroup_metrics, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model")
    parser.add_argument("--output", default="results/external")
    parser.add_argument("--allow-overlap", action="store_true", help="Diagnostic evaluation only: disclose known training/validation overlap")
    args = parser.parse_args(argv)
    source = Path(args.dataset)
    raw = pd.read_excel(source) if source.suffix.lower() == ".xlsx" else pd.read_csv(source)
    frame = convert_dataset(raw, source.name)
    cfg = load_config()
    existing = set()
    for name in ["train", "validation"]:
        if Path(cfg["paths"][name]).exists():
            existing.update(pd.read_csv(cfg["paths"][name]).text.map(text_fingerprint))
    overlap = frame.text.map(text_fingerprint).isin(existing)
    if overlap.any() and not args.allow_overlap:
        raise ValueError(f"External corpus contains {int(overlap.sum())} known training/validation texts. Use independent data; --allow-overlap is for disclosed diagnostics only.")
    detector = BanglaAIDetector(cfg, args.model)
    results = []
    for start in range(0, len(frame), 200):
        results.extend(detector.predict_many(frame.text.iloc[start:start+200]))
    predictions = pd.DataFrame(results)
    if frame.empty:
        raise ValueError("External corpus is empty.")
    valid = predictions.get("error", pd.Series(None, index=predictions.index)).isna()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    table = pd.concat([frame[["id", "label", "generator", "source"]], predictions.drop(columns=["style", "cues"], errors="ignore").add_prefix("detector_")], axis=1)
    table.to_csv(output / f"{detector.model_id}_predictions.csv", index=False)
    report = {"model": detector.model_id, "threshold": detector.threshold, "source_sha256": file_hash(source),
        "total_rows": len(frame), "scored_rows": int(valid.sum()), "excluded_rows": int((~valid).sum()),
        "known_training_validation_overlap": int(overlap.sum()), "overlap_audit": "exact normalized texts; semantic paraphrases were not audited",
        "note": "Locked model/threshold. Excluded invalid inputs are disclosed; metrics cover scored rows only."}
    if valid.any():
        subset, p = frame.loc[valid], predictions.loc[valid, "ai_probability"].to_numpy()
        report["metrics"] = binary_metrics(subset.label.eq("ai").astype(int), p, detector.threshold)
        report["subgroups"] = subgroup_metrics(subset, p, detector.threshold)
    write_json(output / f"{detector.model_id}_metrics.json", report)
    print(f"Scored {report['scored_rows']}/{len(frame)} external documents; threshold unchanged.")
    return report


if __name__ == "__main__":
    main()
