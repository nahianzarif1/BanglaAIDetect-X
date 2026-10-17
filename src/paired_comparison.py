"""Paired document-group comparisons from saved test predictions; no retraining."""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score
from src.config import load_config
from src.models.registry import load_registry
from src.research import file_hash, write_json


def paired_comparison(y, candidate, reference, candidate_threshold, reference_threshold, groups, seed=42, repetitions=1000):
    y, groups = np.asarray(y), np.asarray(groups)
    a, b = np.asarray(candidate) >= candidate_threshold, np.asarray(reference) >= reference_threshold
    difference = (a == y).astype(float) - (b == y).astype(float)
    unique = np.unique(groups)
    indices = [np.flatnonzero(groups == group) for group in unique]
    group_sums = np.array([difference[index].sum() for index in indices])
    rng = np.random.default_rng(seed)
    null = rng.choice([-1, 1], (repetitions, len(unique))) @ group_sums / len(y)
    effect = float(difference.mean())
    p_value = float((1 + np.sum(np.abs(null) >= abs(effect)-1e-12))/(repetitions+1))
    deltas = []
    for _ in range(repetitions):
        sample = np.concatenate([indices[i] for i in rng.integers(0, len(indices), len(indices))])
        deltas.append(f1_score(y[sample], a[sample], labels=[0, 1], average="macro", zero_division=0) - f1_score(y[sample], b[sample], labels=[0, 1], average="macro", zero_division=0))
    return {"accuracy_difference": effect, "macro_f1_difference": float(f1_score(y, a, labels=[0, 1], average="macro", zero_division=0)-f1_score(y, b, labels=[0, 1], average="macro", zero_division=0)),
        "macro_f1_difference_95pct_group_ci": np.quantile(deltas, [.025, .975]).tolist(),
        "paired_group_permutation_p": p_value, "groups": len(unique), "rows": len(y), "repetitions": repetitions}


def compare_saved(cfg=None, reference=None, repetitions=1000):
    import json
    cfg = cfg or load_config()
    registry = load_registry(cfg)
    if not registry["models"]:
        raise ValueError("Train an experiment before comparing predictions.")
    for split in ["train", "validation", "test"]:
        if file_hash(cfg["paths"][split]) != registry["split_hashes"][split]:
            raise ValueError("Current partitions differ from the trained experiment.")
    experiment = json.loads(Path(registry["latest_report"]).read_text(encoding="utf-8"))
    if reference is None:
        singles = [key for key, entry in registry["models"].items() if entry["kind"] != "ensemble"]
        reference = max(singles, key=lambda key: experiment["models"][key]["validation"]["macro_f1"])
    if reference not in registry["models"]:
        raise ValueError("Unknown reference model.")
    frame = pd.read_csv(cfg["paths"]["test"])
    probabilities = {}
    for key, entry in registry["models"].items():
        predictions = pd.read_csv(Path(entry["report_dir"]) / "predictions.csv")
        subset = predictions.loc[predictions.split == "test"].set_index("id")
        probabilities[key] = subset.loc[frame.id.astype(str), "ai_probability"].to_numpy()
    results = {}
    for key in probabilities:
        if key == reference:
            continue
        results[key] = paired_comparison(frame.label.eq("ai").astype(int), probabilities[key], probabilities[reference],
            registry["models"][key]["threshold"], registry["models"][reference]["threshold"],
            frame.get("group_id", frame.topic_id), cfg["seed"], repetitions)
    ordered = sorted(results, key=lambda key: results[key]["paired_group_permutation_p"])
    previous = 0.
    for index, key in enumerate(ordered):
        adjusted = min(1., (len(ordered)-index)*results[key]["paired_group_permutation_p"])
        previous = max(previous, adjusted)
        results[key]["holm_adjusted_p"] = previous
    report = {"run_id": registry["run_id"], "reference": reference, "reference_selection": "highest validation macro-F1 among single models unless explicitly supplied",
        "comparisons": results, "note": "Paired group sign permutation assumes exchangeability under the null. Holm adjustment covers this family. Corpus-specific significance does not establish external robustness."}
    write_json(Path(cfg["paths"]["reports"]) / "paired_comparisons.json", report)
    write_json(Path(registry["latest_report"]).parent / "paired_comparisons.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference")
    parser.add_argument("--repetitions", type=int, default=1000)
    args = parser.parse_args()
    if args.repetitions < 10:
        parser.error("At least 10 repetitions are required.")
    report = compare_saved(reference=args.reference, repetitions=args.repetitions)
    print(f"Compared {len(report['comparisons'])} models against {report['reference']} using paired document-group tests.")


if __name__ == "__main__":
    main()
