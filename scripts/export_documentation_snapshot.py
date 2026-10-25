"""Export aggregate documentation evidence; excludes text and local file paths.

Run after a verified experiment: python scripts/export_documentation_snapshot.py
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.models.registry import available_models, load_registry
from src.research import write_json


def main():
    cfg = load_config()
    registry = load_registry(cfg)
    if not registry.get("latest_report"):
        raise SystemExit("Train an experiment before exporting documentation evidence.")
    experiment = json.loads(Path(registry["latest_report"]).read_text(encoding="utf-8"))
    inventory = available_models(cfg)
    destination = ROOT / "docs/results"
    destination.mkdir(parents=True, exist_ok=True)
    rows, models = [], {}
    keys = ["n", "accuracy", "precision", "recall", "f1", "macro_f1", "balanced_accuracy", "mcc", "roc_auc", "pr_auc", "brier", "human_false_positive_rate", "threshold"]
    for model_id, metadata in registry["models"].items():
        report = experiment["models"][model_id]
        metrics = {key: report["test"][key] for key in keys}
        row = {"model": model_id, "name": metadata["name"], "kind": metadata["kind"],
            "validation_macro_f1": report["validation"]["macro_f1"], **metrics}
        rows.append(row)
        transformer = report.get("transformer")
        public_transformer = ({key: transformer.get(key) for key in ["checkpoint", "checkpoint_revision", "mode", "training_method", "max_length", "normalizer", "training_rows", "seed"]} if transformer else None)
        models[model_id] = {"name": metadata["name"], "kind": metadata["kind"], "local_artifact_verified": inventory[model_id]["available"],
            "validation_macro_f1": report["validation"]["macro_f1"], "test": metrics,
            "test_95pct_group_bootstrap_ci": report["test_95pct_group_bootstrap_ci"],
            "selective_test": report.get("selective_test"), "transformer": public_transformer}
    with (destination / "model-comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    split = experiment["split_manifest"]
    summary = {"source_run_id": experiment["run_id"], "training_seed": experiment["seed"],
        "trained_options": len(models), "default_model": registry["default_model"], "selection": experiment["selection"],
        "split_hashes": experiment["split_hashes"], "split_manifest": split, "versions": experiment["versions"],
        "models": models, "ensemble_weights": registry["models"].get("ensemble", {}).get("weights", {}),
        "not_run": {"banglabert_small": "Supported checkpoint; no completed local training run.",
            "full_transformer_finetuning": "Implemented and tested with a tiny offline encoder; the three reported full checkpoints use frozen encoders."},
        "limitations": [split.get("limitation"), "Single corpus and seed; external, topic-held-out, and unseen-generator generalization are not established.",
            "Local trained artifacts are not included in Git; a fresh clone must train models.",
            "Illustrative sample styles do not establish authorship; high held-out scores do not guarantee reliable new-text predictions."]}
    write_json(destination / "experiment-summary.json", summary)
    paired_path = Path(registry["latest_report"]).parent / "paired_comparisons.json"
    if paired_path.exists():
        write_json(destination / "paired-comparisons.json", json.loads(paired_path.read_text(encoding="utf-8")))
    print(f"Exported aggregate evidence for {len(models)} options to {destination}.")


if __name__ == "__main__":
    main()
