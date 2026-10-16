"""Recover completed neural predictions/artifacts after an interrupted experiment.

No model is retrained. Only reports and locally completed detectors with matching
partition hashes, seeds, and checkpoint revisions are admitted to the registry.
"""
import argparse
import json
from pathlib import Path
from src.config import load_config
from src.experiment import _run_experiment
from src.models.catalog import TRANSFORMERS
from src.models.registry import load_registry
from src.research import file_hash, write_json


def recover_run(report_run, artifact_roots, cfg=None):
    cfg = cfg or load_config()
    from filelock import FileLock
    with FileLock(str(Path(cfg["paths"]["models"])/".training.lock"), timeout=0):
        return _recover_run(report_run, artifact_roots, cfg)


def _recover_run(report_run, artifact_roots, cfg):
    root = Path(cfg["paths"]["models"])
    hashes = {name: file_hash(cfg["paths"][name]) for name in ["train", "validation", "test"]}
    registry = load_registry(cfg)
    if registry.get("split_hashes") != hashes:
        raise ValueError("Recovery requires the same validated partitions as the active registry.")
    recovered = []
    for model_id in TRANSFORMERS:
        report_dir = Path(report_run) / model_id
        if not all((report_dir / file).exists() for file in ["metrics.json", "predictions.csv"]):
            continue
        report = json.loads((report_dir / "metrics.json").read_text(encoding="utf-8"))
        if report.get("split_hashes") != hashes or report.get("seed") != cfg["seed"]:
            raise ValueError(f"Report partition hashes/seed do not match for {model_id}.")
        metadata = report.get("transformer")
        artifact = next((Path(candidate).resolve()/model_id for candidate in artifact_roots
            if (Path(candidate)/model_id/"detector.json").exists()), None)
        if artifact is None:
            raise ValueError(f"Completed artifact missing for {model_id}.")
        saved = json.loads((artifact / "detector.json").read_text(encoding="utf-8"))
        required = ["split_hashes", "seed", "mode", "checkpoint_revision", "model_id", "max_length"]
        if not metadata or any(saved.get(key) != metadata.get(key) for key in required):
            raise ValueError(f"Artifact provenance differs from the report for {model_id}.")
        if not (artifact/"model.safetensors").exists() or (saved["mode"] == "embeddings" and not (artifact/"head.joblib").exists()):
            raise ValueError(f"Artifact files are incomplete for {model_id}.")
        registry["models"][model_id] = {"name": TRANSFORMERS[model_id]["name"],
            "kind": "transformer_embeddings" if saved["mode"] == "embeddings" else "transformer_finetuned",
            "artifact": str(artifact.relative_to(root.resolve())), "threshold": report["test"]["threshold"],
            "report_dir": str(report_dir.resolve()), "run_id": Path(report_run).name}
        recovered.append(model_id)
    if not recovered:
        raise ValueError("No completed neural reports were found in this run.")
    write_json(root/"registry.json", registry)
    result = _run_experiment(cfg, [])
    print("Recovered completed models: " + ", ".join(recovered))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-run", required=True)
    parser.add_argument("--artifact-roots", nargs="+", required=True)
    args = parser.parse_args()
    recover_run(args.report_run, args.artifact_roots)


if __name__ == "__main__":
    main()
