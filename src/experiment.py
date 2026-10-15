"""Train, select on validation, and report independent test measurements."""
import gc
import json
import subprocess
import time
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.data.validator import main as validate
from src.models.baseline import load_splits
from src.models.catalog import CLASSICAL, TRANSFORMERS
from src.models.classical import train_classical
from src.models.registry import load_registry, load_predictor
from src.research import (binary_metrics, bootstrap_intervals, choose_threshold,
    file_hash, selective_metrics, subgroup_metrics, versions, write_json)


def learn_weights(y, matrix):
    from sklearn.metrics import log_loss
    size = matrix.shape[1]
    uniform = np.ones(size)/size
    def objective(weights):
        return log_loss(y, np.clip(matrix @ weights, 1e-7, 1-1e-7), labels=[0, 1]) + .01*np.sum((weights-uniform)**2)
    result = minimize(objective, uniform, method="SLSQP", bounds=[(0, 1)]*size,
        constraints={"type": "eq", "fun": lambda w: w.sum()-1}, options={"maxiter": 500, "ftol": 1e-9})
    weights = result.x if result.success else uniform
    weights = np.maximum(weights, 0)
    weights[weights < 1e-8] = 0
    return weights/weights.sum()


def run_experiment(cfg, model_ids=None, transformer_mode="embeddings", resume_artifacts=None):
    from filelock import FileLock, Timeout
    root = Path(cfg["paths"]["models"])
    root.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(root / ".training.lock"), timeout=0):
            return _run_experiment(cfg, model_ids, transformer_mode, resume_artifacts)
    except Timeout as exc:
        raise RuntimeError("Another experiment is already writing to this model directory. Wait for it to finish.") from exc


def _run_experiment(cfg, model_ids=None, transformer_mode="embeddings", resume_artifacts=None):
    model_ids = list(dict.fromkeys(list(CLASSICAL) if model_ids is None else model_ids))
    if resume_artifacts and not Path(resume_artifacts).is_dir():
        raise ValueError("Resume artifact directory does not exist.")
    unknown = set(model_ids)-set(CLASSICAL)-set(TRANSFORMERS)
    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}")
    validate(cfg=cfg)
    train, val, test = load_splits(cfg)
    hashes = {name: file_hash(cfg["paths"][name]) for name in ["train", "validation", "test"]}
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    models_root = Path(cfg["paths"]["models"])
    reports_root = Path(cfg["paths"]["reports"])
    output = reports_root.parent / "experiments" / run_id
    output.mkdir(parents=True)
    source_root = Path(__file__).resolve().parents[1]
    source_hashes = {str(p.relative_to(source_root)): file_hash(p) for p in sorted((source_root / 'src').rglob('*.py'))}
    for relative in source_hashes:
        destination = output / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / relative, destination)
    model_output = models_root / "artifacts" / run_id
    model_output.mkdir(parents=True)
    registry = load_registry(cfg)
    # Never mix artifacts trained/evaluated against different data partitions.
    if registry.get("split_hashes") != hashes:
        registry = {"schema_version": 1, "models": {}, "default_model": None}
    entries = {key: value for key, value in registry["models"].items() if value["kind"] != "ensemble"}
    val_probabilities, test_probabilities, all_reports = {}, {}, {}
    for model_id, entry in list(entries.items()):
        if model_id in model_ids:
            continue
        path = Path(entry["report_dir"])
        if not (path / "predictions.csv").exists():
            entries.pop(model_id)
            continue
        previous = pd.read_csv(path / "predictions.csv")
        for split_name, frame, destination in [("validation", val, val_probabilities), ("test", test, test_probabilities)]:
            subset = previous.loc[previous.split == split_name].set_index("id")
            destination[model_id] = subset.loc[frame.id.astype(str), "ai_probability"].to_numpy()
        original_report = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
        report = {**original_report, **_report(model_id, val, test, val_probabilities[model_id], test_probabilities[model_id], entry["threshold"], cfg)}
        report_dir = output / model_id
        _save_model_report(report_dir, report, val, test, val_probabilities[model_id], test_probabilities[model_id])
        entry["report_dir"] = str(report_dir)
        all_reports[model_id] = report
    y_val = val.label.eq("ai").astype(int).to_numpy()
    if not entries and not model_ids:
        raise ValueError("No compatible saved model predictions to refresh. Train models first.")
    y_test = test.label.eq("ai").astype(int).to_numpy()
    for model_id in model_ids:
        started = time.monotonic()
        print(f"Training {model_id} ({len(train)} documents)...", flush=True)
        neural_metadata = None
        recovered = False
        if model_id in CLASSICAL:
            predictor = train_classical(model_id, train, cfg)
            artifact = model_output / f"{model_id}.joblib"
            joblib.dump(predictor, artifact)
            kind, name = "sklearn", CLASSICAL[model_id]
        else:
            from src.models.transformer import train_transformer, TransformerPredictor
            artifact = (Path(resume_artifacts) / model_id).resolve() if resume_artifacts else model_output / model_id
            if resume_artifacts and (artifact / "detector.json").exists():
                neural_metadata = json.loads((artifact / "detector.json").read_text(encoding="utf-8"))
                if neural_metadata.get("split_hashes") != hashes or neural_metadata["mode"] != transformer_mode or neural_metadata["seed"] != cfg["seed"] or neural_metadata["model_id"] != model_id:
                    raise ValueError("Resumed artifact must match the split hashes, mode, and seed of this experiment.")
                predictor = TransformerPredictor(artifact, cfg["transformers"].get("device", "auto"))
                recovered = True
                print(f"Recovering completed artifact for {model_id}; training is preserved.", flush=True)
            else:
                artifact = model_output / model_id
                predictor, neural_metadata = train_transformer(model_id, train, val, cfg, artifact, transformer_mode)
            kind, name = "transformer_" + ("embeddings" if transformer_mode == "embeddings" else "finetuned"), TRANSFORMERS[model_id]["name"]
        val_probabilities[model_id] = predictor.predict_proba(val.text)[:, 1]
        threshold = choose_threshold(y_val, val_probabilities[model_id])
        test_probabilities[model_id] = predictor.predict_proba(test.text)[:, 1]
        report = _report(model_id, val, test, val_probabilities[model_id], test_probabilities[model_id], threshold, cfg)
        report.update({"training_seconds": None if recovered else time.monotonic()-started, "transformer": neural_metadata,
            "seed": cfg["seed"], "split_hashes": hashes, "training_source_hashes": None if recovered else source_hashes,
            "recovered_artifact": recovered})
        report_dir = output / model_id
        _save_model_report(report_dir, report, val, test, val_probabilities[model_id], test_probabilities[model_id])
        all_reports[model_id] = report
        entries[model_id] = {"name": name, "kind": kind, "artifact": str(artifact.relative_to(models_root)),
            "threshold": threshold, "report_dir": str(report_dir), "run_id": run_id}
        print(f"{model_id}: validation macro-F1={report['validation']['macro_f1']:.4f}; test macro-F1={report['test']['macro_f1']:.4f}", flush=True)
        del predictor
        gc.collect()
        if model_id in TRANSFORMERS:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
    if len(entries) > 1:
        ids = sorted(entries)
        weights = learn_weights(y_val, np.column_stack([val_probabilities[key] for key in ids]))
        p_val = np.column_stack([val_probabilities[key] for key in ids]) @ weights
        p_test = np.column_stack([test_probabilities[key] for key in ids]) @ weights
        threshold = choose_threshold(y_val, p_val)
        report = _report("ensemble", val, test, p_val, p_test, threshold, cfg)
        report["weights"] = dict(zip(ids, weights.tolist()))
        _save_model_report(output / "ensemble", report, val, test, p_val, p_test)
        all_reports["ensemble"] = report
        artifact = model_output / "ensemble.json"
        write_json(artifact, report["weights"])
        entries["ensemble"] = {"name": "Validation-weighted ensemble", "kind": "ensemble", "weights": report["weights"],
            "threshold": threshold, "artifact": str(artifact.relative_to(models_root)), "report_dir": str(output / "ensemble"), "run_id": run_id}
    default = max(entries, key=lambda key: (all_reports[key]["validation"]["macro_f1"], -all_reports[key]["validation"]["brier"]))
    split_manifest_path = reports_root / "split_manifest.json"
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8")) if split_manifest_path.exists() else {"limitation": "Split provenance unavailable."}
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    manifest = {"run_id": run_id, "seed": cfg["seed"], "config": cfg, "versions": versions(), "git_revision": revision,
        "source_hashes": source_hashes, "split_hashes": hashes, "split_manifest": split_manifest,
        "selection": "maximum validation macro-F1, then minimum validation Brier; test never used for selection",
        "default_model": default, "models": all_reports}
    write_json(output / "experiment.json", manifest)
    registry = {"schema_version": 1, "models": entries, "default_model": default, "split_hashes": hashes,
        "latest_report": str(output / "experiment.json"), "run_id": run_id}
    rows = []
    for model_id, report in all_reports.items():
        rows.append({"model": model_id, "name": entries[model_id]["name"], "kind": entries[model_id]["kind"],
            "validation_macro_f1": report["validation"]["macro_f1"], **{key: value for key, value in report["test"].items() if not isinstance(value, (list, dict))}})
    pd.DataFrame(rows).to_csv(output / "comparison.csv", index=False)
    pd.DataFrame(rows).to_csv(reports_root / "comparison.csv", index=False)
    if any(file_hash(cfg["paths"][name]) != hashes[name] for name in hashes):
        raise RuntimeError("Split files changed during training. The previous model registry was preserved; rerun on fixed partitions.")
    write_json(reports_root / "experiment.json", manifest)
    write_json(models_root / "registry.json", registry)
    print(f"Experiment complete. Default selected on validation: {default}. Reports: {output}", flush=True)
    return manifest


def _report(model_id, val, test, p_val, p_test, threshold, cfg):
    y_test = test.label.eq("ai").astype(int).to_numpy()
    result = {"model": model_id, "validation": binary_metrics(val.label.eq("ai").astype(int), p_val, threshold),
        "test": binary_metrics(y_test, p_test, threshold), "test_subgroups": subgroup_metrics(test, p_test, threshold),
        "selective_test": selective_metrics(y_test, p_test, threshold, cfg["inference"].get("uncertain_margin", .08))}
    group_column = "group_id" if "group_id" in test else "topic_id"
    result["test_95pct_group_bootstrap_ci"] = bootstrap_intervals(y_test, p_test, test[group_column], threshold,
        cfg["seed"], cfg.get("evaluation", {}).get("bootstrap_repetitions", 200))
    return result


def _save_model_report(path, report, val, test, p_val, p_test):
    path.mkdir(parents=True, exist_ok=True)
    write_json(path / "metrics.json", report)
    rows = []
    for name, frame, probabilities in [("validation", val, p_val), ("test", test, p_test)]:
        table = frame[[c for c in ["id", "label", "group_id", "generator", "source", "genre"] if c in frame]].copy()
        table["split"] = name
        table["ai_probability"] = probabilities
        table["prediction"] = np.where(probabilities >= report["test"]["threshold"], "ai", "human")
        margin = report["selective_test"]["uncertain_margin"]
        table["decision"] = np.where(np.abs(probabilities-report["test"]["threshold"]) < margin, "uncertain", table["prediction"])
        rows.append(table)
    predictions = pd.concat(rows, ignore_index=True)
    predictions.to_csv(path / "predictions.csv", index=False)
    predictions.loc[(predictions.split == "test") & (predictions.prediction != predictions.label)].to_csv(path / "test_errors.csv", index=False)
