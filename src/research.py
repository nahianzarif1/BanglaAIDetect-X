"""Reproducible binary evaluation with document-group bootstrap intervals."""
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, log_loss, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def versions():
    result = {}
    for package in ["numpy", "pandas", "scikit-learn", "torch", "transformers", "streamlit", "normalizer"]:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def binary_metrics(y, p, threshold=.5):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    if not len(y) or len(y) != len(p) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Metrics require nonempty labels and finite probabilities in [0, 1].")
    prediction = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    bins = []
    ece = 0.
    edges = np.linspace(0, 1, 11)
    for index in range(10):
        mask = (p >= edges[index]) & ((p < edges[index+1]) if index < 9 else (p <= 1))
        if mask.any():
            mean, observed = float(p[mask].mean()), float(y[mask].mean())
            ece += float(mask.mean()) * abs(mean - observed)
            bins.append({"mean_probability": mean, "observed_ai_rate": observed, "n": int(mask.sum())})
    return {"n": int(len(y)), "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)) if both else None,
        "precision": float(precision_score(y, prediction, zero_division=0)) if tp+fp else None,
        "recall": float(recall_score(y, prediction, zero_division=0)) if tp+fn else None,
        "f1": float(f1_score(y, prediction, zero_division=0)) if tp+fp+fn else None,
        "macro_f1": float(f1_score(y, prediction, labels=[0, 1], average="macro", zero_division=0)) if both else None,
        "mcc": float(matthews_corrcoef(y, prediction)) if both else None,
        "roc_auc": float(roc_auc_score(y, p)) if both else None,
        "pr_auc": float(average_precision_score(y, p)) if both else None,
        "brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "human_false_positive_rate": float(fp / (tn + fp)) if tn + fp else None,
        "ai_false_negative_rate": float(fn / (fn + tp)) if fn + tp else None,
        "ece": ece, "calibration_bins": bins,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]], "threshold": float(threshold)}


def selective_metrics(y, p, threshold, margin):
    y, p = np.asarray(y), np.asarray(p)
    confident = np.abs(p-threshold) >= margin
    return {"n_scored": len(y), "n_confident": int(confident.sum()), "n_uncertain": int((~confident).sum()),
        "coverage": float(confident.mean()), "uncertain_margin": float(margin),
        "confident_accuracy": float(accuracy_score(y[confident], p[confident] >= threshold)) if confident.any() else None,
        "note": "Primary classification metrics use forced binary labels. This table measures the app's abstention policy."}


def choose_threshold(y, p):
    # Only validation labels may be supplied here. A fixed grid avoids arbitrary precision.
    thresholds = np.linspace(.1, .9, 81)
    return float(max(thresholds, key=lambda t: (f1_score(y, p >= t, average="macro", zero_division=0), -abs(t - .5))))


def bootstrap_intervals(y, p, groups, threshold, seed=42, repetitions=200):
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    unique = np.unique(groups)
    if len(unique) < 2:
        return {}
    indices = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    values = {key: [] for key in ["accuracy", "macro_f1", "f1", "roc_auc", "brier"]}
    for _ in range(repetitions):
        sample = np.concatenate([indices[g] for g in rng.choice(unique, len(unique), replace=True)])
        metrics = binary_metrics(y[sample], p[sample], threshold)
        for key in values:
            if metrics[key] is not None:
                values[key].append(metrics[key])
    return {key: [float(v) for v in np.quantile(value, [.025, .975])] for key, value in values.items() if value}


def subgroup_metrics(df, probabilities, threshold):
    result = {}
    y = (df.label == "ai").astype(int).to_numpy()
    p = np.asarray(probabilities)
    for column in ["generator", "source", "genre", "writer_subgroup"]:
        if column not in df:
            continue
        result[column] = {}
        for value in sorted(df[column].dropna().astype(str).unique()):
            mask = (df[column].astype(str) == value).to_numpy()
            result[column][value] = binary_metrics(y[mask], p[mask], threshold)
    return result
