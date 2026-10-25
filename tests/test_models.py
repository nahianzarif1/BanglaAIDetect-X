import json
from pathlib import Path
import joblib
import numpy as np
import pytest
from src.inference import BanglaAIDetector, _confidence_band
from src.models.catalog import CLASSICAL
from src.models.classical import train_classical
from src.models.registry import available_models
from src.research import binary_metrics, bootstrap_intervals, write_json


@pytest.mark.parametrize("model_id", list(CLASSICAL))
def test_real_model_training_save_and_inference(model_id, corpus, config):
    pipe = train_classical(model_id, corpus, config)
    p = pipe.predict_proba(corpus.text)
    assert p.shape == (len(corpus), 2)
    assert np.isfinite(p).all() and np.allclose(p.sum(axis=1), 1)
    assert np.mean((p[:, 1] >= .5) == corpus.label.eq("ai")) > .75
    model_path = Path(config["paths"]["models"])
    model_path.mkdir(parents=True)
    joblib.dump(pipe, model_path / "trained.joblib")
    write_json(model_path / "registry.json", {"schema_version": 1, "default_model": model_id, "models": {
        model_id: {"kind": "sklearn", "name": model_id, "artifact": "trained.joblib", "threshold": .65}}})
    detector = BanglaAIDetector(config)
    result = detector.predict(corpus.text.iloc[0])
    assert "error" not in result
    assert result["model"] == model_id and result["threshold"] == .65
    assert result["ai_probability"] + result["human_probability"] == pytest.approx(1)
    assert "error" in detector.predict("English only passage that must be rejected")
    assert "error" in detector.predict("")
    batch = detector.predict_many([corpus.text.iloc[0], None, "সংক্ষিপ্ত"])
    assert "error" not in batch[0] and "error" in batch[1] and "error" in batch[2]


def test_uncertainty_uses_selected_threshold():
    assert _confidence_band(.65, .08, .65) == "low"
    assert _confidence_band(.5, .08, .65) == "medium"


def test_metrics_and_group_bootstrap_are_valid():
    y, p = np.array([0, 0, 1, 1]), np.array([0, .3, .7, 1])
    metrics = binary_metrics(y, p)
    assert metrics["accuracy"] == 1 and metrics["confusion_matrix"] == [[2, 0], [0, 2]]
    assert sum(b["n"] for b in metrics["calibration_bins"]) == 4
    intervals = bootstrap_intervals(y, p, ["a", "b", "a", "b"], .5, repetitions=20)
    assert intervals["accuracy"] == [1, 1]
    assert binary_metrics([0, 0], [.1, .2])["roc_auc"] is None
    json.dumps(metrics, allow_nan=False)
    with pytest.raises(ValueError):
        binary_metrics(y, [float("nan")]*4)


def test_missing_model_has_actionable_error(config):
    with pytest.raises(FileNotFoundError, match="run_pipeline"):
        BanglaAIDetector(config)
    assert available_models(config) == {}


def test_paired_comparison_detects_large_effect():
    from src.paired_comparison import paired_comparison
    y = np.tile([0, 1], 10)
    good, bad = y*.8+.1, 1-(y*.8+.1)
    result = paired_comparison(y, good, bad, .5, .5, np.arange(len(y)), repetitions=100)
    assert result["accuracy_difference"] == 1
    assert result["paired_group_permutation_p"] < .05
    assert result["macro_f1_difference_95pct_group_ci"] == [1, 1]
