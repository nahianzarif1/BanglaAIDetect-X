import json
from pathlib import Path
import numpy as np
from src.data.splitter import split_by_topic
from src.experiment import learn_weights, run_experiment
from src.models.registry import load_registry, load_predictor
from src.evaluate_external import main as evaluate_external


def test_weights_are_convex_and_use_validation_signal():
    weights = learn_weights([0, 0, 1, 1], np.array([[.05, .9], [.1, .8], [.9, .1], [.95, .05]]))
    assert weights[0] > .95
    assert weights.min() >= 0 and np.isclose(weights.sum(), 1)


def test_experiment_and_incremental_models(corpus, config):
    splits = split_by_topic(corpus, {"train": .6, "validation": .2, "test": .2}, 42)
    for name in ["train", "validation", "test"]:
        path = Path(config["paths"][name])
        path.parent.mkdir(parents=True)
        splits.loc[splits.split == name].to_csv(path, index=False)
    first = run_experiment(config, ["word_lr", "char_lr"])
    assert "ensemble" in first["models"]
    registry = load_registry(config)
    assert registry["default_model"] in registry["models"]
    ensemble = load_predictor("ensemble", config)
    assert np.isfinite(ensemble.predict_proba(corpus.text)).all()
    second = run_experiment(config, ["style_lr"])
    assert set(second["models"]) == {"word_lr", "char_lr", "style_lr", "ensemble"}
    assert first["split_hashes"] == second["split_hashes"]
    assert Path(registry["latest_report"]).exists()
    json.dumps(second, allow_nan=False)


def test_external_overlap_is_rejected(corpus, config, tmp_path, monkeypatch):
    import pytest
    import src.evaluate_external as external
    path = Path(config["paths"]["train"])
    path.parent.mkdir(parents=True)
    corpus.to_csv(path, index=False)
    uploaded = tmp_path / "external.csv"
    corpus.to_csv(uploaded, index=False)
    monkeypatch.setattr(external, "load_config", lambda: config)
    with pytest.raises(ValueError, match="overlap|known training"):
        external.main(["--dataset", str(uploaded), "--output", str(tmp_path / "external_results")])
