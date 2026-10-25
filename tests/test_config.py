from pathlib import Path
import pytest
import yaml
from src.config import ROOT, load_config
from run_pipeline import main


def test_config_paths_do_not_depend_on_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    assert Path(cfg["paths"]["models"]).parent == ROOT


@pytest.mark.parametrize("invalid", ["[]", "broken: [", "seed: 42"])
def test_invalid_configuration_is_actionable(tmp_path, invalid):
    path = tmp_path / "config.yaml"
    path.write_text(invalid)
    with pytest.raises(ValueError):
        load_config(path)
    assert main(["--config", str(path)]) == 1


def test_missing_configuration_returns_failure(tmp_path):
    assert main(["--config", str(tmp_path / "missing.yaml")]) == 1
