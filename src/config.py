"""One configuration loader; paths are independent of the launch directory."""
from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    path = Path(path or ROOT / "config.yaml").resolve()
    try:
        with path.open(encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML configuration: {path}") from exc
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    cfg = deepcopy(cfg)
    required = {"paths", "dataset", "baseline", "inference", "seed"}
    if missing := required-set(cfg):
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")
    if any(not isinstance(cfg[key], dict) for key in required-{"seed"}):
        raise ValueError("Configuration sections must be mappings.")
    needed_paths = {"processed", "train", "validation", "test", "reports", "models"}
    if not needed_paths.issubset(cfg["paths"]):
        raise ValueError("Configuration paths require processed, train, validation, test, reports, and models.")
    if not isinstance(cfg["seed"], int) or cfg["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer.")
    minimum = cfg["dataset"].get("min_tokens")
    if not isinstance(minimum, int) or minimum < 1:
        raise ValueError("min_tokens must be a positive integer.")
    for key, value in cfg["paths"].items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Invalid path for {key}.")
        cfg["paths"][key] = str((path.parent / value).resolve())
    threshold = cfg["inference"].get("decision_threshold")
    if not isinstance(threshold, (int, float)) or not 0 < threshold < 1:
        raise ValueError("decision_threshold must be between zero and one.")
    margin = cfg["inference"].get("uncertain_margin", .08)
    if not isinstance(margin, (int, float)) or not 0 <= margin < .5:
        raise ValueError("uncertain_margin must be between zero and 0.5.")
    return cfg
