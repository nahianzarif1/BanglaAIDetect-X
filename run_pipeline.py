"""Single entry point for ingestion, validated splits, and model experiments."""
import argparse
import sys
from pathlib import Path
from src.config import load_config
from src.models.catalog import CLASSICAL, TRANSFORMERS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--dataset", help="Labeled CSV or Excel; defaults to the supplied Excel file")
    parser.add_argument("--models", nargs="+", choices=list(CLASSICAL)+list(TRANSFORMERS), help="Default: all seven classical models")
    parser.add_argument("--with-banglabert", action="store_true")
    parser.add_argument("--all-models", action="store_true", help="Seven classical models plus BanglaBERT, mBERT, and XLM-R")
    parser.add_argument("--transformer-mode", choices=["embeddings", "finetune"], default="embeddings")
    parser.add_argument("--reuse-splits", action="store_true", help="Keep existing partitions when adding models to an experiment")
    parser.add_argument("--resume-artifacts", help="Recover completed transformer artifacts from a failed run; requires matching split hashes/mode/seed")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--refresh-only", action="store_true", help="Rebuild evaluations/ensemble from saved predictions without retraining or changing splits")
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
    except (OSError, ValueError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1
    if args.seed is not None:
        if args.seed < 0:
            parser.error("--seed must be nonnegative")
        cfg["seed"] = args.seed
    selected = args.models or list(CLASSICAL)
    if args.all_models:
        selected = list(CLASSICAL)+["banglabert", "mbert", "xlm_roberta"]
    if args.with_banglabert and "banglabert" not in selected:
        selected.append("banglabert")
    if args.refresh_only:
        selected = []
    try:
        if not args.reuse_splits and not args.refresh_only:
            from load_custom_dataset import load_custom_dataset
            from src.data.cleaner import main as clean
            from src.data.splitter import main as split
            dataset = Path(args.dataset) if args.dataset else Path(cfg["paths"]["processed"]).parents[1] / "raw/ieee_dataport_dataset.xlsx"
            if not dataset.exists():
                raise FileNotFoundError(f"Labeled dataset missing: {dataset}. Supply --dataset; synthetic data is never silently substituted.")
            load_custom_dataset(dataset, cfg)
            clean(cfg)
            split(cfg)
        from src.experiment import run_experiment
        run_experiment(cfg, selected, args.transformer_mode, args.resume_artifacts)
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        return 1
    print("Launch: .venv/bin/streamlit run app/app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
