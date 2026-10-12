"""Train BanglaBERT using the shared experiment protocol."""
import argparse
from src.config import load_config
from src.models.catalog import TRANSFORMERS


def main():
    from src.experiment import run_experiment
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(TRANSFORMERS), default="banglabert")
    parser.add_argument("--mode", choices=["embeddings", "finetune"], default="finetune")
    args = parser.parse_args()
    run_experiment(load_config(), [args.model], args.mode)


if __name__ == "__main__":
    main()
