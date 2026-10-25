"""Compatibility entry point for the research baseline experiment."""
from sklearn.metrics import classification_report
from src.config import load_config
from src.models.classical import make_pipeline, train_classical
from src.research import binary_metrics


def load_splits(cfg):
    import pandas as pd
    return tuple(pd.read_csv(cfg["paths"][name]) for name in ["train", "validation", "test"])


def to_xy(df, cfg):
    return df[cfg["dataset"]["text_col"]].fillna(""), df[cfg["dataset"]["label_col"]].eq("ai").astype(int)


def build_pipeline(cfg):
    return make_pipeline("fusion_lr", cfg)


def train_baseline(train_df, cfg):
    return train_classical("fusion_lr", train_df, cfg)


def evaluate(pipe, df, cfg):
    x, y = to_xy(df, cfg)
    p = pipe.predict_proba(x)[:, 1]
    threshold = cfg["inference"]["decision_threshold"]
    return binary_metrics(y, p, threshold), classification_report(y, p >= threshold, labels=[0, 1], target_names=["human", "ai"], zero_division=0)


def main():
    from src.experiment import run_experiment
    run_experiment(load_config(), ["fusion_lr"])


if __name__ == "__main__":
    main()
