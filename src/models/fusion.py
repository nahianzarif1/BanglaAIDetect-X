"""Compatibility alias: one shared fusion implementation, no divergent training."""
from src.models.baseline import load_config, load_splits, to_xy, evaluate, main
from src.models.classical import make_pipeline, train_classical


def create_fusion_pipeline(cfg):
    return make_pipeline("fusion_lr", cfg)


def train_fusion(train_df, cfg):
    return train_classical("fusion_lr", train_df, cfg)


if __name__ == "__main__":
    main()
