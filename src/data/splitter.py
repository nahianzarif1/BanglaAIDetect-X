"""Deterministic group splits with both classes in every partition."""
from pathlib import Path
import numpy as np
import pandas as pd
from src.config import load_config
from src.data.grouping import group_documents
from src.research import write_json


def split_by_topic(df, ratios, seed):
    if set(ratios) != {"train", "validation", "test"} or not np.isclose(sum(ratios.values()), 1) or min(ratios.values()) <= 0:
        raise ValueError("Three positive split ratios must sum to one.")
    df = df.copy()
    column = "group_id" if "group_id" in df else "topic_id"
    if column not in df or df[column].isna().any() or df[column].astype(str).str.strip().eq("").any():
        raise ValueError("All documents need a nonempty split group.")
    groups = np.array(sorted(df[column].unique()))
    if len(groups) < 3:
        raise ValueError("At least three independent document groups are required.")
    rng = np.random.default_rng(seed)
    n_train = max(1, min(len(groups) - 2, int(round(len(groups) * ratios["train"]))))
    n_val = max(1, min(len(groups) - n_train - 1, int(round(len(groups) * ratios["validation"]))))
    best, best_score = None, float("inf")
    for _ in range(128):
        shuffled = rng.permutation(groups)
        mapping = {g: name for name, subset in [("train", shuffled[:n_train]), ("validation", shuffled[n_train:n_train+n_val]), ("test", shuffled[n_train+n_val:])] for g in subset}
        candidate = df[column].map(mapping)
        score = 0.
        for name in ratios:
            subset = df.loc[candidate == name]
            if subset.empty or subset.label.nunique() != 2:
                score = float("inf")
                break
            score += abs(len(subset)/len(df) - ratios[name]) + abs(subset.label.eq("ai").mean() - df.label.eq("ai").mean())
        if score < best_score:
            best, best_score = candidate, score
    if best is None:
        raise ValueError("Cannot create three disjoint splits with both labels. Add independent groups.")
    df["split"] = best
    return df


def verify_no_leakage(df):
    for column in ["group_id", "topic_id", "parent_id", "prompt_id", "text_hash"]:
        if column in df:
            populated = df.loc[df[column].notna() & df[column].astype(str).str.strip().ne("")]
            populated = populated.loc[~populated[column].astype(str).str.lower().isin(["unknown", "none", "nan"])]
            if (populated.groupby(column).split.nunique() > 1).any():
                raise ValueError(f"DATA LEAKAGE across splits in {column}.")


def main(cfg=None):
    cfg = cfg or load_config()
    df, report = group_documents(pd.read_csv(cfg["paths"]["processed"]), cfg["dataset"].get("near_duplicate_threshold", .9))
    df = split_by_topic(df, cfg["dataset"]["split_ratio"], cfg["seed"])
    verify_no_leakage(df)
    for name in ["train", "validation", "test"]:
        path = Path(cfg["paths"][name])
        path.parent.mkdir(parents=True, exist_ok=True)
        subset = df.loc[df.split == name]
        subset.to_csv(path, index=False)
        report[name] = {"rows": len(subset), "groups": int(subset.group_id.nunique()), "labels": subset.label.value_counts().to_dict()}
    write_json(Path(cfg["paths"]["reports"]) / "split_manifest.json", report)
    print(f"Created {report['grouping']} splits: " + str({s: report[s]['rows'] for s in ['train','validation','test']}))
    return df


if __name__ == "__main__":
    main()
