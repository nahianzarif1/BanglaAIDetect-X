"""Import an Excel/CSV corpus without inventing topics or generator names."""
import argparse
import hashlib
from pathlib import Path
import pandas as pd
from src.config import load_config
from src.preprocessing.normalize import normalize_text
from src.research import file_hash, write_json


def convert_dataset(df, source_path="uploaded"):
    df = df.copy().rename(columns={"source_text": "text"})
    if "text" not in df or not ({"label", "Source"} & set(df)):
        raise ValueError("Dataset requires text/label or source_text/Source columns.")
    sources = df.get("Source", df.get("source", pd.Series("unknown", index=df.index))).fillna("unknown").astype(str)
    raw = df.get("label", sources).fillna("").astype(str).str.strip().str.lower()
    labels = raw.map({"ai": "ai", "human": "human", "ittefaq": "human", "wiki": "human"})
    if labels.isna().any():
        raise ValueError(f"Unrecognized labels/sources: {sorted(raw[labels.isna()].unique())}. Provide human/ai labels.")
    namespace = hashlib.sha256(str(source_path).encode()).hexdigest()[:8]
    result = pd.DataFrame({
        "id": df.get("id", pd.Series([f"{namespace}_{i}" for i in range(len(df))], index=df.index)),
        "text": df.text.fillna("").astype(str).map(normalize_text), "label": labels,
        "generator": df.get("generator", df.get("Source Name", sources)).fillna("unknown"),
        "source": sources, "genre": df.get("genre", sources.map({"ittefaq": "news", "wiki": "encyclopedia"}).fillna("unknown")),
        "language_type": "bangla", "edit_type": df.get("edit_type", "unknown"),
        "topic_id": df.get("topic_id", pd.Series("", index=df.index)),
        "writer_subgroup": df.get("writer_subgroup", "unknown"), "source_file": str(source_path)})
    for column in ["parent_id", "prompt_id", "author_id"]:
        if column in df:
            result[column] = df[column]
    return result


def load_custom_dataset(path=None, cfg=None):
    cfg = cfg or load_config()
    path = Path(path or Path(cfg["paths"]["processed"]).parents[1] / "raw/ieee_dataport_dataset.xlsx")
    frame = pd.read_excel(path) if path.suffix.lower() == ".xlsx" else pd.read_csv(path)
    result = convert_dataset(frame, path.name)
    destination = Path(cfg["paths"]["processed"]).parent / "collected_raw.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    write_json(Path(cfg["paths"]["reports"]) / "import_manifest.json", {
        "source_path": str(path.resolve()), "source_sha256": file_hash(path), "rows": len(result),
        "labels": result.label.value_counts().to_dict(), "generators": result.generator.value_counts().to_dict(),
        "real_topic_ids_available": bool(result.topic_id.fillna("").astype(str).str.strip().ne("").all()),
        "note": "Missing topics remain missing. Document/near-duplicate grouping cannot establish topic-held-out generalization."})
    print(f"Imported {len(result)} rows; generator and source metadata preserved.")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    load_custom_dataset(parser.parse_args().dataset)
