"""
src/data/collector.py

Supports two data collection modes:
1. Raw text files from data/raw/human/<topic>.txt and data/raw/ai/<generator>/<topic>.txt
2. IEEE DataPort Excel dataset (Bangla AI-Generated and Human-Written Text Dataset)

For IEEE DataPort dataset:
- Page: https://ieee-dataport.org/documents/bangla-ai-generated-and-human-written-text-dataset
- Columns: Text, Generated_By, Text_Generation
- Maps to our schema: text, label (Human->human, AI->ai), generator (human or gpt)
"""

import os
import uuid
import yaml
import pandas as pd


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def collect_human(raw_human_dir: str, genre: str = "unspecified",
                   language_type: str = "bangla") -> list:
    rows = []
    if not os.path.isdir(raw_human_dir):
        return rows
    for fname in sorted(os.listdir(raw_human_dir)):
        if not fname.endswith(".txt"):
            continue
        topic_id = os.path.splitext(fname)[0]
        text = _read_txt(os.path.join(raw_human_dir, fname))
        rows.append({
            "id": str(uuid.uuid4()),
            "text": text,
            "label": "human",
            "generator": "human",
            "genre": genre,
            "language_type": language_type,
            "edit_type": "none",
            "topic_id": topic_id,
            "writer_subgroup": "standard",   # fill in properly from your metadata source
            "split": None,
        })
    return rows


def collect_ai(raw_ai_dir: str, genre: str = "unspecified",
                language_type: str = "bangla") -> list:
    rows = []
    if not os.path.isdir(raw_ai_dir):
        return rows
    for generator in sorted(os.listdir(raw_ai_dir)):
        gen_dir = os.path.join(raw_ai_dir, generator)
        if not os.path.isdir(gen_dir):
            continue
        for fname in sorted(os.listdir(gen_dir)):
            if not fname.endswith(".txt"):
                continue
            topic_id = os.path.splitext(fname)[0]
            text = _read_txt(os.path.join(gen_dir, fname))
            rows.append({
                "id": str(uuid.uuid4()),
                "text": text,
                "label": "ai",
                "generator": generator,
                "genre": genre,
                "language_type": language_type,
                "edit_type": "none",
                "topic_id": topic_id,
                "writer_subgroup": None,   # not applicable to AI rows
                "split": None,
            })
    return rows


def collect_ieee_dataport(excel_path: str, genre: str = "unspecified",
                           language_type: str = "bangla") -> list:
    """
    Load IEEE DataPort Excel dataset and convert to our schema.
    Expected columns: Text, Generated_By, Text_Generation
    """
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"IEEE DataPort Excel file not found: {excel_path}")

    df = pd.read_excel(excel_path)
    
    # Map IEEE columns to our schema
    # Assuming columns are: Text, Generated_By, Text_Generation
    # Generated_By: "Human" -> human, "AI" -> ai
    # Text_Generation: additional info (we can ignore or use for metadata)
    
    rows = []
    for idx, row in df.iterrows():
        text = str(row.get("Text", ""))
        generated_by = str(row.get("Generated_By", "")).strip()
        
        # Map label
        if generated_by.lower() == "human":
            label = "human"
            generator = "human"
        elif generated_by.lower() == "ai":
            label = "ai"
            generator = "gpt"  # assume GPT as default
        else:
            # Skip if label is unclear
            continue
        
        # Create topic_id from index (since we don't have real topics)
        topic_id = f"topic_{idx}"
        
        rows.append({
            "id": str(uuid.uuid4()),
            "text": text,
            "label": label,
            "generator": generator,
            "genre": genre,
            "language_type": language_type,
            "edit_type": "none",
            "topic_id": topic_id,
            "writer_subgroup": "standard" if label == "human" else None,
            "split": None,
        })
    
    return rows


def main():
    cfg = load_config()
    rows = []
    
    # Check if IEEE DataPort Excel file exists
    ieee_excel_path = "data/raw/ieee_dataport_dataset.xlsx"
    if os.path.exists(ieee_excel_path):
        print(f"Loading IEEE DataPort dataset from {ieee_excel_path}")
        rows = collect_ieee_dataport(ieee_excel_path)
    else:
        # Fall back to traditional text file collection
        print("IEEE DataPort Excel not found, using traditional text file collection")
        rows += collect_human(cfg["paths"]["raw_human"])
        rows += collect_ai(cfg["paths"]["raw_ai"])

    df = pd.DataFrame(rows)
    out_dir = os.path.dirname(cfg["paths"]["processed"])
    os.makedirs(out_dir, exist_ok=True)
    raw_merged_path = os.path.join(out_dir, "collected_raw.csv")
    df.to_csv(raw_merged_path, index=False)
    print(f"Collected {len(df)} rows -> {raw_merged_path}")
    print("Next step: run src/data/cleaner.py to normalize and produce data/processed/dataset.csv")


if __name__ == "__main__":
    main()
