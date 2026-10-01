# BanglaAIDetect-X — Starter Kit

CSE 4120 (NLP Lab), KUET. This is a runnable Level-1 skeleton for the full
guideline: dataset → preprocessing → TF-IDF baseline → BanglaBERT →
inference → Streamlit demo. Every script here has been executed end-to-end
on synthetic data and produces no errors.

**🚀 Quick Start**: See `SETUP_GUIDE.md` for step-by-step setup instructions.

## 0. Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Corpora (small, real, free)

You need TWO kinds of Bangla text: human-written and AI-generated,
ideally on the *same topics* (matched-topic design — see guideline PART 2.1).
Two real options below; pick based on your time budget.

### Option A — a ready-made matched human+AI dataset (fastest)
**"Bangla AI-Generated and Human-Written Text Dataset"** (IEEE DataPort,
DOI: 10.21227/8cmg-6267, by Farjana, Abir & Ferdosy). Both human-written
and GPT-generated Bangla text, already labeled, in one Excel file
(`Text`, `Generated_By`, `Text_Generation` columns).
- Page: https://ieee-dataport.org/documents/bangla-ai-generated-and-human-written-text-dataset
- **How to get it:** open the link → "Download" (free IEEE DataPort account
  required, no payment for this dataset) → you get an `.xlsx` file.
- **How to use it:** load with `pandas.read_excel(...)`, rename columns to
  match this project's schema (`text`, `label` where `Generated_By` "Human"→
  `human`/"AI"→`ai`, `generator`="human" or "gpt"), assign a `topic_id` per
  row (if the source doesn't give one, cluster/bucket by subject or just
  number rows in matched pairs), then feed the result into
  `src/data/cleaner.py` → `src/data/splitter.py` as your `collected_raw.csv`.

### Option B — pure human corpus + generate your own AI half (matched-topic, best practice)
**Bangla News Dataset** (Mendeley Data, Aisha Khatun et al., 28.5M+ tokens,
12 topics, Shahjalal University of Science and Technology) — real
human-written Bangla newspaper articles, free download, no login needed.
- Page: https://data.mendeley.com/datasets/xp92jxr8wn/2
- **How to get it:** open the link → "Download all files" (a zip of the
  crawled articles by topic).
- **How to get the matching AI half:** for each human article you sample,
  take its headline/topic and prompt an LLM (e.g. via the Anthropic or
  OpenAI API, or ChatGPT/Gemini web UI) with something like:
  *"এই বিষয়ে বাংলায় প্রায় ২০০-৩০০ শব্দের একটি সংবাদ প্রতিবেদন লিখুন: `<topic>`"*
  (≈"Write a ~200-300 word Bangla news report on this topic: `<topic>`").
  Save each output as `data/raw/ai/<generator>/<topic_id>.txt` and the
  matching human article as `data/raw/human/<topic_id>.txt` — this is
  exactly the layout `src/data/collector.py` expects. Do this for 2-3
  generators (e.g. gpt, gemini, claude) to support the attribution task
  in Level 3.

Either way, run:
```bash
python -m src.data.collector    # merges data/raw/human + data/raw/ai -> collected_raw.csv
python -m src.data.cleaner      # normalizes text, drops <50-token rows -> data/processed/dataset.csv
python -m src.data.splitter     # topic-based train/val/test split, seed=42
python -m src.data.validator    # sanity + leakage checks -> results/reports/validation_report.txt
```

**Quick Setup Options:**
- `python create_sample_dataset.py` - Creates sample data for testing
- `python run_pipeline.py` - Runs the complete pipeline automatically
- `python setup_ieee_dataset.py` - Helper for IEEE DataPort dataset setup

## 2. Train the Level-1 baseline (do this before BanglaBERT)

```bash
python -m src.models.baseline
```
Saves `models/tfidf_vectorizer.joblib`, `models/baseline_logreg.joblib`,
and `results/reports/baseline_metrics.json` (precision/recall/F1/ROC-AUC/PR-AUC
on validation and test — never accuracy alone, per guideline 5.5).

## 3. Fine-tune BanglaBERT (once the baseline works)

```bash
python -m src.models.banglabert
```
Fine-tunes `csebuetnlp/banglabert` with early stopping on F1, saves to
`models/banglabert_detector/`.

## 4. Run inference / the demo app

```bash
python -m src.inference                 # quick CLI sanity check
streamlit run app/app.py                # interactive demo
```

## 5. Tests

```bash
pytest tests/ -q
```

## Project layout

See `config.yaml` for every tunable value (seed, thresholds, hyperparameters —
never hard-coded in the `.py` files). Folder structure follows the full
BanglaAIDetect-X guideline (`src/data`, `src/preprocessing`, `src/models`,
`app`, `tests`, `results/reports`). Levels 2-4 (linguistic/statistical
branches, fusion, attribution, explainability, fairness, robustness) build
on top of this same skeleton — add one module at a time and re-run
`pytest tests/` after each addition, per guideline 5.7.
