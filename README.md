# BanglaAIDetect-X — Enhanced AI Detection System

CSE 4120 (NLP Lab), KUET. Enhanced Bangla-only decision-support tool for **human vs AI** text with fusion modeling and comprehensive analysis.

English and Banglish are **out of scope** (untested). The app rejects them.

## ✨ New Features
- **🔬 Fusion Model**: TF-IDF + 22 stylometric features with isotonic calibration
- **🎨 Enhanced UI**: Beautiful, optimized Streamlit interface with gradient design
- **📁 Dataset Management**: Upload and manage custom datasets through the app
- **🔧 Feature Extensibility**: Modular framework for adding custom features
- **⚖️ Turnitin Comparison**: Detailed comparison with commercial detection tools
- **📊 Comprehensive Metrics**: Brier score, calibration curves, confidence bands

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py
streamlit run app/app_enhanced.py  # Enhanced app
# OR: streamlit run app/app.py  # Original app
```

Open **Detect**, click a sample (Wikipedia-style vs ChatGPT-style), then paste your own Bangla paragraph.

## 📚 Documentation
- **SETUP_GUIDE.md**: Step-by-step setup instructions
- **PIPELINE_DOCUMENTATION.md**: Complete pipeline documentation and technical details

## Why scores used to look wrong

A 20-row toy set can hit **100% F1** and still fail in the app:

- Wikipedia is formal → the old model treated it as AI.
- New ChatGPT text had almost no overlapping words → probability sat near **50%**. That is “no evidence”, not “half human”.

This repo now trains a **fusion** model: word/char TF-IDF **plus** stylometry (burstiness, Bangla discourse markers, years/numbers, formulaic openings). Near 50% is labelled **uncertain**.

This is **not Turnitin**. Turnitin searches a huge paper index and commercial AI detectors; this is a lab prototype. Do not fail a student on the score alone.

## Pipeline

```
create_sample_dataset.py   # or IEEE DataPort xlsx via src/data/collector
        ↓
src/data/cleaner.py        # NFC, drop short / non-Bangla
        ↓
src/data/splitter.py       # split by topic_id, seed=42
        ↓
src/data/validator.py      # leakage + schema
        ↓
src/models/baseline.py     # fusion LogReg
        ↓
src/inference.py  →  streamlit run app/app.py
```

IEEE DataPort (optional, stronger data):  
https://ieee-dataport.org/documents/bangla-ai-generated-and-human-written-text-dataset  
Save as `data/raw/ieee_dataport_dataset.xlsx` and re-run `python run_pipeline.py`.

BanglaBERT is optional: `python run_pipeline.py --with-banglabert`

## Metrics

Never headline accuracy. Use precision / recall / F1 / ROC-AUC / Brier on the **topic-held-out** test set (`results/reports/baseline_metrics.json`).

`pytest tests/ -q`
