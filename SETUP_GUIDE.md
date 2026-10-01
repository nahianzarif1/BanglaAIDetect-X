# BanglaAIDetect-X - Setup and Run Guide

This guide will help you set up and run the BanglaAIDetect-X project for detecting AI-generated Bangla text.

## Quick Start

### Option 1: Using Sample Data (Fastest)

```bash
# 1. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create sample dataset
python create_sample_dataset.py

# 4. Run the complete pipeline
python run_pipeline.py

# 5. Run the Streamlit app
streamlit run app/app.py
```

### Option 2: Using IEEE DataPort Dataset

```bash
# 1. Download the dataset
# Visit: https://ieee-dataport.org/documents/bangla-ai-generated-and-human-written-text-dataset
# Create a free IEEE DataPort account and download the .xlsx file

# 2. Place the dataset
# Move the downloaded .xlsx file to: data/raw/ieee_dataport_dataset.xlsx

# 3. Run setup and pipeline
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.data.collector  # This will detect and use the IEEE dataset
python -m src.data.cleaner
python -m src.data.splitter
python -m src.data.validator
python -m src.models.baseline
streamlit run app/app.py
```

## Manual Step-by-Step Setup

### 1. Environment Setup

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Preparation

#### Using Sample Data (for testing)
```bash
python create_sample_dataset.py
```

#### Using IEEE DataPort Dataset (for production)
```bash
# Download from IEEE DataPort and place in data/raw/ieee_dataport_dataset.xlsx
python -m src.data.collector
```

### 3. Data Processing

```bash
# Clean and normalize data
python -m src.data.cleaner

# Split into train/validation/test
python -m src.data.splitter

# Validate dataset
python -m src.data.validator
```

### 4. Model Training

#### TF-IDF Baseline (Required)
```bash
python -m src.models.baseline
```

#### BanglaBERT (Optional - requires more dependencies)
```bash
# First install additional dependencies
pip install torch transformers datasets

# Then train
python -m src.models.banglabert
```

### 5. Testing Inference

```bash
# Test the inference pipeline
python -m src.inference
```

### 6. Running the Streamlit App

```bash
streamlit run app/app.py
```

The app will open in your browser at `http://localhost:8501`

## Project Structure

```
BanglaAIDetect-X/
├── README.md                  # Project documentation
├── SETUP_GUIDE.md            # This file
├── requirements.txt          # Python dependencies
├── config.yaml               # Configuration settings
├── create_sample_dataset.py  # Script to create sample data
├── run_pipeline.py           # Complete pipeline runner
├── setup_ieee_dataset.py    # IEEE dataset setup helper
├── data/
│   ├── raw/                  # Raw data files
│   ├── processed/            # Cleaned data
│   ├── splits/               # Train/val/test splits
│   └── metadata/             # Dataset metadata
├── src/
│   ├── data/                 # Data collection and processing
│   ├── preprocessing/        # Text normalization
│   ├── models/               # Model training
│   └── inference.py          # Inference wrapper
├── models/                   # Trained models
├── results/                  # Results and reports
├── app/
│   └── app.py               # Streamlit application
└── tests/                   # Unit tests
```

## Configuration

Edit `config.yaml` to customize:

- **Dataset settings**: Minimum token length, split ratios
- **Model settings**: Learning rates, batch sizes, epochs
- **Paths**: Input/output directories
- **Inference settings**: Decision thresholds

## Troubleshooting

### Virtual Environment Issues
```bash
# If python3 is not found, try:
python -m venv .venv

# If activation fails, try:
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows
```

### Package Installation Issues
```bash
# If packages fail to install, try:
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

### Dataset Issues
```bash
# If dataset validation fails, check:
python -m src.data.validator

# View validation report:
cat results/reports/validation_report.txt
```

### Model Training Issues
```bash
# If baseline model fails, ensure data is properly prepared:
python -m src.data.validator

# Check if splits exist:
ls data/splits/
```

## Testing

Run the test suite:
```bash
pytest tests/
```

## Next Steps

After getting the baseline working:

1. **Use Real Data**: Replace sample data with IEEE DataPort dataset
2. **Train BanglaBERT**: Install PyTorch and train the transformer model
3. **Evaluate Results**: Check metrics in `results/reports/`
4. **Customize**: Modify `config.yaml` for your needs
5. **Extend**: Add features from the full project guideline

## Additional Resources

- **Project Documentation**: See `README.md` for full project details
- **IEEE DataPort**: https://ieee-dataport.org/documents/bangla-ai-generated-and-human-written-text-dataset
- **Configuration**: Edit `config.yaml` for customization

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the validation reports in `results/reports/`
3. Ensure all dependencies are installed correctly
4. Verify dataset integrity with `python -m src.data.validator`

## Performance Notes

- **Sample Dataset**: 20 samples (10 human, 10 AI) - for testing only
- **IEEE Dataset**: Much larger - use for production
- **Baseline Model**: Fast, lightweight, good for quick results
- **BanglaBERT**: Slower, more accurate, requires GPU for best performance

## System Requirements

- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (8GB recommended for BanglaBERT)
- **Storage**: 500MB for sample data, more for real datasets
- **GPU**: Optional (recommended for BanglaBERT training)