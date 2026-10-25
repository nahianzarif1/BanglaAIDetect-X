# Setup

The maintained setup and commands are in [README.md](README.md).

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_pipeline.py
streamlit run app/app.py
```

Optional models require `requirements-transformers.txt`. The actual application
entry point is `app/app.py`. See [RESEARCH_GUIDE.md](RESEARCH_GUIDE.md) before
interpreting a high score as evidence of generalization.
