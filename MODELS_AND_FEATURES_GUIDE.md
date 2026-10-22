# Models and features

The maintained model inventory and training commands are in [README.md](README.md).
Classical models cover word/character TF-IDF, style-only, fusion, SVM, and
Complement Naive Bayes. Neural backbones include BanglaBERT, BanglaBERT small,
mBERT, and XLM-R with supervised frozen-head and fine-tuning modes.

Writing-style cues are descriptive heuristics. They are not reliable proofs of
authorship or explanations of the transformer models. Ensemble weights and
operating thresholds use validation data; final metrics use independent test rows.

See [RESEARCH_GUIDE.md](RESEARCH_GUIDE.md) for meaningful comparisons and ablations.
