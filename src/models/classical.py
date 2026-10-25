"""Complementary baselines and feature ablations with group-safe calibration."""
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from src.features.stylometry import StylometryTransformer


def make_pipeline(model_id, cfg):
    options = cfg["baseline"]
    word = TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False,
        ngram_range=tuple(options["tfidf_ngram_range"]), max_features=options["tfidf_max_features"],
        min_df=options.get("tfidf_min_df", 2), sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", lowercase=False,
        ngram_range=tuple(options.get("char_ngram_range", [3, 5])),
        max_features=options.get("char_max_features", 12000), min_df=options.get("tfidf_min_df", 2), sublinear_tf=True)
    style = Pipeline([("extract", StylometryTransformer()), ("scale", StandardScaler())])
    feature_names = {
        "word_lr": ["word"], "char_lr": ["char"], "word_char_lr": ["word", "char"],
        "style_lr": ["style"], "fusion_lr": ["word", "char", "style"],
        "linear_svm": ["word", "char"], "naive_bayes": ["word", "char"]}
    if model_id not in feature_names:
        raise ValueError(f"Unknown classical model: {model_id}")
    features = FeatureUnion([(name, {"word": word, "char": char, "style": style}[name]) for name in feature_names[model_id]])
    if model_id == "linear_svm":
        classifier = LinearSVC(C=options["logreg_C"], class_weight="balanced", random_state=cfg["seed"], max_iter=5000)
    elif model_id == "naive_bayes":
        classifier = ComplementNB(alpha=1.0)
    else:
        classifier = LogisticRegression(C=options["logreg_C"], class_weight="balanced", solver="liblinear",
            random_state=cfg["seed"], max_iter=options["logreg_max_iter"])
    return Pipeline([("features", features), ("classifier", classifier)])


def train_classical(model_id, frame, cfg):
    pipeline = make_pipeline(model_id, cfg)
    y = frame.label.eq("ai").astype(int).to_numpy()
    x = frame.text.to_numpy()
    # Calibration is entirely inside training, with related documents in the same fold.
    if model_id in {"linear_svm", "fusion_lr"}:
        groups = frame.get("group_id", frame.topic_id).to_numpy()
        maximum_folds = cfg["baseline"].get("calibration_folds", 3)
        class_groups = int(frame.groupby("label").group_id.nunique().min()) if "group_id" in frame else len(np.unique(groups))
        folds = min(maximum_folds, class_groups)
        if folds < 2:
            raise ValueError("Calibrated models need at least two independent groups per class.")
        cv = list(StratifiedGroupKFold(folds, shuffle=True, random_state=cfg["seed"]).split(x, y, groups))
        if any(len(np.unique(y[a])) != 2 or len(np.unique(y[b])) != 2 for a, b in cv):
            raise ValueError("Group calibration folds need both labels. Add independent training groups.")
        pipeline = CalibratedClassifierCV(pipeline, method=cfg["baseline"].get("calibration_method", "sigmoid"), cv=cv)
    return pipeline.fit(x, y)
