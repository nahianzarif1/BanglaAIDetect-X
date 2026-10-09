"""Supported models, separate from the inventory of trained artifacts."""
CLASSICAL = {
    "word_lr": "Word TF-IDF · logistic regression",
    "char_lr": "Character TF-IDF · logistic regression",
    "word_char_lr": "Word + character TF-IDF · logistic regression",
    "style_lr": "Stylometry · logistic regression",
    "fusion_lr": "Word + character + stylometry · logistic regression",
    "linear_svm": "Word + character · calibrated linear SVM",
    "naive_bayes": "Word + character · Complement Naive Bayes",
}
TRANSFORMERS = {
    "banglabert": {"name": "BanglaBERT", "checkpoint": "csebuetnlp/banglabert", "normalizer": "buet", "minimum_free_gb": 1.5},
    "banglabert_small": {"name": "BanglaBERT small", "checkpoint": "csebuetnlp/banglabert_small", "normalizer": "buet", "minimum_free_gb": .3},
    "mbert": {"name": "Multilingual BERT", "checkpoint": "google-bert/bert-base-multilingual-cased", "normalizer": "project", "minimum_free_gb": 2.},
    "xlm_roberta": {"name": "XLM-RoBERTa", "checkpoint": "FacebookAI/xlm-roberta-base", "normalizer": "project", "minimum_free_gb": 3.},
}
