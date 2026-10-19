"""Real offline neural save/reload and optimizer tests using a tiny local encoder."""
from pathlib import Path
import shutil
import numpy as np
import pytest


def test_optional_transformer_modes_train_and_reload(tmp_path, corpus, config, monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("normalizer")
    from src.models.catalog import TRANSFORMERS
    from src.models.transformer import train_transformer, TransformerPredictor
    pretrained = tmp_path / "tiny_pretrained"
    pretrained.mkdir()
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "এটি", "বাংলাদেশের", "ঢাকা", "একটি", "গুরুত্বপূর্ণ", "।"]
    (pretrained / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    tokenizer = transformers.BertTokenizerFast(vocab_file=str(pretrained / "vocab.txt"), do_lower_case=False)
    tokenizer.save_pretrained(pretrained)
    model = transformers.ElectraModel(transformers.ElectraConfig(vocab_size=len(vocab), embedding_size=16,
        hidden_size=16, intermediate_size=24, num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=64))
    model.save_pretrained(pretrained)
    cache = Path(config["paths"]["models"]).parent / ".cache/huggingface"
    cache.mkdir(parents=True)
    cached_weights = cache / "cached_model.safetensors"
    shutil.copy2(pretrained / "model.safetensors", cached_weights)
    import transformers.utils.hub as hub
    monkeypatch.setattr(hub, "cached_file", lambda *args, **kwargs: str(cached_weights))
    monkeypatch.setitem(TRANSFORMERS, "banglabert", {"name": "Tiny offline test encoder", "checkpoint": str(pretrained), "normalizer": "buet", "minimum_free_gb": 0})
    config["transformers"].update({"device": "cpu", "max_length": 32, "batch_size": 2, "epochs": 1})
    for mode in ["embeddings", "finetune"]:
        predictor, metadata = train_transformer("banglabert", corpus.iloc[:12], corpus.iloc[12:], config, tmp_path / mode, mode)
        probability = predictor.predict_proba(corpus.text.iloc[:2])
        assert probability.shape == (2, 2) and np.isfinite(probability).all()
        assert np.allclose(probability.sum(axis=1), 1)
        assert np.allclose(TransformerPredictor(tmp_path / mode, "cpu").predict_proba(corpus.text.iloc[:2]), probability)
        assert metadata["mode"] == mode and metadata["training_rows"] == 12
        assert predictor.truncation_count("এটি " * 100) > 0
        if mode == "finetune":
            assert len(metadata["history"]) == 1
        else:
            assert (tmp_path / mode / "model.safetensors").stat().st_ino == cached_weights.stat().st_ino
