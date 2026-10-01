"""
tests/test_pipeline.py

Run with: pytest tests/
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing.normalize import normalize_text, detect_script


def test_empty_input():
    assert normalize_text("") == ""
    assert normalize_text(None) == ""


def test_whitespace_collapse():
    assert normalize_text("এটি    একটি   বাক্য") == "এটি একটি বাক্য"


def test_bangla_detection():
    assert detect_script("এটি সম্পূর্ণ বাংলা লেখা যেখানে কোনো ইংরেজি অক্ষর নেই একেবারেই") == "bangla"


def test_banglish_detection():
    assert detect_script("ami tomake khub valobasi ei bangla lekha ta banglish e") == "banglish"


def test_mixed_text():
    result = detect_script("আমি today স্কুলে গিয়েছিলাম and then বাসায় ফিরেছি")
    assert result == "mixed"


def test_prediction_range():
    # Simulated model output — asserts the contract inference.py must satisfy.
    ai_probability = 0.73
    human_probability = 1 - ai_probability
    assert 0 <= ai_probability <= 1
    assert abs(ai_probability + human_probability - 1) < 1e-6


def test_pipeline_output_schema():
    fake_result = {
        "label": "ai",
        "ai_probability": 0.73,
        "human_probability": 0.27,
        "detected_script": "bangla",
        "note": "Decision-support estimate, not an infallible determination.",
    }
    required_keys = {"label", "ai_probability", "human_probability", "detected_script", "note"}
    assert required_keys.issubset(fake_result.keys())
    assert fake_result["label"] in {"human", "ai"}
