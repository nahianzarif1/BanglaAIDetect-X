"""tests/test_pipeline.py — run with: pytest tests/ -q"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing.normalize import normalize_text, detect_script, is_supported_bangla
from src.features.stylometry import extract_style_features, style_vector, FEATURE_NAMES


def test_empty_input():
    assert normalize_text("") == ""
    assert normalize_text(None) == ""


def test_whitespace_collapse():
    assert normalize_text("এটি    একটি   বাক্য") == "এটি একটি বাক্য"


def test_bangla_detection():
    assert detect_script("এটি সম্পূর্ণ বাংলা লেখা যেখানে কোনো ইংরেজি অক্ষর নেই একেবারেই") == "bangla"
    assert is_supported_bangla("বাংলাদেশের রাজধানী ঢাকা একটি ঘনবসতিপূর্ণ শহর।")


def test_latin_rejected():
    latin = "This English paragraph should not be scored by the Bangla detector at all."
    assert detect_script(latin) == "latin"
    assert not is_supported_bangla(latin)


def test_mixed_text():
    result = detect_script("আমি today স্কুলে গিয়েছিলাম and then বাসায় ফিরেছি")
    assert result == "mixed"


def test_stylometry_vector_size():
    vec = style_vector("ঢাকা বাংলাদেশের রাজধানী। যদিও যানজট আছে, ২০২২ সালে মেট্রো চালু হয়।")
    assert vec.shape == (len(FEATURE_NAMES),)


def test_ai_markers_increase_density():
    ai = "ঢাকা হলো একটি গুরুত্বপূর্ণ শহর। এছাড়াও এটি ভূমিকা পালন করে। অন্যদিকে চ্যালেঞ্জ আছে। সর্বোপরি উন্নয়ন অপরিহার্য।"
    hu = "ঢাকা বাংলাদেশের রাজধানী। ২০২২ সালের হিসাব অনুযায়ী জনসংখ্যা বেশি, যদিও যানজট নিত্য সমস্যা।"
    assert extract_style_features(ai)["ai_marker_density"] > extract_style_features(hu)["ai_marker_density"]



def test_numbers_are_not_bangla_prose():
    assert not is_supported_bangla("১ ২ ৩ ৪ ৫ ৬ ৭ ৮ ৯ ১০ ১১")
    assert detect_script("১২৩৪৫৬৭৮৯") == "other"


def test_other_scripts_count_toward_language_ratio():
    assert not is_supported_bangla("বাংলা " + "пример текста на русском языке " * 10)
    assert detect_script("हिन्दी भाषा") == "other"
