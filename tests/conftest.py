from pathlib import Path
import sys
import pytest
import pandas as pd
from src.config import load_config


@pytest.fixture
def corpus():
    human = [
        "গতকাল সকালে নদীর পাশে হাঁটতে গিয়ে পুরোনো বন্ধুদের সঙ্গে দেখা হলো। গ্রামের মানুষ বন্যার পানি নিয়ে চিন্তিত ছিলেন।",
        "বাংলাদেশের রাজধানী ঢাকা বুড়িগঙ্গা নদীর তীরে অবস্থিত। মুঘল আমলে এখানে প্রশাসনিক কেন্দ্র প্রতিষ্ঠিত হয়েছিল বলে ইতিহাসে উল্লেখ আছে।",
        "শিক্ষক জানান যে স্কুলের ভবন মেরামতের জন্য বরাদ্দ পাওয়া যায়নি। তবে শিক্ষার্থীরা প্রতিদিন ক্লাসে এসে পড়াশোনা চালিয়ে যাচ্ছে।",
    ]
    ai = [
        "প্রযুক্তি হলো আধুনিক সমাজের একটি গুরুত্বপূর্ণ অংশ। এছাড়াও এটি উন্নয়নে গুরুত্বপূর্ণ ভূমিকা পালন করে। সর্বোপরি সচেতনতা বৃদ্ধি অপরিহার্য।",
        "শিক্ষা জাতীয় অগ্রগতির মূল চালিকাশক্তি হিসেবে পরিচিত। অন্যদিকে বিভিন্ন চ্যালেঞ্জ মোকাবিলায় পরিকল্পিত উদ্যোগ এবং টেকসই উন্নয়ন নিশ্চিত করা প্রয়োজন।",
        "পরিবেশ রক্ষার ক্ষেত্রে সমন্বিত উদ্যোগ গ্রহণ করা অত্যন্ত গুরুত্বপূর্ণ। এছাড়াও সঠিক পরিকল্পনা এবং সচেতনতা বৃদ্ধির মাধ্যমে ইতিবাচক প্রভাব নিশ্চিত করা সম্ভব।",
    ]
    rows = []
    for i in range(18):
        label = "ai" if i % 2 else "human"
        text = (ai if label == "ai" else human)[(i//2) % 3] + f" বিষয়ক্রম {i}।"
        rows.append({"id": f"sample_{i}", "text": text, "label": label, "generator": "test_generator" if label == "ai" else "test_human",
            "source": "test", "genre": "test", "topic_id": f"topic_{i//2}", "group_id": f"group_{i//2}", "writer_subgroup": "unknown"})
    return pd.DataFrame(rows)


@pytest.fixture
def config(tmp_path):
    cfg = load_config()
    for key in cfg["paths"]:
        original = Path(cfg["paths"][key])
        cfg["paths"][key] = str(tmp_path / key / original.name) if original.suffix else str(tmp_path / key)
    cfg["baseline"]["tfidf_min_df"] = 1
    cfg["baseline"]["tfidf_max_features"] = 200
    cfg["baseline"]["char_max_features"] = 200
    cfg["evaluation"]["bootstrap_repetitions"] = 10
    return cfg
