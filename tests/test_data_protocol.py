import pandas as pd
import pytest
from load_custom_dataset import convert_dataset
from src.data.cleaner import clean_dataframe
from src.data.grouping import group_documents
from src.data.splitter import split_by_topic, verify_no_leakage
from src.data.validator import run_all_validations


def test_import_preserves_provenance_without_fabricating_topics():
    df = convert_dataset(pd.DataFrame({"source_text": ["লেখা", "মানুষ"], "Source": ["AI", "wiki"], "Source Name": ["Gemini", "Wikipedia"]}))
    assert df.generator.tolist() == ["Gemini", "Wikipedia"]
    assert df.topic_id.eq("").all()
    assert df.label.tolist() == ["ai", "human"]
    with pytest.raises(ValueError, match="Unrecognized"):
        convert_dataset(pd.DataFrame({"text": ["লেখা"], "label": ["probably-human"]}))


def test_cleaning_excludes_all_conflicting_labels(corpus):
    original = corpus.iloc[0].copy()
    conflicting = original.copy()
    conflicting["id"], conflicting["label"] = "conflict", "ai"
    frame = pd.concat([corpus, pd.DataFrame([conflicting])], ignore_index=True)
    kept, dropped = clean_dataframe(frame, 10)
    assert not kept.text.eq(original.text).any()
    assert dropped.drop_reason.eq("conflicting_labels").sum() == 2


def test_related_documents_stay_together(corpus):
    grouped, report = group_documents(corpus)
    splits = split_by_topic(grouped, {"train": .6, "validation": .2, "test": .2}, 42)
    verify_no_leakage(splits)
    assert (splits.groupby("topic_id").split.nunique() == 1).all()
    again = split_by_topic(grouped, {"train": .6, "validation": .2, "test": .2}, 42)
    assert splits.split.equals(again.split)


def test_leakage_and_schema_fail_closed(corpus, config):
    frame = corpus.copy()
    frame["split"] = ["train", "test"] * 9
    assert any("LEAKAGE" in issue for issue in run_all_validations(frame, config))
    assert run_all_validations(frame.drop(columns="label"), config)
    with pytest.raises(ValueError):
        split_by_topic(corpus.iloc[:2], {"train": .7, "validation": .15, "test": .15}, 42)


def test_near_duplicate_groups_without_topics(corpus):
    frame = corpus.iloc[:2].copy()
    frame["topic_id"] = ""
    passage = " ".join([f"শব্দ{i}" for i in range(100)])
    frame["text"] = [passage, passage + " অতিরিক্ত"]
    grouped, report = group_documents(frame)
    assert grouped.group_id.nunique() == 1
    assert report["grouping"] == "document-and-near-duplicate"
