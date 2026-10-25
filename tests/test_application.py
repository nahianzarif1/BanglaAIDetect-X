from pathlib import Path
from streamlit.testing.v1 import AppTest
from src.data.splitter import split_by_topic
from src.experiment import run_experiment


def test_application_boots_and_analyzes_sample(corpus, config, monkeypatch):
    import src.config as configuration
    splits = split_by_topic(corpus, {"train": .6, "validation": .2, "test": .2}, 42)
    for name in ["train", "validation", "test"]:
        path = Path(config["paths"][name])
        path.parent.mkdir(parents=True)
        splits.loc[splits.split == name].to_csv(path, index=False)
    run_experiment(config, ["word_lr", "char_lr"])
    monkeypatch.setattr(configuration, "load_config", lambda: config)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app/app.py"), default_timeout=30).run()
    assert not app.exception
    app.selectbox[0].set_value("word_lr").run()
    assert not app.exception
    next(button for button in app.button if button.label == "Wikipedia-style · ঢাকা").click().run()
    assert len(app.text_area[0].value.split()) > 10
    next(button for button in app.button if button.label == "Analyze text").click().run()
    assert not app.exception and len(app.metric) >= 3 and not app.error
    next(button for button in app.button if button.label == "Compare selected models").click().run()
    assert not app.exception
    app.text_area[0].set_value("English input must be rejected by the Bangla detector.").run()
    next(button for button in app.button if button.label == "Analyze text").click().run()
    assert app.error and not app.exception


def test_application_handles_missing_models(config, monkeypatch):
    import src.config as configuration
    monkeypatch.setattr(configuration, "load_config", lambda: config)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app/app.py"), default_timeout=30).run()
    assert not app.exception
    assert any("No trained models" in warning.value for warning in app.warning)
