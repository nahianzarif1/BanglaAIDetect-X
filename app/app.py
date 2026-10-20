"""BanglaAIDetect-X · multi-model application and experiment explorer."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
from app.samples import DEMOS
from src.config import load_config
from src.inference import BanglaAIDetector
from src.models.catalog import CLASSICAL, TRANSFORMERS
from src.models.registry import available_models, load_registry

st.set_page_config(page_title="BanglaAIDetect-X", page_icon="🔎", layout="wide")
st.markdown("""<style>
.block-container{max-width:1250px;padding-top:2rem}
h1,h2,h3{letter-spacing:-.025em}
[data-testid="stMetric"]{border:1px solid #79969b40;border-radius:12px;padding:16px}
</style>""", unsafe_allow_html=True)
CFG = load_config()
SAMPLES = {sample["title"]: sample["text"] for sample in DEMOS}


@st.cache_resource(max_entries=2)
def get_detector(model_id, registry_revision):
    return BanglaAIDetector(CFG, model_id)


def set_sample(name):
    st.session_state["input_text"] = SAMPLES[name]
    st.session_state.pop("analysis_result", None)


def clear_analysis():
    st.session_state.pop("analysis_result", None)


def clear_batch():
    st.session_state.pop("batch_output", None)


def render_result(result):
    if "error" in result:
        st.error(result["error"])
        return
    label = {"ai": "AI-like writing", "human": "Human-like writing", "uncertain": "Uncertain — review needed"}[result["label"]]
    st.subheader(label)
    a, b, c = st.columns(3)
    a.metric("AI model score", f"{result['ai_probability']:.1%}")
    b.metric("Human model score", f"{result['human_probability']:.1%}")
    c.metric("Words", result["token_count"])
    st.caption(f"{result['model_name']} · threshold {result['threshold']:.2f} · score separation {result['score_separation']}")
    st.info(result["note"])
    with st.expander("Writing style observations"):
        st.caption("Descriptive heuristics; these observations do not explain every model's prediction or prove authorship.")
        if result["style"]:
            st.dataframe(pd.DataFrame([result["style"]]).T.rename(columns={0: "value"}), width="stretch")
        for cue in result["cues"]:
            st.write(f"**{cue.get('title', 'Observation')}** — {cue.get('detail', '')}")
    st.download_button("Download analysis JSON", json.dumps(result, ensure_ascii=False, indent=2), "analysis.json", "application/json")


def main():
    st.caption("BANGLA LANGUAGE RESEARCH · AUTHORSHIP ANALYSIS")
    st.title("BanglaAIDetect-X")
    st.write("Compare complementary models, analyze Bangla passages, and inspect reproducible experiments.")
    try:
        inventory, registry = available_models(CFG), load_registry(CFG)
    except (OSError, ValueError) as exc:
        st.error(f"Cannot read model inventory: {exc}")
        return
    ready = {key: value for key, value in inventory.items() if value["available"]}
    with st.sidebar:
        st.subheader("Model selection")
        if ready:
            ids = list(ready)
            default = registry["default_model"] if registry["default_model"] in ready else ids[0]
            model_id = st.selectbox("Trained detector", ids, index=ids.index(default), format_func=lambda key: ready[key]["name"])
            st.caption("Default selected by validation macro-F1.")
            st.caption(f"Training mode: {ready[model_id]['kind']}")
        else:
            model_id = None
        st.caption("Only locally trained models appear here.")
        st.write(f"{len(ready)} trained models available")
    if not ready:
        st.warning("No trained models are available. Run the pipeline to create an experiment.")
        st.code(".venv/bin/python run_pipeline.py")
    detect, batch, research, models = st.tabs(["Analyze text", "Batch analysis", "Research results", "Models & data"])
    with detect:
        st.subheader("Analyze a Bangla passage")
        st.caption(f"Minimum {CFG['dataset']['min_tokens']} words. Longer passages provide more evidence.")
        st.caption("Examples illustrate writing styles; their authorship is not independently verified and their predictions are not accuracy tests.")
        st.session_state.setdefault("input_text", "")
        sample_columns = st.columns(max(1, len(SAMPLES)))
        for column, name in zip(sample_columns, SAMPLES):
            column.button(name, on_click=set_sample, args=(name,), width="stretch")
        text = st.text_area("Bangla text", key="input_text", height=240, max_chars=100000, on_change=clear_analysis)
        if st.button("Analyze text", type="primary", disabled=not ready):
            try:
                with st.spinner("Analyzing the passage…"):
                    st.session_state["analysis_result"] = get_detector(model_id, registry["run_id"]).predict(text)
            except Exception as exc:
                st.error(f"Analysis could not finish: {exc}")
        result = st.session_state.get("analysis_result")
        if result:
            if result.get("model") != model_id:
                st.caption("Showing the previous analysis; analyze again to use the newly selected model.")
            render_result(result)
        if ready:
            selected = st.multiselect("Compare models on this passage", list(ready), default=[model_id], format_func=lambda key: ready[key]["name"])
            if st.button("Compare selected models", disabled=not selected):
                rows = []
                with st.spinner("Running model comparison…"):
                    for key in selected:
                        try:
                            result = get_detector(key, registry["run_id"]).predict(text)
                            rows.append({"Model": ready[key]["name"], "AI score": result.get("ai_probability"), "Decision": result.get("label", result.get("error"))})
                        except Exception as exc:
                            rows.append({"Model": ready[key]["name"], "AI score": None, "Decision": str(exc)})
                st.dataframe(pd.DataFrame(rows), width="stretch")
    with batch:
        st.subheader("Analyze a CSV batch")
        st.write("Upload a UTF-8 CSV with a `text` column. Up to 200 passages per upload; invalid rows receive an error message.")
        upload = st.file_uploader("Batch CSV", type=["csv"], on_change=clear_batch)
        if upload is not None:
            if upload.size > 10*1024*1024:
                st.error("Upload limit is 10 MB.")
            else:
                try:
                    frame = pd.read_csv(upload)
                    if "text" not in frame or frame.empty or len(frame) > 200:
                        st.error("Provide a nonempty text column with at most 200 rows.")
                    else:
                        st.dataframe(frame.head(5), width="stretch")
                        if st.button("Analyze batch", disabled=not ready):
                            with st.spinner("Analyzing batch…"):
                                predictions = get_detector(model_id, registry["run_id"]).predict_many(frame.text.fillna(""))
                            output = frame.reset_index(drop=True).copy()
                            scored = pd.DataFrame(predictions).drop(columns=["style", "cues"], errors="ignore").add_prefix("detector_")
                            for column in scored:
                                output[column] = scored[column]
                            st.session_state["batch_output"] = output
                except Exception as exc:
                    st.error(f"Could not analyze CSV: {exc}")
        if "batch_output" in st.session_state:
            output = st.session_state["batch_output"]
            errors = int(output["detector_error"].notna().sum()) if "detector_error" in output else 0
            st.caption(f"{len(output) - errors} passages scored · {errors} invalid rows. Download includes every input and output column.")
            visible = [column for column in ["id", "detector_label", "detector_ai_probability", "detector_error", "detector_model"] if column in output]
            st.dataframe(output, column_order=visible, hide_index=True, width="stretch", column_config={
                "detector_label": st.column_config.TextColumn("Decision"),
                "detector_ai_probability": st.column_config.NumberColumn("AI score", format="%.3f"),
                "detector_error": st.column_config.TextColumn("Validation error"),
                "detector_model": st.column_config.TextColumn("Model"),
            })
            st.download_button("Download predictions CSV", output.to_csv(index=False).encode("utf-8-sig"), "predictions.csv", "text/csv")
    with research:
        path = Path(CFG["paths"]["reports"])
        experiment_path = Path(registry.get("latest_report", path / "experiment.json"))
        if experiment_path.exists():
            experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
            limitation = experiment["split_manifest"].get("limitation")
            if limitation:
                st.warning(limitation)
            st.caption(f"Run {experiment['run_id']} · seed {experiment['seed']} · {experiment['selection']}")
            comparison = pd.read_csv(experiment_path.parent / "comparison.csv")
            columns = ["model", "kind", "validation_macro_f1", "accuracy", "macro_f1", "roc_auc", "human_false_positive_rate", "brier"]
            st.dataframe(comparison[[c for c in columns if c in comparison]], width="stretch")
            st.write("Test macro-F1 by model · scale 0 to 1")
            st.bar_chart(comparison.set_index("model")[["macro_f1"]], horizontal=True)
            key = st.selectbox("Inspect model evaluation", list(experiment["models"]))
            report = experiment["models"][key]
            if "selective_test" in report:
                selective = report["selective_test"]
                st.caption(f"Forced binary metrics above. With the app's uncertainty policy: coverage {selective['coverage']:.1%}, uncertain passages {selective['n_uncertain']}.")
            a, b = st.columns(2)
            with a:
                st.write("Test confusion matrix · rows actual, columns predicted")
                st.dataframe(pd.DataFrame(report["test"]["confusion_matrix"], index=["Human", "AI"], columns=["Human", "AI"]))
                st.write("95% document-group bootstrap intervals")
                intervals = report["test_95pct_group_bootstrap_ci"]
                st.dataframe(pd.DataFrame(intervals, index=["Lower", "Upper"]).T)
            with b:
                bins = pd.DataFrame(report["test"]["calibration_bins"])
                if not bins.empty:
                    st.write("Calibration · predicted vs observed AI rate")
                    st.line_chart(bins.set_index("mean_probability")[["observed_ai_rate"]])
                st.caption("Intervals describe resampling uncertainty within this corpus; they do not establish external generalization.")
            with st.expander("Generator, source, genre, and subgroup evaluation"):
                for column, values in report["test_subgroups"].items():
                    st.write(column)
                    st.dataframe(pd.DataFrame(values).T[["n", "accuracy", "recall", "human_false_positive_rate"]], width="stretch")
            st.download_button("Download experiment JSON", json.dumps(experiment, ensure_ascii=False, indent=2), "experiment.json", "application/json")
            st.download_button("Download model comparison", comparison.to_csv(index=False), "comparison.csv", "text/csv")
            paired_path = experiment_path.parent / "paired_comparisons.json"
            if paired_path.exists():
                paired = json.loads(paired_path.read_text(encoding="utf-8"))
                with st.expander("Paired model comparisons and significance"):
                    st.caption(f"Reference: {paired['reference']}. {paired['note']}")
                    st.dataframe(pd.DataFrame(paired["comparisons"]).T, width="stretch")
        else:
            st.info("Train models to view measured results and reproducibility records.")
    with models:
        rows = [{"model": key, "name": name, "status": "Trained" if key in ready else "Requires training"} for key, name in CLASSICAL.items()]
        rows += [{"model": key, "name": spec["name"], "status": "Trained" if key in ready else "Requires training", "checkpoint": spec["checkpoint"]} for key, spec in TRANSFORMERS.items()]
        rows.append({"model": "ensemble", "name": "Validation-weighted ensemble", "status": "Trained" if "ensemble" in ready else "Requires training"})
        st.dataframe(pd.DataFrame(rows), column_order=["model", "status", "name", "checkpoint"], hide_index=True, width="stretch")
        st.write("Frozen-encoder experiments train a supervised classifier on contextual embeddings. Fine-tuning updates the encoder itself. Reports record the method used.")
        for name in ["import_manifest", "cleaning_report", "split_manifest", "validation_report"]:
            path = Path(CFG["paths"]["reports"]) / f"{name}.json"
            if path.exists():
                with st.expander(name.replace("_", " ").title()):
                    st.json(json.loads(path.read_text(encoding="utf-8")))
        st.caption("Dataset preparation, model training, and paper methodology are documented in README.md and RESEARCH_GUIDE.md.")


if __name__ == "__main__":
    main()
