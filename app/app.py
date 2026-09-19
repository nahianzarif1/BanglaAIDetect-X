"""
BanglaAIDetect-X — Streamlit demo

Run:  streamlit run app/app.py
"""

import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import streamlit as st

from src.inference import BanglaAIDetector

try:
    from app.samples import DEMOS
except ImportError:
    from samples import DEMOS

ROOT = os.path.join(os.path.dirname(__file__), "..")
METRICS_PATH = os.path.join(ROOT, "results", "reports", "baseline_metrics.json")

st.set_page_config(
    page_title="BanglaAIDetect-X",
    page_icon="ন",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=Noto+Sans+Bengali:wght@400;500;600;700&display=swap');

html, body, [class*="stApp"] {
  font-family: "Noto Sans Bengali", "IBM Plex Sans", sans-serif;
  background-color: white;
  color: black;
}
.stApp {
  background-color: white;
  color: black;
}
.block-container { padding-top: 1.4rem; max-width: 1180px; background-color: white; color: black; }

h1, h2, h3 { letter-spacing: -0.02em; color: black; }
.stMetric { background: #f0f0f0; border-radius: 8px; padding: 1rem; color: black; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_detector():
    os.chdir(ROOT)
    return BanglaAIDetector()


def badge(label: str) -> str:
    if label == "ai":
        return '<span style="background:#ef4444;color:white;padding:4px 12px;border-radius:999px;font-size:0.85rem;font-weight:600;">AI-generated</span>'
    if label == "human":
        return '<span style="background:#10b981;color:white;padding:4px 12px;border-radius:999px;font-size:0.85rem;font-weight:600;">Human-written</span>'
    return '<span style="background:#f59e0b;color:black;padding:4px 12px;border-radius:999px;font-size:0.85rem;font-weight:600;">Uncertain</span>'


def main():
    st.title("BanglaAIDetect-X")
    st.markdown("Bangla text classification tool for human vs AI detection.")

    # Initialize session state for text input
    if "current_text" not in st.session_state:
        st.session_state.current_text = ""

    st.markdown("### Analyze Bangla text")
    text = st.text_area(
        "Paste Bangla text (≈40+ words) for analysis.",
        height=240,
        value=st.session_state.current_text,
        key="main_text_area",
    )
    run = st.button("Analyze Bangla text", type="primary")

    if run:
        try:
            detector = get_detector()
            with st.spinner("Running detection..."):
                result = detector.predict(text)
        except FileNotFoundError as e:
            st.error(f"Model not found: {e}. Run `python run_pipeline.py` first.")
            return
        except Exception as e:
            st.error(f"Analysis error: {e}")
            return

        st.divider()
        st.markdown("### Result")

        # Check for error in result
        if "error" in result:
            st.error(f"Analysis error: {result['error']}")
            return

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.metric("AI probability", f"{result['ai_probability'] * 100:.1f}%")
        with col2:
            st.metric("Human probability", f"{result['human_probability'] * 100:.1f}%")
        with col3:
            # Display label without icon
            label_text = result["label"].upper()
            st.metric("Label", label_text)

        st.markdown(f"""
        <div style="background:#f0f0f0;border-radius:8px;padding:1rem;margin:1rem 0;color:black;">
        <strong>Detected script:</strong> {result.get('detected_script', 'unknown')}<br>
        <strong>Bangla ratio:</strong> {result.get('bangla_ratio', 0) * 100:.0f}%<br>
        <strong>Token count:</strong> {result.get('token_count', 0)}<br>
        <strong>Confidence:</strong> {result.get('confidence', 'unknown')}
        </div>
        """, unsafe_allow_html=True)

        if result.get("style"):
            st.markdown("### Style features")
            style = result["style"]
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Burstiness", f"{style.get('burstiness', 0):.3f}")
                st.metric("AI markers", f"{style.get('ai_marker_density', 0):.3f}")
            with col2:
                st.metric("Human markers", f"{style.get('human_marker_density', 0):.3f}")
                st.metric("Digit ratio", f"{style.get('digit_ratio', 0):.3f}")

        if result.get("cues"):
            st.markdown("### Detection cues")
            for cue in result["cues"]:
                side_text = cue.get("side", "neutral").upper()
                st.info(f"{side_text} - {cue.get('title', 'Unknown')}: {cue.get('detail', '')}")

        st.warning(result.get("note", "This is a decision-support estimate for Bangla text only — not an academic-integrity verdict."))


if __name__ == "__main__":
    main()