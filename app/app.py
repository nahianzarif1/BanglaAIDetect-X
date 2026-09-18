"""
app/app.py

Run with: streamlit run app/app.py
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from src.inference import BanglaAIDetector

st.set_page_config(page_title="BanglaAIDetect-X", page_icon="🔎")
st.title("🔎 BanglaAIDetect-X")
st.caption("Decision-support tool for estimating whether Bangla text is human-written or AI-generated. "
           "Not an infallible determination.")

@st.cache_resource
def get_detector():
    return BanglaAIDetector()

text = st.text_area("Paste Bangla / Banglish / mixed text (at least ~50 tokens):", height=220)

if st.button("Analyze"):
    try:
        detector = get_detector()
    except FileNotFoundError as e:
        st.error(str(e))
    else:
        result = detector.predict(text)
        if "error" in result:
            st.warning(result["error"])
        else:
            col1, col2 = st.columns(2)
            col1.metric("AI probability", f"{result['ai_probability']*100:.1f}%")
            col2.metric("Human probability", f"{result['human_probability']*100:.1f}%")
            st.write(f"**Predicted label:** {result['label'].upper()}")
            st.write(f"**Detected script:** {result['detected_script']}")
            st.caption(result["note"])
