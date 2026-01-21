import streamlit as st

st.set_page_config(
    page_title="EMIPredict AI",
    layout="wide"
)

st.title("EMIPredict AI")
st.subheader("Intelligent Financial Risk Assessment Platform")

st.markdown("""
Welcome to **EMIPredict AI**, an end-to-end financial risk assessment system.

### Capabilities
- EMI eligibility classification
- Safe EMI regression prediction
- Financial data exploration
- ML model monitoring via MLflow
- Administrative data controls

Use the sidebar to navigate between modules.
""")
