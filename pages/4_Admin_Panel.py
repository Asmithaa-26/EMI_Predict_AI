import streamlit as st
import pandas as pd

st.title("Administrative Panel")
st.warning("Restricted Access Authorized Users Only")

uploaded_file = st.file_uploader(
    "Upload New Loan Dataset (CSV)",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset uploaded successfully")
    st.dataframe(df.head())

    if st.button("Persist Dataset"):
        df.to_csv("data/loan_applications.csv", index=False)
        st.success("Dataset saved and ready for retraining")
