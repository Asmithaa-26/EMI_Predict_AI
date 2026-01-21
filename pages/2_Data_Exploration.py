import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Financial Data Exploration & Insights")

# -----------------------------------
# DATA LOADING
# -----------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/loan_applications.csv")

df = load_data()

st.subheader("Dataset Overview")
st.write(f"Total Records: {len(df)}")
st.dataframe(df.head(50), use_container_width=True)

# -----------------------------------
# SCHEMA VALIDATION
# -----------------------------------
required_columns = {
    "monthly_salary",
    "max_monthly_emi",
    "credit_score",
    "emi_eligibility",
    "debt_to_income",
    "expense_to_income",
    "savings_ratio",
    "credit_risk_bucket"
}

missing_cols = required_columns - set(df.columns)
if missing_cols:
    st.error(f"Missing required columns: {missing_cols}")
    st.stop()

# -----------------------------------
# INTERACTIVE FILTERS
# -----------------------------------
st.markdown("---")
st.subheader("Data Filters")

col1, col2, col3 = st.columns(3)

with col1:
    credit_range = st.slider(
        "Credit Score Range",
        int(df.credit_score.min()),
        int(df.credit_score.max()),
        (600, 800)
    )

with col2:
    eligibility_filter = st.multiselect(
        "EMI Eligibility",
        options=df["emi_eligibility"].unique().tolist(),
        default=df["emi_eligibility"].unique().tolist()
    )

with col3:
    risk_filter = st.multiselect(
        "Credit Risk Bucket",
        options=df["credit_risk_bucket"].unique().tolist(),
        default=df["credit_risk_bucket"].unique().tolist()
    )

filtered_df = df[
    (df["credit_score"].between(*credit_range)) &
    (df["emi_eligibility"].isin(eligibility_filter)) &
    (df["credit_risk_bucket"].isin(risk_filter))
]

st.write(f"Filtered Records: {len(filtered_df)}")

# -----------------------------------
# VISUALIZATIONS
# -----------------------------------
st.markdown("---")
st.subheader("Key Financial Visualizations")

# ---- Salary vs Max EMI
st.markdown("### Monthly Salary vs Maximum Safe EMI")

fig, ax = plt.subplots()
ax.scatter(
    filtered_df["monthly_salary"],
    filtered_df["max_monthly_emi"]
)
ax.set_xlabel("Monthly Salary (INR)")
ax.set_ylabel("Maximum Safe EMI (INR)")
st.pyplot(fig)

# ---- Credit Score vs EMI
st.markdown("### Credit Score vs EMI Capacity")

fig, ax = plt.subplots()
ax.scatter(
    filtered_df["credit_score"],
    filtered_df["max_monthly_emi"]
)
ax.set_xlabel("Credit Score")
ax.set_ylabel("Maximum Safe EMI (INR)")
st.pyplot(fig)

# ---- Debt to Income Distribution
st.markdown("### Debt-to-Income Ratio Distribution")

fig, ax = plt.subplots()
ax.hist(filtered_df["debt_to_income"], bins=30)
ax.set_xlabel("Debt-to-Income Ratio")
st.pyplot(fig)

# ---- EMI Eligibility Breakdown
st.markdown("### EMI Eligibility Distribution")
eligibility_counts = filtered_df["emi_eligibility"].value_counts()
st.bar_chart(eligibility_counts)

# ---- Credit Risk Bucket Distribution
st.markdown("### Credit Risk Bucket Distribution")
risk_counts = filtered_df["credit_risk_bucket"].value_counts()
st.bar_chart(risk_counts)

# -----------------------------------
# SUMMARY STATISTICS
# -----------------------------------
st.markdown("---")
st.subheader("Summary Statistics")

col4, col5, col6 = st.columns(3)

with col4:
    st.metric(
        "Average Salary",
        f"₹ {filtered_df['monthly_salary'].mean():,.0f}"
    )

with col5:
    st.metric(
        "Average Max EMI",
        f"₹ {filtered_df['max_monthly_emi'].mean():,.0f}"
    )

with col6:
    st.metric(
        "Average Credit Score",
        f"{filtered_df['credit_score'].mean():.0f}"
    )

# -----------------------------------
# EXPORT OPTION
# -----------------------------------
st.markdown("---")
st.subheader("Export Filtered Data")

csv = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download Filtered Dataset",
    data=csv,
    file_name="filtered_loan_data.csv",
    mime="text/csv"
)
