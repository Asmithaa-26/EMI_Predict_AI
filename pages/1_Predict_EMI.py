import streamlit as st
import mlflow
from inference import predict_emi

mlflow.set_tracking_uri("http://127.0.0.1:5000")

st.set_page_config(layout="wide")
st.title("EMI Eligibility & Risk Prediction")

# -------------------------------
# PERSONAL & DEMOGRAPHIC DETAILS
# -------------------------------
st.header("Personal Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 18, 70, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married"])

with col2:
    education = st.selectbox(
        "Education",
        ["High School", "Graduate", "Post Graduate", "Professional"]
    )
    family_size = st.number_input("Family Size", 1, 10, 3)
    dependents = st.number_input("Dependents", 0, 10, 1)

with col3:
    house_type = st.selectbox("House Type", ["Rented", "Own", "Family"])
    company_type = st.selectbox(
        "Company Type", ["Startup", "SME", "MNC", "Government"]
    )

# -------------------------------
# EMPLOYMENT & INCOME
# -------------------------------
st.header("Employment & Income")

col4, col5, col6 = st.columns(3)

with col4:
    employment_type = st.selectbox(
        "Employment Type",
        ["Private", "Government", "Self-employed"]
    )
    years_of_employment = st.number_input("Years of Employment", 0, 40, 5)

with col5:
    monthly_salary = st.number_input("Monthly Salary (INR)", 10000, 300000, 50000)
    monthly_rent = st.number_input("Monthly Rent (INR)", 0, 100000, 10000)

with col6:
    credit_score = st.number_input("Credit Score", 300, 900, 700)
    existing_loans = st.selectbox("Existing Loans", ["Yes", "No"])

# -------------------------------
# EXPENSES
# -------------------------------
st.header("Monthly Expenses")

col7, col8, col9 = st.columns(3)

with col7:
    school_fees = st.number_input("School Fees", 0, 50000, 0)
    college_fees = st.number_input("College Fees", 0, 100000, 0)

with col8:
    travel_expenses = st.number_input("Travel Expenses", 0, 50000, 3000)
    groceries_utilities = st.number_input("Groceries & Utilities", 0, 50000, 8000)

with col9:
    other_monthly_expenses = st.number_input("Other Monthly Expenses", 0, 50000, 5000)
    current_emi_amount = st.number_input("Current EMI Amount", 0, 100000, 0)

# -------------------------------
# FINANCIAL STATUS
# -------------------------------
st.header("Financial Status")

col10, col11 = st.columns(2)

with col10:
    bank_balance = st.number_input("Bank Balance (INR)", 0, 10000000, 200000)

with col11:
    emergency_fund = st.number_input("Emergency Fund (INR)", 0, 5000000, 100000)

# -------------------------------
# LOAN DETAILS
# -------------------------------
st.header("Loan Application Details")

col12, col13, col14 = st.columns(3)

with col12:
    emi_scenario = st.selectbox(
        "EMI Scenario",
        [
            "E-commerce Shopping EMI",
            "Home Appliances EMI",
            "Vehicle EMI",
            "Personal Loan EMI",
            "Education EMI"
        ]
    )

with col13:
    requested_amount = st.number_input("Requested Loan Amount (INR)", 10000, 20000000, 300000)

with col14:
    requested_tenure = st.number_input("Requested Tenure (Months)", 3, 84, 24)

# -------------------------------
# PREDICTION
# -------------------------------
st.markdown("---")

if st.button("Run Prediction"):

    raw_input = {
        "age": age,
        "gender": gender,
        "marital_status": marital_status,
        "education": education,
        "monthly_salary": monthly_salary,
        "employment_type": employment_type,
        "years_of_employment": years_of_employment,
        "company_type": company_type,
        "house_type": house_type,
        "monthly_rent": monthly_rent,
        "family_size": family_size,
        "dependents": dependents,
        "school_fees": school_fees,
        "college_fees": college_fees,
        "travel_expenses": travel_expenses,
        "groceries_utilities": groceries_utilities,
        "other_monthly_expenses": other_monthly_expenses,
        "existing_loans": existing_loans,
        "current_emi_amount": current_emi_amount,
        "credit_score": credit_score,
        "bank_balance": bank_balance,
        "emergency_fund": emergency_fund,
        "emi_scenario": emi_scenario,
        "requested_amount": requested_amount,
        "requested_tenure": requested_tenure
    }

    eligibility, max_emi = predict_emi(raw_input)

    st.success(f"Eligibility Status: {eligibility}")
    st.info(f"Maximum Safe EMI: ₹ {max_emi:,.2f}")
