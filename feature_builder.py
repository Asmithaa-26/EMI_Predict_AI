import pandas as pd

def prepare_input_features(raw_input: dict) -> pd.DataFrame:
    df = pd.DataFrame([raw_input])

    # -------------------------------
    # EXPENSE AGGREGATION
    # -------------------------------
    df["total_monthly_expenses"] = (
        df["school_fees"]
        + df["college_fees"]
        + df["travel_expenses"]
        + df["groceries_utilities"]
        + df["other_monthly_expenses"]
        + df["monthly_rent"]
    )

    # -------------------------------
    # FINANCIAL RATIOS
    # -------------------------------
    df["debt_to_income"] = df["current_emi_amount"] / df["monthly_salary"]
    df["expense_to_income"] = df["total_monthly_expenses"] / df["monthly_salary"]
    df["savings_ratio"] = df["bank_balance"] / df["monthly_salary"]

    # -------------------------------
    # CREDIT RISK BUCKET (CRITICAL FIX)
    # -------------------------------
    df["credit_risk_bucket"] = pd.cut(
        df["credit_score"],
        bins=[0, 600, 700, 900],
        labels=["Low", "Medium", "High"]
    )

    # -------------------------------
    # EMPLOYMENT STABILITY SCORE
    # -------------------------------
    df["employment_stability_score"] = (
        df["years_of_employment"] *
        df["employment_type"].map({
            "Government": 1.5,
            "Private": 1.0,
            "Self-employed": 0.8
        })
    )

    return df