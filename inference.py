import mlflow.pyfunc
import pandas as pd
from feature_builder import prepare_input_features

ELIGIBILITY_MAP = {
    0: "High Risk",
    1: "Not Eligible",
    2: "Eligible"
}

CLASSIFIER_URI = "models:/EMI_Eligibility_XGB/Production"
REGRESSOR_URI = "models:/Max_EMI_XGB/Production"

classifier = mlflow.pyfunc.load_model(CLASSIFIER_URI)
regressor = mlflow.pyfunc.load_model(REGRESSOR_URI)


def predict_emi(raw_input: dict):
    input_df = prepare_input_features(raw_input)

    # Hard eligibility rules
    if (
        input_df["credit_score"].iloc[0] < 500
        or input_df["debt_to_income"].iloc[0] > 0.65
        or input_df["expense_to_income"].iloc[0] > 0.75
    ):
        return "Not Eligible", 0.0  # or None if preferred

    # ML predictions
    eligibility_code = int(classifier.predict(input_df)[0])
    max_emi = float(regressor.predict(input_df)[0])

    eligibility_label = ELIGIBILITY_MAP.get(
        eligibility_code, "Unknown"
    )

    return eligibility_label, max_emi
