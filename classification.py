# ===============================
# 1. IMPORTS
# ===============================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, f1_score

import mlflow
import mlflow.sklearn

# ===============================
# 2. MLFLOW CONFIG (CRITICAL)
# ===============================
mlflow.set_tracking_uri(
    "file:///C:/Users/asmis/OneDrive/Desktop/EMI_Predict V1/EMI_Predict/mlruns"
)
mlflow.set_experiment("EMI_Eligibility_Classification")

# ===============================
# 3. LOAD DATA
# ===============================
df = pd.read_csv("emi_feature_engineered.csv")

X = df.drop(["emi_eligibility", "max_monthly_emi"], axis=1)
y = df["emi_eligibility"]

# Encode labels ONCE
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# ===============================
# 4. TRAIN / TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ===============================
# 5. PREPROCESSOR
# ===============================
num_features = X.select_dtypes(include=["int64", "float64"]).columns
cat_features = X.select_dtypes(include=["object"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ]
)

# ===============================
# 6. LOGISTIC REGRESSION
# ===============================
with mlflow.start_run(run_name="Logistic_Regression"):
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("f1_weighted", f1_score(y_test, y_pred, average="weighted"))

    mlflow.sklearn.log_model(
        model,
        name="logistic_regression_model",
        registered_model_name="EMI_Eligibility_Logistic"
    )

# ===============================
# 7. RANDOM FOREST
# ===============================
with mlflow.start_run(run_name="Random_Forest"):
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("f1_weighted", f1_score(y_test, y_pred, average="weighted"))

    mlflow.sklearn.log_model(
        model,
        name="random_forest_model",
        registered_model_name="EMI_Eligibility_RF"
    )

# ===============================
# 8. XGBOOST
# ===============================
with mlflow.start_run(run_name="XGBoost"):
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("f1_weighted", f1_score(y_test, y_pred, average="weighted"))

    mlflow.sklearn.log_model(
        model,
        name="xgboost_model",
        registered_model_name="EMI_Eligibility_XGB"
    )
