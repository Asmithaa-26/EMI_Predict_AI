import streamlit as st
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

st.title("Model Performance & Monitoring")

client = MlflowClient()
experiments = client.search_experiments()

exp_names = [exp.name for exp in experiments]
selected_exp = st.selectbox("Select Experiment", exp_names)

exp = client.get_experiment_by_name(selected_exp)
runs = client.search_runs(exp.experiment_id)

runs_df = pd.DataFrame([
    {
        "run_id": r.info.run_id,
        "status": r.info.status,
        "accuracy": r.data.metrics.get("accuracy"),
        "rmse": r.data.metrics.get("rmse"),
    }
    for r in runs
])

st.dataframe(runs_df)

st.markdown("For full details, access the MLflow UI at http://127.0.0.1:5000")
