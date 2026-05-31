from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import streamlit as st

ARTIFACTS_DIR = Path("artifacts")


def _load_metrics(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


st.set_page_config(page_title="TechPulse Risk Dashboard", layout="wide")
st.title("TechPulse Technology Risk Dashboard")

prediction_files = sorted(ARTIFACTS_DIR.glob("predictions_*.csv"))
if not prediction_files:
    st.error("No prediction files found. Run the training pipeline first.")
    st.stop()

model_options = {path.stem.replace("predictions_", ""): path for path in prediction_files}
selected_model = st.selectbox("Model", list(model_options.keys()))

predictions = pd.read_csv(model_options[selected_model])
risk_threshold = st.slider("Decline risk threshold", 0.0, 1.0, 0.6, 0.05)

filtered = predictions[predictions["decline_probability"] >= risk_threshold].copy()
filtered = filtered.sort_values("decline_probability", ascending=False)

st.subheader("Highest Risk Technologies")
st.dataframe(filtered.head(50), use_container_width=True)

st.subheader("All Predictions")
st.dataframe(predictions, use_container_width=True)

metrics = _load_metrics(ARTIFACTS_DIR / "metrics.json")
if selected_model in metrics:
    st.subheader("Model Metrics")
    st.json(metrics[selected_model])

importance_path = ARTIFACTS_DIR / "shap" / f"{selected_model}_importance.csv"
if importance_path.exists():
    st.subheader("Top Feature Importance (SHAP)")
    importance = pd.read_csv(importance_path).head(20)
    st.bar_chart(importance.set_index("feature"))
