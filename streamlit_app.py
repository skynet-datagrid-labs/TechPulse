from __future__ import annotations

"""Streamlit dashboard for TechPulse risk predictions."""

from pathlib import Path
import logging

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import matplotlib.pyplot as plt

from src.features import FEATURE_COLUMNS, LABEL_COLUMN, TECH_COLUMN

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure logging for the Streamlit app."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


@st.cache_data
def load_feature_matrix() -> pd.DataFrame:
    """Load the feature matrix if it exists."""
    path = DATA_DIR / "feature_matrix.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_model_comparison() -> pd.DataFrame:
    """Load the model comparison summary if it exists."""
    path = REPORTS_DIR / "model_comparison.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_best_model_name() -> str | None:
    """Load the best model name from training outputs."""
    best_path = REPORTS_DIR / "best_model.txt"
    if best_path.exists():
        return best_path.read_text(encoding="utf-8").strip()
    comparison = load_model_comparison()
    if comparison.empty:
        return None
    return str(comparison.sort_values("accuracy", ascending=False).iloc[0]["model"])


@st.cache_data
def load_feature_importance() -> pd.DataFrame:
    """Load the SHAP feature importance ranking."""
    path = REPORTS_DIR / "feature_importance.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_model(model_name: str):
    """Load a trained model pipeline by name."""
    model_path = MODELS_DIR / f"{model_name}.pkl"
    if not model_path.exists():
        return None
    return joblib.load(model_path)


@st.cache_data
def load_label_encoder() -> list[str]:
    """Load label encoder classes if available."""
    encoder_path = MODELS_DIR / "label_encoder.pkl"
    if not encoder_path.exists():
        return []
    encoder = joblib.load(encoder_path)
    if hasattr(encoder, "classes_"):
        return list(encoder.classes_)
    return []


def render_sidebar() -> None:
    """Render the sidebar with project details and navigation."""
    st.sidebar.title("TechPulse ML Dashboard")
    st.sidebar.markdown(
        "Final Year Project: **Predicting Developer Technology Decline Using Community Signals "
        "and Enterprise Adoption Patterns**"
    )
    st.sidebar.markdown("GitHub: https://github.com/skynet-datagrid-labs/TechPulse")


def render_risk_scores(features: pd.DataFrame, model) -> None:
    """Display the technology risk score table with filters."""
    st.header("Technology Risk Scores")

    if features.empty:
        st.warning("Feature matrix not found. Run feature engineering first.")
        return

    if model is None:
        st.warning("Model artifacts missing. Run the training pipeline first.")
        return

    # Run the model to obtain class predictions and confidence scores.
    X = features[FEATURE_COLUMNS]
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    confidence = probabilities.max(axis=1)

    label_classes = load_label_encoder()
    if label_classes:
        predictions = [label_classes[int(idx)] for idx in predictions]

    results = pd.DataFrame(
        {
            "technology": features[TECH_COLUMN],
            "predicted_class": predictions,
            "confidence": confidence,
        }
    )

    if "category" in features.columns:
        results["category"] = features["category"]
    else:
        results["category"] = "Unknown"

    # Interactive filters for category, class, and search.
    category_filter = st.selectbox(
        "Filter by category", sorted(results["category"].unique())
    )
    class_filter = st.multiselect(
        "Filter by class",
        ["growing", "stable", "declining"],
        default=["growing", "stable", "declining"],
    )
    search_term = st.text_input("Search by technology name", "")

    filtered = results[results["category"] == category_filter]
    if class_filter:
        filtered = filtered[filtered["predicted_class"].isin(class_filter)]
    if search_term:
        filtered = filtered[
            filtered["technology"].str.contains(search_term, case=False, na=False)
        ]

    # Color-coded styling for quick visual scanning.
    def color_class(value: str) -> str:
        if value == "growing":
            return "background-color: #c6f6d5"
        if value == "stable":
            return "background-color: #fefcbf"
        if value == "declining":
            return "background-color: #fed7d7"
        return ""

    styled = filtered.style.applymap(color_class, subset=["predicted_class"])
    st.dataframe(styled, use_container_width=True)


def render_model_performance(best_model_name: str | None) -> None:
    """Render model comparison metrics and evaluation plots."""
    st.header("Model Performance")
    comparison = load_model_comparison()
    if comparison.empty:
        st.warning("Model comparison file missing. Run training first.")
        return

    st.dataframe(comparison, use_container_width=True)

    if best_model_name:
        cm_path = FIGURES_DIR / f"confusion_matrix_{best_model_name}.png"
        if cm_path.exists():
            st.subheader("Best Model Confusion Matrix")
            st.image(str(cm_path), use_column_width=True)

    roc_path = FIGURES_DIR / "roc_auc_curves.png"
    if roc_path.exists():
        st.subheader("ROC-AUC Curves")
        st.image(str(roc_path), use_column_width=True)


def render_feature_importance() -> None:
    """Render SHAP summary visuals and top decline drivers."""
    st.header("Feature Importance (SHAP)")

    summary_path = FIGURES_DIR / "shap_summary.png"
    bar_path = FIGURES_DIR / "shap_bar.png"
    if summary_path.exists():
        st.image(str(summary_path), use_column_width=True)
    if bar_path.exists():
        st.image(str(bar_path), use_column_width=True)

    importance = load_feature_importance()
    if importance.empty:
        st.warning("Feature importance not found. Run SHAP explainability first.")
        return

    st.subheader("Top 10 Predictors of Decline")
    st.dataframe(importance.head(10), use_container_width=True)
    st.caption("Higher mean absolute SHAP values indicate stronger influence on predictions.")


def render_prediction_form(features: pd.DataFrame, model) -> None:
    """Render the prediction page with sliders and SHAP explanation."""
    st.header("Predict a Technology")

    if features.empty or model is None:
        st.warning("Feature matrix or model missing. Run the pipeline first.")
        return

    defaults = features[FEATURE_COLUMNS].median().to_dict()
    ranges = {
        column: (
            float(features[column].min()),
            float(features[column].max()),
            float(defaults[column]),
        )
        for column in FEATURE_COLUMNS
    }

    inputs = {}
    for column, (min_val, max_val, default_val) in ranges.items():
        inputs[column] = st.slider(
            column.replace("_", " ").title(),
            min_value=min_val,
            max_value=max_val,
            value=default_val,
        )

    input_df = pd.DataFrame([inputs])
    predicted_class = model.predict(input_df)[0]
    predicted_proba = model.predict_proba(input_df)[0]
    confidence = float(predicted_proba.max())

    label_classes = load_label_encoder()
    if label_classes:
        predicted_class = label_classes[int(predicted_class)]

    st.metric("Predicted Class", predicted_class)
    st.metric("Confidence", f"{confidence:.2f}")

    st.subheader("SHAP Explanation")
    try:
        predictor = model.predict_proba if hasattr(model, "predict_proba") else model.predict
        explainer = shap.Explainer(predictor, features[FEATURE_COLUMNS])
        shap_values = explainer(input_df)
        plt.figure()
        shap.plots.bar(shap_values, max_display=10, show=False)
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close()
    except Exception as exc:  # noqa: BLE001 - surface SHAP errors for user visibility
        st.warning(f"Unable to generate SHAP explanation: {exc}")


def main() -> None:
    """Run the Streamlit application."""
    configure_logging()
    st.set_page_config(page_title="TechPulse Dashboard", layout="wide")

    render_sidebar()
    pages = [
        "Technology Risk Scores",
        "Model Performance",
        "Feature Importance (SHAP)",
        "Predict a Technology",
    ]
    selected_page = st.sidebar.radio("Pages", pages)

    features = load_feature_matrix()
    best_model_name = load_best_model_name()
    model = load_model(best_model_name) if best_model_name else None

    if selected_page == "Technology Risk Scores":
        render_risk_scores(features, model)
    elif selected_page == "Model Performance":
        render_model_performance(best_model_name)
    elif selected_page == "Feature Importance (SHAP)":
        render_feature_importance()
    elif selected_page == "Predict a Technology":
        render_prediction_form(features, model)


if __name__ == "__main__":
    main()
