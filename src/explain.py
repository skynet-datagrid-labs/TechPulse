from __future__ import annotations

"""SHAP explainability utilities for TechPulse models."""

from pathlib import Path
import logging

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from src.features import FEATURE_COLUMNS, LABEL_COLUMN, TECH_COLUMN

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure logging once for explainability."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_feature_matrix() -> pd.DataFrame:
    """Load the feature matrix for SHAP analysis."""
    path = DATA_DIR / "feature_matrix.csv"
    if not path.exists():
        raise FileNotFoundError("Feature matrix not found. Run src/features.py first.")
    return pd.read_csv(path)


def load_best_model_name() -> str:
    """Resolve the best model name from training outputs."""
    best_path = REPORTS_DIR / "best_model.txt"
    if best_path.exists():
        return best_path.read_text(encoding="utf-8").strip()

    comparison_path = REPORTS_DIR / "model_comparison.csv"
    if not comparison_path.exists():
        raise FileNotFoundError("Model comparison file missing. Run src/train.py first.")

    comparison = pd.read_csv(comparison_path)
    return str(comparison.sort_values("accuracy", ascending=False).iloc[0]["model"])


def load_model(model_name: str):
    """Load a trained model pipeline by name."""
    model_path = MODELS_DIR / f"{model_name}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(model_path)


def load_label_encoder() -> list[str]:
    """Load the label encoder classes if available."""
    if LABEL_ENCODER_PATH.exists():
        encoder = joblib.load(LABEL_ENCODER_PATH)
        if hasattr(encoder, "classes_"):
            return list(encoder.classes_)
    return []


def compute_shap_values(model, X: pd.DataFrame) -> shap.Explanation:
    """Compute SHAP values for a model and dataset."""
    predictor = model.predict_proba if hasattr(model, "predict_proba") else model.predict
    try:
        explainer = shap.Explainer(predictor, X, algorithm="kernel")
        return explainer(X, max_evals=100)
    except Exception as exc:  # noqa: BLE001 - fallback when SHAP backend is unavailable
        LOGGER.warning("SHAP backend failed (%s). Falling back to linear approximation.", exc)

    # Linear fallback for environments where SHAP backends are unavailable.
    estimator = model
    scaler = None
    if hasattr(model, "named_steps"):
        scaler = model.named_steps.get("scaler")
        estimator = model.named_steps.get("model", model)

    if not hasattr(estimator, "coef_"):
        raise RuntimeError("SHAP fallback requires a linear model with coef_.")

    X_values = X.values
    if scaler is not None:
        X_values = scaler.transform(X_values)

    mean_values = X_values.mean(axis=0)
    coef = np.asarray(estimator.coef_)
    intercept = np.asarray(getattr(estimator, "intercept_", 0.0))

    if coef.ndim == 1:
        values = (X_values - mean_values) * coef
        base_values = np.full(X_values.shape[0], intercept + mean_values @ coef)
    else:
        values = np.stack(
            [(X_values - mean_values) * coef_row for coef_row in coef],
            axis=2,
        )
        base_values = np.tile(intercept, (X_values.shape[0], 1))

    return shap.Explanation(
        values=values,
        base_values=base_values,
        data=X_values,
        feature_names=list(X.columns),
    )


def summarize_shap_values(shap_values: shap.Explanation) -> tuple[np.ndarray, list[str]]:
    """Summarize SHAP values into mean absolute importance per feature."""
    values = shap_values.values
    if values.ndim == 3:
        mean_abs = np.mean(np.abs(values), axis=(0, 2))
    else:
        mean_abs = np.mean(np.abs(values), axis=0)
    return mean_abs, list(shap_values.feature_names)


def extract_class_importance(shap_values: shap.Explanation) -> dict[int, np.ndarray]:
    """Compute per-class mean absolute SHAP values for multi-class outputs."""
    values = shap_values.values
    if values.ndim != 3:
        return {0: np.mean(np.abs(values), axis=0)}
    return {idx: np.mean(np.abs(values[:, :, idx]), axis=0) for idx in range(values.shape[2])}


def generate_shap_outputs() -> None:
    """Generate SHAP plots, feature importance files, and console summaries."""
    configure_logging()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load trained model and feature data for explainability analysis.
    features = load_feature_matrix()
    X = features[FEATURE_COLUMNS]
    model_name = load_best_model_name()
    model = load_model(model_name)

    LOGGER.info("Generating SHAP outputs for %s", model_name)
    shap_values = compute_shap_values(model, X)

    # Summary plot shows the top 10 drivers across the dataset.
    plt.figure()
    shap.summary_plot(shap_values, X, max_display=10, show=False)
    plt.tight_layout()
    summary_path = FIGURES_DIR / "shap_summary.png"
    plt.savefig(summary_path, dpi=150)
    plt.close()

    # Bar plot highlights mean absolute SHAP values.
    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", max_display=10, show=False)
    plt.tight_layout()
    bar_path = FIGURES_DIR / "shap_bar.png"
    plt.savefig(bar_path, dpi=150)
    plt.close()

    mean_abs, feature_names = summarize_shap_values(shap_values)
    importance = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    importance = importance.sort_values("mean_abs_shap", ascending=False)
    importance_path = REPORTS_DIR / "feature_importance.csv"
    importance.to_csv(importance_path, index=False)

    # Print top predictors for each class and for decline in particular.
    class_importance = extract_class_importance(shap_values)
    class_labels = load_label_encoder()
    if not class_labels and hasattr(model, "named_steps"):
        classifier = model.named_steps.get("model")
        if hasattr(classifier, "classes_"):
            class_labels = list(classifier.classes_)

    for idx, values in class_importance.items():
        label = class_labels[idx] if idx < len(class_labels) else f"class_{idx}"
        top_features = (
            pd.DataFrame({"feature": feature_names, "importance": values})
            .sort_values("importance", ascending=False)
            .head(3)
        )
        LOGGER.info("Top predictors for %s:\n%s", label, top_features.to_string(index=False))

    decline_index = class_labels.index("declining") if "declining" in class_labels else 0
    decline_importance = class_importance[decline_index]
    decline_top = (
        pd.DataFrame({"feature": feature_names, "importance": decline_importance})
        .sort_values("importance", ascending=False)
        .head(10)
    )
    LOGGER.info(
        "Top 10 predictors of technology decline:\n%s",
        decline_top.to_string(index=False),
    )


def main() -> None:
    """Run SHAP explainability from the command line."""
    generate_shap_outputs()


if __name__ == "__main__":
    main()
