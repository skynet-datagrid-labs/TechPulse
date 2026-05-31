from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from ml_pipeline.io import ensure_dir

matplotlib.use("Agg")


def _mean_abs_shap(values: np.ndarray, feature_count: int) -> np.ndarray:
    if values.ndim == 3:
        if values.shape[1] == feature_count:
            values = np.mean(np.abs(values), axis=2)
        elif values.shape[2] == feature_count:
            values = np.mean(np.abs(values), axis=1)
        else:
            raise ValueError("Unexpected SHAP value shape for feature importance.")
    return np.mean(np.abs(values), axis=0)


def generate_shap_outputs(
    model_name: str,
    pipeline,
    X: pd.DataFrame,
    artifacts_dir: Path,
) -> Path:
    shap_dir = ensure_dir(artifacts_dir / "shap")
    preprocessor = pipeline[:-1]
    model = pipeline[-1]

    X_processed = preprocessor.transform(X)
    feature_names = list(X.columns)

    if model_name in {"random_forest", "xgboost"}:
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.LinearExplainer(model, X_processed)

    shap_values = explainer(X_processed)
    values = np.asarray(shap_values.values)

    if values.ndim == 2 and values.shape[1] != len(feature_names):
        raw_values = explainer.shap_values(X_processed)
        if isinstance(raw_values, list):
            values = np.stack(raw_values, axis=1)
        else:
            values = np.asarray(raw_values)

    importance = _mean_abs_shap(values, len(feature_names))

    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importance}
    ).sort_values("importance", ascending=False)

    importance_path = shap_dir / f"{model_name}_importance.csv"
    importance_df.to_csv(importance_path, index=False)

    plt.figure(figsize=(8, 6))
    top = importance_df.head(15).iloc[::-1]
    plt.barh(top["feature"], top["importance"])
    plt.title(f"{model_name} SHAP importance")
    plt.tight_layout()
    plt.savefig(shap_dir / f"{model_name}_importance.png", dpi=150)
    plt.close()

    return importance_path
