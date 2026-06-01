from __future__ import annotations

"""Tests for SHAP explainability utilities."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

import src.explain as explain
from src.features import FEATURE_COLUMNS, LABEL_COLUMN, TECH_COLUMN


def test_shap_values_sum_to_expected() -> None:
    """Confirm SHAP additivity for a linear regression model."""
    X = pd.DataFrame({"feature_a": [0, 1, 2, 3], "feature_b": [1, 2, 3, 4]})
    y = np.array([1.0, 2.0, 3.0, 4.0])
    model = LinearRegression().fit(X, y)

    shap_values = explain.compute_shap_values(model, X)
    reconstructed = shap_values.values.sum(axis=1) + shap_values.base_values
    assert np.allclose(reconstructed, model.predict(X), atol=1e-6)


def test_feature_importance_csv_exists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the feature importance CSV is generated during SHAP processing."""
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    figures_dir = reports_dir / "figures"

    data_dir.mkdir(parents=True)
    models_dir.mkdir(parents=True)
    figures_dir.mkdir(parents=True)

    feature_matrix = pd.DataFrame(
        {
            TECH_COLUMN: ["alpha", "beta", "gamma", "delta", "epsilon"],
            LABEL_COLUMN: ["growing", "stable", "declining", "growing", "declining"],
        }
    )
    for column in FEATURE_COLUMNS:
        feature_matrix[column] = np.linspace(0.1, 1.0, len(feature_matrix))

    feature_matrix.to_csv(data_dir / "feature_matrix.csv", index=False)

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=200, random_state=42)),
        ]
    )
    model.fit(feature_matrix[FEATURE_COLUMNS], feature_matrix[LABEL_COLUMN])
    joblib.dump(model, models_dir / "test_model.pkl")

    (reports_dir / "best_model.txt").write_text("test_model", encoding="utf-8")

    monkeypatch.setattr(explain, "DATA_DIR", data_dir)
    monkeypatch.setattr(explain, "MODELS_DIR", models_dir)
    monkeypatch.setattr(explain, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(explain, "FIGURES_DIR", figures_dir)

    explain.generate_shap_outputs()
    assert (reports_dir / "feature_importance.csv").exists()
