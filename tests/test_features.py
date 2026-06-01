from __future__ import annotations

"""Tests for feature engineering and labeling."""

import numpy as np
import pandas as pd

from src.features import (
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    TECH_COLUMN,
    TREND_COLUMN,
    derive_labels,
    impute_and_scale,
)
from src.train import build_pipelines
from sklearn.model_selection import train_test_split


def test_label_derivation_thresholds() -> None:
    """Verify label rules for growing, stable, and declining classes."""
    sample = pd.DataFrame(
        {
            "growth_momentum_index": [80, 55, 20],
            TREND_COLUMN: ["positive", "flat", "negative"],
        }
    )
    labels = derive_labels(sample)
    assert labels.tolist() == ["growing", "stable", "declining"]


def test_missing_value_handling() -> None:
    """Ensure missing values are imputed with medians and no NaNs remain."""
    data = pd.DataFrame(
        {
            "technology_health_score": [1.0, np.nan, 3.0],
            "growth_momentum_index": [1.0, 2.0, np.nan],
            "question_quality_score": [np.nan, np.nan, np.nan],
            "company_diversity_score": [1.0, 2.0, 3.0],
            "sentiment_delta": [1.0, 2.0, 3.0],
            "adoption_velocity": [1.0, 2.0, 3.0],
            "community_decay_rate": [1.0, 2.0, 3.0],
        }
    )
    cleaned = impute_and_scale(data)
    assert not cleaned[FEATURE_COLUMNS].isna().any().any()


def test_output_shape(feature_matrix: pd.DataFrame) -> None:
    """Confirm the feature matrix contains required columns and labels."""
    for column in FEATURE_COLUMNS + [LABEL_COLUMN, TECH_COLUMN]:
        assert column in feature_matrix.columns
    assert feature_matrix[LABEL_COLUMN].notna().all()


def test_no_data_leakage_between_train_test(synthetic_dataset) -> None:
    """Ensure the scaler is fit only on the training split."""
    X, y = synthetic_dataset
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.33, stratify=y, random_state=42
    )
    pipeline = build_pipelines(random_state=42)["logistic_regression"]
    pipeline.fit(X_train, y_train)
    scaler_mean = pipeline.named_steps["scaler"].mean_
    assert np.allclose(scaler_mean, X_train.mean().values)
