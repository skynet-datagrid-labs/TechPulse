from __future__ import annotations

"""Tests for model training utilities."""

import joblib
from pathlib import Path

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.train import build_pipelines, load_model


def test_all_models_train_without_errors(synthetic_dataset) -> None:
    """Confirm all specified models can train on the sample dataset."""
    X, y = synthetic_dataset
    pipelines = build_pipelines(random_state=42)
    for pipeline in pipelines.values():
        pipeline.fit(X, y)


def test_prediction_output_shape(synthetic_dataset) -> None:
    """Ensure prediction outputs align with the test split size."""
    X, y = synthetic_dataset
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.33, stratify=y, random_state=42
    )
    pipeline = build_pipelines(random_state=42)["random_forest"]
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    assert predictions.shape[0] == X_test.shape[0]


def test_accuracy_above_threshold(synthetic_dataset) -> None:
    """Validate that a model exceeds the minimum accuracy requirement."""
    X, y = synthetic_dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, stratify=y, random_state=42
    )
    pipeline = build_pipelines(random_state=42)["logistic_regression"]
    pipeline.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, pipeline.predict(X_test))
    assert accuracy > 0.7


def test_model_loading(synthetic_dataset, tmp_path: Path) -> None:
    """Ensure saved models can be loaded from disk."""
    X, y = synthetic_dataset
    pipeline = build_pipelines(random_state=42)["knn"]
    pipeline.fit(X, y)
    model_path = tmp_path / "knn.pkl"
    joblib.dump(pipeline, model_path)
    loaded = load_model(model_path)
    assert loaded is not None
