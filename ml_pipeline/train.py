from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

from ml_pipeline.config import LABEL_COLUMN, PipelineConfig, TAG_COLUMN
from ml_pipeline.explain import generate_shap_outputs
from ml_pipeline.features import build_labeled_feature_matrix, derive_labels
from ml_pipeline.io import ensure_dir


def prepare_dataset(config: PipelineConfig) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    features = build_labeled_feature_matrix(
        config.results_dir,
        config.label_growth_thresholds[0],
        config.label_growth_thresholds[1],
    )
    labels = features[LABEL_COLUMN]

    numeric_features = features.select_dtypes(include=[np.number]).copy()
    numeric_features = numeric_features.dropna(axis=1, how="all")
    tag = features[TAG_COLUMN].astype("string")

    return numeric_features, labels.astype("string"), tag


def _build_pipelines(
    random_state: int,
    model_params: dict[str, dict[str, object]] | None = None,
) -> dict[str, Pipeline]:
    model_params = model_params or {}
    tree_preprocess = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    linear_preprocess = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    rf_params = {
        "n_estimators": 300,
        "random_state": random_state,
        "class_weight": "balanced",
    }
    rf_params.update(model_params.get("random_forest", {}))

    xgb_params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "random_state": random_state,
    }
    xgb_params.update(model_params.get("xgboost", {}))

    lr_params = {
        "max_iter": 2000,
        "multi_class": "multinomial",
        "class_weight": "balanced",
        "random_state": random_state,
    }
    lr_params.update(model_params.get("logistic_regression", {}))

    return {
        "random_forest": Pipeline(
            [
                ("prep", tree_preprocess),
                ("model", RandomForestClassifier(**rf_params)),
            ]
        ),
        "xgboost": Pipeline(
            [
                ("prep", tree_preprocess),
                ("model", XGBClassifier(**xgb_params)),
            ]
        ),
        "logistic_regression": Pipeline(
            [
                ("prep", linear_preprocess),
                ("model", LogisticRegression(**lr_params)),
            ]
        ),
    }


def _evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    preds = model.predict(X)
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "macro_f1": float(f1_score(y, preds, average="macro")),
    }


def train_all(
    config: PipelineConfig,
    with_shap: bool = True,
    model_params: dict[str, dict[str, object]] | None = None,
) -> dict[str, dict[str, float]]:
    X, y, tags = prepare_dataset(config)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    n_samples = len(y_encoded)
    n_classes = len(label_encoder.classes_)
    min_test_fraction = n_classes / n_samples
    test_size = max(config.test_size, min_test_fraction)

    X_train, X_test, y_train, y_test, tags_train, tags_test = train_test_split(
        X, y_encoded, tags, test_size=test_size, random_state=config.random_state, stratify=y_encoded
    )

    artifacts_dir = ensure_dir(config.artifacts_dir)
    models_dir = ensure_dir(artifacts_dir / "models")

    metrics: dict[str, dict[str, float]] = {}
    pipelines = _build_pipelines(config.random_state, model_params=model_params)

    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        metrics[name] = _evaluate(pipeline, X_test, y_test)

        joblib.dump(pipeline, models_dir / f"{name}.joblib")

        probs = pipeline.predict_proba(X)
        class_order = label_encoder.classes_
        decline_index = int(np.where(class_order == "declining")[0][0])

        predictions = pd.DataFrame(
            {
                TAG_COLUMN: tags,
                "predicted_label": label_encoder.inverse_transform(pipeline.predict(X)),
                "decline_probability": probs[:, decline_index],
            }
        )
        for idx, label in enumerate(class_order):
            predictions[f"prob_{label}"] = probs[:, idx]

        predictions.to_csv(artifacts_dir / f"predictions_{name}.csv", index=False)

        if with_shap:
            generate_shap_outputs(name, pipeline, X, artifacts_dir)

    with open(artifacts_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TechPulse ML models.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--no-shap", action="store_true")
    args = parser.parse_args()

    config = PipelineConfig(
        results_dir=Path(args.results_dir),
        artifacts_dir=Path(args.artifacts_dir),
    )
    train_all(config, with_shap=not args.no_shap)


if __name__ == "__main__":
    main()
