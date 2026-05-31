from __future__ import annotations

from ml_pipeline.train import train_all


def test_train_all_creates_artifacts(config):
    metrics = train_all(
        config,
        with_shap=False,
        model_params={
            "random_forest": {"n_estimators": 10},
            "xgboost": {"n_estimators": 10, "max_depth": 3},
            "logistic_regression": {"max_iter": 200},
        },
    )

    assert "random_forest" in metrics
    assert (config.artifacts_dir / "predictions_random_forest.csv").exists()
    assert (config.artifacts_dir / "metrics.json").exists()
