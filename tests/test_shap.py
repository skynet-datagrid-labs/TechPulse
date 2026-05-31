from __future__ import annotations

from ml_pipeline.explain import generate_shap_outputs
from ml_pipeline.train import _build_pipelines, prepare_dataset


def test_shap_outputs_created(config):
    X, y, _ = prepare_dataset(config)
    pipelines = _build_pipelines(config.random_state, model_params={"random_forest": {"n_estimators": 5}})

    pipeline = pipelines["random_forest"]
    pipeline.fit(X, y)

    output = generate_shap_outputs("random_forest", pipeline, X, config.artifacts_dir)
    assert output.exists()
