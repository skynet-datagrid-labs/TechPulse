from __future__ import annotations

from ml_pipeline.config import QUERY7_GROWTH_COL, QUERY9_RISK_COL, TAG_COLUMN
from ml_pipeline.features import build_feature_matrix, derive_labels


def test_build_feature_matrix(fixture_results_dir):
    features = build_feature_matrix(fixture_results_dir)

    assert TAG_COLUMN in features.columns
    assert QUERY7_GROWTH_COL in features.columns
    assert QUERY9_RISK_COL in features.columns
    assert features[TAG_COLUMN].nunique() == 8


def test_derive_labels_respects_risk_status(fixture_results_dir):
    features = build_feature_matrix(fixture_results_dir)
    labels = derive_labels(features, 0.33, 0.66)
    labels_by_tag = dict(zip(features[TAG_COLUMN], labels))

    assert labels_by_tag["beta"] == "declining"
    assert labels_by_tag["epsilon"] == "declining"
    assert labels_by_tag["alpha"] in {"growing", "stable"}
