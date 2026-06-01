from __future__ import annotations

"""Shared pytest fixtures for the TechPulse pipeline."""

from pathlib import Path

import pandas as pd
import pytest

from src.features import FEATURE_COLUMNS, build_feature_matrix


@pytest.fixture()
def fixture_results_dir() -> Path:
    """Return the path to the test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture()
def feature_matrix(fixture_results_dir: Path, tmp_path: Path) -> pd.DataFrame:
    """Create a feature matrix from fixture results and return it."""
    output_path = tmp_path / "feature_matrix.csv"
    return build_feature_matrix(results_dir=fixture_results_dir, output_path=output_path)


@pytest.fixture()
def synthetic_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Provide a separable synthetic dataset for fast model tests."""
    data = pd.DataFrame(
        {
            "technology_health_score": [2, 2.1, 1.9, -2, -2.1, -1.8],
            "growth_momentum_index": [3, 2.8, 3.2, -3, -2.9, -3.1],
            "question_quality_score": [2.5, 2.6, 2.4, -2.5, -2.4, -2.6],
            "company_diversity_score": [1.8, 1.9, 2.0, -1.8, -1.9, -2.0],
            "sentiment_delta": [1.0, 1.1, 0.9, -1.0, -1.1, -0.9],
            "adoption_velocity": [1.5, 1.4, 1.6, -1.5, -1.6, -1.4],
            "community_decay_rate": [0.1, 0.2, 0.0, 0.9, 1.0, 0.8],
        }
    )
    labels = pd.Series([1, 1, 1, 0, 0, 0])
    return data[FEATURE_COLUMNS], labels
