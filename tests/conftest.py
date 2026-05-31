from __future__ import annotations

from pathlib import Path

import pytest

from ml_pipeline.config import PipelineConfig


@pytest.fixture()
def fixture_results_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture()
def config(fixture_results_dir: Path, tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(results_dir=fixture_results_dir, artifacts_dir=tmp_path / "artifacts")
