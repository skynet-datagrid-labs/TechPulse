from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

TAG_COLUMN = "tag"
TEXT_COLUMNS = {"risk_status", "quality_category"}

QUERY7_GROWTH_COL = "query7__growth_percentage"
QUERY9_RISK_COL = "query9__risk_status"
FEATURE_QUERY_IDS = ("query7", "query9")
LABEL_COLUMN = "momentum_label"


@dataclass(frozen=True)
class PipelineConfig:
    results_dir: Path = Path("results")
    artifacts_dir: Path = Path("artifacts")
    label_growth_thresholds: tuple[float, float] = (40.0, 70.0)
    random_state: int = 42
    test_size: float = 0.2
    val_size: float = 0.1
