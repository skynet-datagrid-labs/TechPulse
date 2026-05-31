from __future__ import annotations

from pathlib import Path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_query_csvs(results_dir: Path) -> list[Path]:
    return sorted(results_dir.glob("query*.csv"))
