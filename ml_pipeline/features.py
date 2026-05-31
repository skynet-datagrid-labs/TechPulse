from __future__ import annotations

from functools import reduce
from pathlib import Path

import pandas as pd

from ml_pipeline.config import QUERY7_GROWTH_COL, QUERY9_RISK_COL, TAG_COLUMN, TEXT_COLUMNS
from ml_pipeline.io import list_query_csvs


def _normalize_column_name(name: str) -> str:
    return (
        name.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("%", "pct")
    )


def _load_tag_frame(csv_path: Path) -> pd.DataFrame | None:
    df = pd.read_csv(csv_path)
    df = df.rename(columns={col: _normalize_column_name(col) for col in df.columns})
    if TAG_COLUMN not in df.columns:
        return None

    text_columns = [col for col in df.columns if col in TEXT_COLUMNS]
    numeric_columns = [col for col in df.columns if col not in text_columns and col != TAG_COLUMN]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in text_columns:
        df[col] = df[col].astype("string")

    prefix = csv_path.stem
    rename = {col: f"{prefix}__{col}" for col in df.columns if col != TAG_COLUMN}
    df = df.rename(columns=rename)

    return df[[TAG_COLUMN] + list(rename.values())]


def build_feature_matrix(results_dir: Path) -> pd.DataFrame:
    tag_frames: list[pd.DataFrame] = []
    for csv_path in list_query_csvs(results_dir):
        frame = _load_tag_frame(csv_path)
        if frame is not None:
            tag_frames.append(frame)

    if not tag_frames:
        raise FileNotFoundError("No tag-level query CSVs were found in the results directory.")

    features = reduce(
        lambda left, right: left.merge(right, on=TAG_COLUMN, how="outer"),
        tag_frames,
    )
    return features


def derive_labels(features: pd.DataFrame, low_quantile: float, high_quantile: float) -> pd.Series:
    if QUERY7_GROWTH_COL not in features.columns:
        raise KeyError(f"Missing growth metric column: {QUERY7_GROWTH_COL}")

    growth = pd.to_numeric(features[QUERY7_GROWTH_COL], errors="coerce")
    low_threshold = growth.quantile(low_quantile)
    high_threshold = growth.quantile(high_quantile)

    risk_status = features.get(QUERY9_RISK_COL, pd.Series(index=features.index, dtype="string"))
    risk_status = risk_status.astype("string").fillna("")
    high_risk = risk_status.str.contains("critical|high|medium", case=False, regex=True)

    labels = pd.Series("stable", index=features.index, dtype="string")
    labels[growth >= high_threshold] = "growing"
    labels[(growth <= low_threshold) | high_risk] = "declining"
    return labels
