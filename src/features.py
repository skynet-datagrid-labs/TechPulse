from __future__ import annotations

"""Feature engineering pipeline for the TechPulse ML workflow."""

from pathlib import Path
import logging
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"
DATA_DIR = BASE_DIR / "data"
FEATURE_OUTPUT = DATA_DIR / "feature_matrix.csv"

TECH_COLUMN = "technology"
LABEL_COLUMN = "label"
TREND_COLUMN = "question_volume_trend"

FEATURE_COLUMNS = [
    "technology_health_score",
    "growth_momentum_index",
    "question_quality_score",
    "company_diversity_score",
    "sentiment_delta",
    "adoption_velocity",
    "community_decay_rate",
]

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure logging once for the pipeline."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def read_csv_safe(path: Path) -> pd.DataFrame | None:
    """Read a CSV file safely, returning None when it cannot be loaded."""
    if not path.exists():
        LOGGER.warning("Missing results file: %s", path.name)
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001 - surface errors as warnings per requirement
        LOGGER.warning("Failed to read %s (%s). Skipping.", path.name, exc)
        return None


def collect_tags(frames: Iterable[pd.DataFrame | None]) -> pd.Series:
    """Collect a unique list of technologies from available result frames."""
    tags: list[str] = []
    for frame in frames:
        if frame is None or "tag" not in frame.columns:
            continue
        tags.extend(frame["tag"].astype("string").dropna().tolist())
    return pd.Series(sorted(set(tags)), name=TECH_COLUMN, dtype="string")


def add_feature_from_tag(
    features: pd.DataFrame,
    frame: pd.DataFrame | None,
    source_column: str,
    target_column: str,
    transform: callable | None = None,
) -> pd.DataFrame:
    """Attach a numeric feature from a tagged results frame to the feature matrix."""
    if frame is None or "tag" not in frame.columns:
        LOGGER.warning("Skipping %s: missing tag column.", target_column)
        features[target_column] = np.nan
        return features
    if source_column not in frame.columns:
        LOGGER.warning("Skipping %s: missing column %s.", target_column, source_column)
        features[target_column] = np.nan
        return features

    values = pd.to_numeric(frame[source_column], errors="coerce")
    if transform is not None:
        values = transform(values)

    merged = frame[["tag"]].copy()
    merged[target_column] = values
    features = features.merge(merged, left_on=TECH_COLUMN, right_on="tag", how="left")
    features = features.drop(columns=["tag"])
    return features


def build_sentiment_delta(
    query1: pd.DataFrame | None, query10: pd.DataFrame | None
) -> pd.DataFrame | None:
    """Build a sentiment delta proxy using unanswered rates from available results."""
    if (
        query1 is None
        or query10 is None
        or "tag" not in query1.columns
        or "tag" not in query10.columns
        or "avg_unanswered_pct" not in query1.columns
        or "overall_unanswered_rate" not in query10.columns
    ):
        LOGGER.warning("Skipping sentiment_delta: required columns not found.")
        return None

    base_rate = pd.to_numeric(query1["avg_unanswered_pct"], errors="coerce")
    recent_rate = pd.to_numeric(query10["overall_unanswered_rate"], errors="coerce") / 100.0
    sentiment_delta = base_rate - recent_rate

    merged = query1[["tag"]].copy()
    merged["sentiment_delta"] = sentiment_delta
    return merged


def build_company_diversity(query12: pd.DataFrame | None) -> pd.DataFrame | None:
    """Build a company diversity score per technology if tagged data is available."""
    if query12 is None or "tag" not in query12.columns:
        LOGGER.warning("Skipping company_diversity_score: no tag column in query12.")
        return None

    if "diversity_rating" in query12.columns:
        mapping = {"High Diversity": 3, "Medium Diversity": 2, "Low Diversity": 1}
        values = query12["diversity_rating"].map(mapping)
    elif "num_tech_categories" in query12.columns:
        values = pd.to_numeric(query12["num_tech_categories"], errors="coerce")
    elif "num_technologies" in query12.columns:
        values = pd.to_numeric(query12["num_technologies"], errors="coerce")
    else:
        LOGGER.warning("Skipping company_diversity_score: no usable numeric columns.")
        return None

    merged = query12[["tag"]].copy()
    merged["company_diversity_score"] = values
    return merged


def derive_labels(features: pd.DataFrame) -> pd.Series:
    """Derive class labels using momentum and question volume trend thresholds."""
    momentum = pd.to_numeric(features["growth_momentum_index"], errors="coerce")
    trend = features[TREND_COLUMN].astype("string")

    labels = pd.Series("stable", index=features.index, dtype="string")
    growing = (momentum > 70) & (trend == "positive")
    stable = momentum.between(40, 70, inclusive="both") & (trend == "flat")
    declining = (momentum < 40) | (trend == "negative")

    labels[growing] = "growing"
    labels[stable] = "stable"
    labels[declining] = "declining"
    return labels


def impute_and_scale(features: pd.DataFrame) -> pd.DataFrame:
    """Median-impute and standardize continuous feature columns."""
    cleaned = features.copy()
    for column in FEATURE_COLUMNS:
        numeric = pd.to_numeric(cleaned[column], errors="coerce")
        median = numeric.median()
        if pd.isna(median):
            LOGGER.warning("All values missing for %s; filling with 0.", column)
            median = 0.0
        cleaned[column] = numeric.fillna(median)

    scaler = StandardScaler()
    cleaned[FEATURE_COLUMNS] = scaler.fit_transform(cleaned[FEATURE_COLUMNS])
    return cleaned


def build_feature_matrix(
    results_dir: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Build the normalized feature matrix and write it to disk."""
    configure_logging()

    results_dir = results_dir or RESULTS_DIR
    output_path = output_path or FEATURE_OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load the required query outputs from the results directory.
    query7 = read_csv_safe(results_dir / "query7.csv")
    query9 = read_csv_safe(results_dir / "query9.csv")
    query10 = read_csv_safe(results_dir / "query10.csv")
    query12 = read_csv_safe(results_dir / "query12.csv")
    query1 = read_csv_safe(results_dir / "query1.csv")

    # Build the base list of technologies from any available tagged file.
    tags = collect_tags([query7, query9, query10, query12, query1])
    if tags.empty:
        raise ValueError("No tagged results available to build features.")

    features = pd.DataFrame({TECH_COLUMN: tags})

    # Core feature extraction from required queries.
    features = add_feature_from_tag(
        features, query9, "health_score", "technology_health_score", transform=None
    )
    features = add_feature_from_tag(
        features, query7, "growth_percentage", "growth_momentum_index", transform=None
    )

    features = add_feature_from_tag(
        features,
        query10,
        "overall_unanswered_rate",
        "question_quality_score",
        transform=lambda values: 100 - values,
    )

    # Optional feature sources (may be missing in the provided results).
    diversity = build_company_diversity(query12)
    if diversity is None:
        features["company_diversity_score"] = np.nan
    else:
        features = features.merge(
            diversity, left_on=TECH_COLUMN, right_on="tag", how="left"
        ).drop(columns=["tag"])

    sentiment = build_sentiment_delta(query1, query10)
    if sentiment is None:
        features["sentiment_delta"] = np.nan
    else:
        features = features.merge(
            sentiment, left_on=TECH_COLUMN, right_on="tag", how="left"
        ).drop(columns=["tag"])

    features["adoption_velocity"] = np.nan
    LOGGER.warning("Skipping adoption_velocity: no quarterly adoption data found.")

    if "growth_momentum_index" in features.columns:
        momentum_values = pd.to_numeric(features["growth_momentum_index"], errors="coerce")
        features["community_decay_rate"] = np.where(momentum_values < 0, -momentum_values, 0.0)
    else:
        features["community_decay_rate"] = np.nan

    # Trend direction is derived from the growth momentum values.
    momentum_values = pd.to_numeric(features["growth_momentum_index"], errors="coerce")
    features[TREND_COLUMN] = np.where(
        momentum_values > 0,
        "positive",
        np.where(momentum_values < 0, "negative", "flat"),
    )

    # Apply labeling rules and report the class distribution.
    features[LABEL_COLUMN] = derive_labels(features)
    class_distribution = features[LABEL_COLUMN].value_counts()
    LOGGER.info("Class distribution:\n%s", class_distribution.to_string())

    # Impute missing values and scale numeric features.
    features = impute_and_scale(features)
    features = features.drop(columns=[TREND_COLUMN])
    features.to_csv(output_path, index=False)
    LOGGER.info("Feature matrix saved to %s", output_path)

    return features


def main() -> None:
    """Run the feature engineering pipeline from the command line."""
    build_feature_matrix()


if __name__ == "__main__":
    main()
