from __future__ import annotations

"""Model training workflow for TechPulse."""

from pathlib import Path
import json
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

from src.features import FEATURE_COLUMNS, LABEL_COLUMN, TECH_COLUMN, build_feature_matrix

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
FEATURE_PATH = DATA_DIR / "feature_matrix.csv"

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure logging once for training."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_feature_matrix(path: Path | None = None) -> pd.DataFrame:
    """Load the feature matrix from disk, generating it if missing."""
    path = path or FEATURE_PATH
    if not path.exists():
        LOGGER.warning("Feature matrix missing at %s. Regenerating.", path)
        build_feature_matrix(output_path=path)
    return pd.read_csv(path)


def load_model(model_path: Path) -> Pipeline:
    """Load a serialized model pipeline."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(model_path)


def split_features_labels(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Split the feature matrix into X, y, and technology name series."""
    X = features[FEATURE_COLUMNS]
    y = features[LABEL_COLUMN]
    technologies = features[TECH_COLUMN]
    return X, y, technologies


def build_pipelines(
    random_state: int,
    model_params: dict[str, dict] | None = None,
) -> dict[str, Pipeline]:
    """Construct model pipelines with a scaler and classifier."""
    model_params = model_params or {}

    pipelines: dict[str, Pipeline] = {}
    # Random Forest handles nonlinear relationships in engineered features.
    pipelines["random_forest"] = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=random_state,
                    **model_params.get("random_forest", {}),
                ),
            ),
        ]
    )
    # XGBoost provides gradient-boosted trees with strong performance on tabular data.
    pipelines["xgboost"] = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                XGBClassifier(
                    n_estimators=100,
                    random_state=random_state,
                    eval_metric="mlogloss",
                    **model_params.get("xgboost", {}),
                ),
            ),
        ]
    )
    # Logistic Regression offers a simple linear baseline for interpretability.
    pipelines["logistic_regression"] = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=random_state,
                    multi_class="multinomial",
                    **model_params.get("logistic_regression", {}),
                ),
            ),
        ]
    )
    # KNN captures local neighborhoods within the scaled feature space.
    pipelines["knn"] = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5, **model_params.get("knn", {}))),
        ]
    )
    return pipelines


def save_json(path: Path, payload: dict) -> None:
    """Persist a dictionary as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def evaluate_model(
    name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    class_labels: list[str],
    class_indices: list[int],
    figures_dir: Path,
) -> dict[str, float]:
    """Fit a model, compute metrics, and write confusion matrix plots."""
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    # Classification report captures per-class precision, recall, and F1.
    report = classification_report(
        y_test,
        predictions,
        labels=class_indices,
        target_names=class_labels,
        output_dict=True,
        zero_division=0,
    )
    save_json(MODELS_DIR / f"{name}_metrics.json", report)

    # Macro-averaged metrics provide a balanced view across classes.
    accuracy = accuracy_score(y_test, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="macro", zero_division=0
    )

    # Confusion matrix visualization for report figures.
    cm = confusion_matrix(y_test, predictions, labels=class_indices)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(cmap="Blues", colorbar=False)
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.title(f"{name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(figures_dir / f"confusion_matrix_{name}.png", dpi=150)
    plt.close()

    return {
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
    }


def plot_roc_curves(
    roc_data: dict[str, dict[str, np.ndarray]],
    output_path: Path,
) -> None:
    """Plot ROC-AUC curves for each model using micro-average scores."""
    plt.figure(figsize=(8, 6))
    for model_name, model_data in roc_data.items():
        plt.plot(
            model_data["fpr"],
            model_data["tpr"],
            label=f"{model_name} (AUC={model_data['auc']:.2f})",
        )
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-AUC Curves (Micro-average)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def train_all(
    random_state: int = 42,
    model_params: dict[str, dict] | None = None,
) -> pd.DataFrame:
    """Train all models, save artifacts, and return the comparison table."""
    configure_logging()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load features and create a stratified train-test split.
    features = load_feature_matrix()
    X, y, _ = split_features_labels(features)

    # Encode class labels for models that require numeric targets (e.g., XGBoost).
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_labels = label_encoder.classes_.tolist()
    class_indices = list(range(len(class_labels)))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        stratify=y_encoded,
        random_state=random_state,
    )

    pipelines = build_pipelines(random_state, model_params=model_params)

    comparison_rows: list[dict[str, float | str]] = []
    roc_data: dict[str, dict[str, np.ndarray]] = {}

    for name, pipeline in pipelines.items():
        LOGGER.info("Training %s...", name)
        # Cross-validate on the training split to avoid leakage.
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")
        save_json(MODELS_DIR / f"{name}_cv_scores.json", {"scores": cv_scores.tolist()})

        # Fit on training data and compute evaluation metrics on the test split.
        metrics = evaluate_model(
            name,
            pipeline,
            X_train,
            X_test,
            y_train,
            y_test,
            class_labels,
            class_indices,
            FIGURES_DIR,
        )

        joblib.dump(pipeline, MODELS_DIR / f"{name}.pkl")

        # Compute ROC-AUC curves using micro-average scores.
        if hasattr(pipeline, "predict_proba"):
            probabilities = pipeline.predict_proba(X_test)
            y_binarized = label_binarize(y_test, classes=class_indices)
            auc_score = roc_auc_score(
                y_binarized,
                probabilities,
                multi_class="ovr",
                average="macro",
            )
            fpr, tpr, _ = roc_curve(y_binarized.ravel(), probabilities.ravel())
            roc_data[name] = {"fpr": fpr, "tpr": tpr, "auc": auc_score}
        else:
            auc_score = float("nan")

        comparison_rows.append(
            {
                "model": name,
                "accuracy": metrics["accuracy"],
                "precision_macro": metrics["precision_macro"],
                "recall_macro": metrics["recall_macro"],
                "f1_macro": metrics["f1_macro"],
                "roc_auc_macro": auc_score,
                "cv_mean": float(np.mean(cv_scores)),
                "cv_std": float(np.std(cv_scores)),
            }
        )

    comparison = pd.DataFrame(comparison_rows).sort_values("accuracy", ascending=False)
    comparison_path = REPORTS_DIR / "model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    # Track the best model name for downstream explainability and dashboard steps.
    best_model = comparison.iloc[0]["model"]
    best_path = REPORTS_DIR / "best_model.txt"
    best_path.write_text(str(best_model), encoding="utf-8")
    joblib.dump(label_encoder, MODELS_DIR / "label_encoder.pkl")
    LOGGER.info("Best model by accuracy: %s", best_model)

    if roc_data:
        plot_roc_curves(roc_data, FIGURES_DIR / "roc_auc_curves.png")

    return comparison


def main() -> None:
    """Run training from the command line."""
    train_all()


if __name__ == "__main__":
    main()
