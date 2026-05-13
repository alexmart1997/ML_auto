from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .artifacts import TrainedModel
from .config import FIGURES_DIR, METRICS_PATH, MODEL_PATH, MONITORING_PATH, TARGET, TRAIN_PATH
from .data import load_train_data
from .features import build_feature_pipeline
from .monitor import build_monitoring_report


def candidate_models() -> list[tuple[str, object]]:
    return [
        (
            "logistic_regression_balanced",
            LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=42),
        ),
        (
            "random_forest",
            RandomForestClassifier(n_estimators=120, max_depth=8, min_samples_leaf=30, random_state=42, n_jobs=-1),
        ),
        (
            "gradient_boosting",
            GradientBoostingClassifier(n_estimators=120, learning_rate=0.05, max_depth=3, random_state=42),
        ),
    ]


def calculate_metrics(y_true: pd.Series, probabilities) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
    }


def run_training(train_path: Path = TRAIN_PATH) -> TrainedModel:
    data = load_train_data(train_path)
    train_data, valid_data = train_test_split(
        data,
        test_size=0.2,
        random_state=42,
        stratify=data[TARGET],
    )
    x_train = train_data.drop(columns=[TARGET])
    y_train = train_data[TARGET]
    x_valid = valid_data.drop(columns=[TARGET])
    y_valid = valid_data[TARGET]

    leaderboard = []
    best: tuple[str, Pipeline, dict[str, float]] | None = None
    for name, model in candidate_models():
        pipeline = Pipeline(
            steps=[
                ("features", build_feature_pipeline(train_data)),
                ("model", model),
            ]
        )
        pipeline.fit(x_train, y_train)
        probabilities = pipeline.predict_proba(x_valid)[:, 1]
        metrics = calculate_metrics(y_valid, probabilities)
        leaderboard.append({"model": name, **metrics})
        if best is None or metrics["roc_auc"] > best[2]["roc_auc"]:
            best = (name, pipeline, metrics)

    assert best is not None
    model_name, pipeline, metrics = best
    artifact = TrainedModel(pipeline=pipeline, model_name=model_name, validation_metrics=metrics)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    with MODEL_PATH.open("wb") as file:
        pickle.dump(artifact, file)

    report = {
        "best_model": model_name,
        "validation_metrics": metrics,
        "leaderboard": sorted(leaderboard, key=lambda item: item["roc_auc"], reverse=True),
        "data": {
            "rows": int(len(data)),
            "features_after_preprocessing": int(pipeline[:-1].transform(x_valid.head(1)).shape[1]),
            "target_rate": float(data[TARGET].mean()),
        },
    }
    METRICS_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    monitoring = build_monitoring_report(train_data, valid_data, artifact)
    MONITORING_PATH.write_text(json.dumps(monitoring, indent=2, ensure_ascii=False), encoding="utf-8")
    write_svg_reports(report, monitoring)
    return artifact


def write_svg_reports(report: dict, monitoring: dict) -> None:
    leaderboard = report["leaderboard"][:6]
    max_auc = max(item["roc_auc"] for item in leaderboard)
    bars = []
    for index, item in enumerate(leaderboard):
        width = int(420 * item["roc_auc"] / max_auc)
        y = 50 + index * 42
        bars.append(
            f'<text x="20" y="{y + 18}" font-size="13">{item["model"][:34]}</text>'
            f'<rect x="250" y="{y}" width="{width}" height="24" fill="#2f7d6d"/>'
            f'<text x="{260 + width}" y="{y + 18}" font-size="13">{item["roc_auc"]:.3f}</text>'
        )
    (FIGURES_DIR / "automl_leaderboard.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="760" height="340">'
        '<rect width="100%" height="100%" fill="#ffffff"/>'
        '<text x="20" y="30" font-size="20" font-weight="700">AutoML leaderboard by ROC-AUC</text>'
        + "".join(bars)
        + "</svg>",
        encoding="utf-8",
    )

    drift_items = monitoring["data_quality"]["numeric_drift"][:8]
    drift_bars = []
    for index, item in enumerate(drift_items):
        width = min(420, int(item["mean_shift_std_units"] * 160))
        y = 50 + index * 36
        color = "#b94e48" if item["status"] == "warning" else "#5d8aa8"
        drift_bars.append(
            f'<text x="20" y="{y + 16}" font-size="13">{item["feature"]}</text>'
            f'<rect x="190" y="{y}" width="{width}" height="20" fill="{color}"/>'
            f'<text x="{200 + width}" y="{y + 16}" font-size="13">{item["mean_shift_std_units"]:.2f}</text>'
        )
    (FIGURES_DIR / "monitoring_drift.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="360">'
        '<rect width="100%" height="100%" fill="#ffffff"/>'
        '<text x="20" y="30" font-size="20" font-weight="700">Data drift monitor</text>'
        + "".join(drift_bars)
        + "</svg>",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=Path, default=TRAIN_PATH)
    args = parser.parse_args()
    model = run_training(args.train_path)
    print(json.dumps({"best_model": model.model_name, **model.validation_metrics}, indent=2))


if __name__ == "__main__":
    main()
