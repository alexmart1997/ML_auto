from __future__ import annotations

import time
from typing import Any

import pandas as pd
from sklearn.metrics import f1_score, recall_score, roc_auc_score

from .config import TARGET


def build_monitoring_report(train: pd.DataFrame, current: pd.DataFrame, model: Any) -> dict:
    started = time.perf_counter()
    probabilities = model.predict_proba(current)
    latency_ms = (time.perf_counter() - started) * 1000
    predictions = (probabilities >= 0.5).astype(int)
    y_true = current[TARGET]

    numeric_drift = []
    for column in train.select_dtypes(include="number").columns:
        if column == TARGET:
            continue
        train_mean = float(train[column].mean())
        current_mean = float(current[column].mean())
        train_std = float(train[column].std(ddof=0)) or 1.0
        shift = abs(current_mean - train_mean) / train_std
        numeric_drift.append(
            {
                "feature": column,
                "train_mean": train_mean,
                "current_mean": current_mean,
                "mean_shift_std_units": float(shift),
                "status": "warning" if shift > 0.2 else "ok",
            }
        )

    numeric_drift = sorted(numeric_drift, key=lambda item: item["mean_shift_std_units"], reverse=True)
    warnings = [item for item in numeric_drift if item["status"] == "warning"]

    return {
        "model_quality": {
            "roc_auc": float(roc_auc_score(y_true, probabilities)),
            "f1": float(f1_score(y_true, predictions, zero_division=0)),
            "recall": float(recall_score(y_true, predictions, zero_division=0)),
            "status": "ok" if roc_auc_score(y_true, probabilities) >= 0.75 else "warning",
        },
        "data_quality": {
            "missing_values": int(current.isna().sum().sum()),
            "rows_checked": int(len(current)),
            "numeric_drift": numeric_drift,
            "drift_warnings": len(warnings),
        },
        "infrastructure": {
            "batch_latency_ms": float(latency_ms),
            "rows_per_second": float(len(current) / max(latency_ms / 1000, 1e-9)),
            "cpu_monitoring": "Docker stats / GitHub Actions runtime logs",
            "resource_status": "ok" if latency_ms < 5000 else "warning",
        },
    }
