import sys
from pathlib import Path

import joblib
import matplotlib

sys.path.append(str(Path(__file__).resolve().parents[1]))
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score, roc_auc_score

try:
    from src.config import MODEL_PATH, REPORTS_DIR, TARGET_COLUMN
    from src.data import load_data, make_train_test_split
    from src.features import get_feature_columns
except ImportError:
    from config import MODEL_PATH, REPORTS_DIR, TARGET_COLUMN
    from data import load_data, make_train_test_split
    from features import get_feature_columns


MONITORING_REPORT_PATH = REPORTS_DIR / "monitoring_report.txt"
TARGET_DISTRIBUTION_PATH = REPORTS_DIR / "target_distribution.png"
PROBABILITY_DISTRIBUTION_PATH = REPORTS_DIR / "churn_probability_distribution.png"
MISSING_VALUES_PATH = REPORTS_DIR / "missing_values.png"


def check_input_data(data):
    """Проверяет базовое качество входных данных."""
    numerical_features, categorical_features = get_feature_columns()
    required_features = numerical_features + categorical_features
    missing_features = [col for col in required_features if col not in data.columns]

    report_lines = [
        "Проверка входных данных",
        "=" * 30,
        f"Количество строк: {data.shape[0]}",
        f"Количество колонок: {data.shape[1]}",
        "",
        "Доля пропусков по колонкам:",
    ]

    missing_share = data.isna().mean().sort_values(ascending=False)
    for column, share in missing_share.items():
        report_lines.append(f"{column}: {share:.4f}")

    report_lines.extend(
        [
            "",
            "Проверка нужных признаков:",
            f"Все нужные признаки есть: {'да' if not missing_features else 'нет'}",
        ]
    )

    if missing_features:
        report_lines.append(f"Отсутствующие признаки: {', '.join(missing_features)}")

    if TARGET_COLUMN in data.columns:
        report_lines.extend(["", f"Распределение {TARGET_COLUMN}:"])
        target_distribution = data[TARGET_COLUMN].value_counts(normalize=True).sort_index()
        for value, share in target_distribution.items():
            report_lines.append(f"{value}: {share:.4f}")

    return report_lines


def check_model_quality(model, data):
    """Считает качество модели на тестовой выборке."""
    _, X_test, _, y_test = make_train_test_split(data)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
    }
    return metrics, y_proba


def save_target_distribution(data):
    """Сохраняет график распределения целевой переменной."""
    if TARGET_COLUMN not in data.columns:
        return

    counts = data[TARGET_COLUMN].value_counts().sort_index()

    plt.figure(figsize=(6, 4))
    plt.bar(counts.index.astype(str), counts.values)
    plt.title("Распределение Exited")
    plt.xlabel("Exited")
    plt.ylabel("Количество")
    plt.tight_layout()
    plt.savefig(TARGET_DISTRIBUTION_PATH)
    plt.close()


def save_probability_distribution(y_proba):
    """Сохраняет график распределения вероятностей оттока."""
    plt.figure(figsize=(7, 4))
    plt.hist(y_proba, bins=30)
    plt.title("Распределение вероятностей оттока")
    plt.xlabel("Вероятность оттока")
    plt.ylabel("Количество")
    plt.tight_layout()
    plt.savefig(PROBABILITY_DISTRIBUTION_PATH)
    plt.close()


def save_missing_values(data):
    """Сохраняет bar chart доли пропусков по колонкам."""
    missing_share = data.isna().mean().sort_values(ascending=False)

    plt.figure(figsize=(10, 5))
    plt.bar(missing_share.index, missing_share.values)
    plt.title("Доля пропусков по колонкам")
    plt.xlabel("Колонка")
    plt.ylabel("Доля пропусков")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(MISSING_VALUES_PATH)
    plt.close()


def run_monitoring():
    """Запускает мониторинг данных и модели."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data()
    model = joblib.load(MODEL_PATH)

    report_lines = check_input_data(data)
    metrics, y_proba = check_model_quality(model, data)

    report_lines.extend(
        [
            "",
            "Проверка качества модели",
            "=" * 30,
            f"ROC-AUC: {metrics['roc_auc']:.4f}",
            f"F1: {metrics['f1']:.4f}",
            f"Recall: {metrics['recall']:.4f}",
        ]
    )

    save_target_distribution(data)
    save_probability_distribution(y_proba)
    save_missing_values(data)

    MONITORING_REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    metrics = run_monitoring()

    print(f"Monitoring report сохранён: {MONITORING_REPORT_PATH}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
