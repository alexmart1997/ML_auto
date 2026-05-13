import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from src.config import MODEL_PATH, REPORTS_DIR
    from src.data import load_data, make_train_test_split
except ImportError:
    from config import MODEL_PATH, REPORTS_DIR
    from data import load_data, make_train_test_split


CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"
ROC_CURVE_PATH = REPORTS_DIR / "roc_curve.png"
CLASSIFICATION_REPORT_PATH = REPORTS_DIR / "classification_report.txt"


def load_saved_model(path=MODEL_PATH):
    """Загружает сохранённую модель."""
    return joblib.load(path)


def save_confusion_matrix(model, X_test, y_test):
    """Сохраняет confusion matrix как картинку."""
    display = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    display.ax_.set_title("Confusion matrix")

    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.close()


def save_roc_curve(model, X_test, y_test):
    """Сохраняет ROC-кривую как картинку."""
    display = RocCurveDisplay.from_estimator(model, X_test, y_test)
    display.ax_.set_title("ROC curve")

    plt.tight_layout()
    plt.savefig(ROC_CURVE_PATH)
    plt.close()


def evaluate_model():
    """Оценивает сохранённую модель на тестовой выборке."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model = load_saved_model()
    data = load_data()
    _, X_test, _, y_test = make_train_test_split(data)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Считаем основные метрики качества
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    report = classification_report(y_test, y_pred)
    CLASSIFICATION_REPORT_PATH.write_text(report, encoding="utf-8")

    save_confusion_matrix(model, X_test, y_test)
    save_roc_curve(model, X_test, y_test)

    return metrics


if __name__ == "__main__":
    metrics = evaluate_model()

    print("Метрики на test:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    print(f"Classification report сохранён: {CLASSIFICATION_REPORT_PATH}")
    print(f"Confusion matrix сохранена: {CONFUSION_MATRIX_PATH}")
    print(f"ROC-кривая сохранена: {ROC_CURVE_PATH}")
