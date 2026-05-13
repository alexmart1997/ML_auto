from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score

from src.config import METRICS_PATH
from src.train import train_model


def evaluate_model():
    """Обучает модель, оценивает качество и сохраняет отчёт."""
    model, X_test, y_test = train_model()
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics_text = "\n".join(
        [
            f"Accuracy: {accuracy_score(y_test, y_pred):.4f}",
            f"F1-score: {f1_score(y_test, y_pred):.4f}",
            f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}",
            "",
            "Classification report:",
            classification_report(y_test, y_pred),
        ]
    )

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(metrics_text, encoding="utf-8")

    return metrics_text


if __name__ == "__main__":
    print(evaluate_model())
    print(f"Отчёт сохранён: {METRICS_PATH}")
