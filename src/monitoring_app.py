import time

import joblib
import pandas as pd
from fastapi import FastAPI
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app
from sklearn.metrics import f1_score, recall_score, roc_auc_score

try:
    from src.config import MODEL_PATH, REPORTS_DIR, TARGET_COLUMN
    from src.data import load_data, make_train_test_split
    from src.predict import prepare_data
except ImportError:
    from config import MODEL_PATH, REPORTS_DIR, TARGET_COLUMN
    from data import load_data, make_train_test_split
    from predict import prepare_data


app = FastAPI(title="Bank Churn ML Monitoring")

# Метрики качества модели
model_roc_auc = Gauge("model_roc_auc", "ROC-AUC модели на test")
model_f1_score = Gauge("model_f1_score", "F1-score модели на test")
model_recall_score = Gauge("model_recall_score", "Recall модели на test")

# Метрики качества данных
dataset_rows_total = Gauge("dataset_rows_total", "Количество строк в датасете")
dataset_missing_values_total = Gauge(
    "dataset_missing_values_total",
    "Общее количество пропусков в датасете",
)
churn_positive_rate = Gauge("churn_positive_rate", "Доля клиентов с Exited = 1")
avg_churn_probability = Gauge(
    "avg_churn_probability",
    "Средняя вероятность оттока на test",
)

# Метрики запросов к сервису
prediction_requests_total = Counter(
    "prediction_requests_total",
    "Количество запросов на предсказание",
)
prediction_latency_seconds = Histogram(
    "prediction_latency_seconds",
    "Время обработки запроса на предсказание",
)


def load_model():
    """Загружает сохранённую модель."""
    if not MODEL_PATH.exists():
        # Если модель ещё не обучена, обучаем её автоматически для demo-запуска
        try:
            from src.train import train_model
        except ImportError:
            from train import train_model

        train_model()

    return joblib.load(MODEL_PATH)


def update_monitoring_metrics():
    """Пересчитывает ML-метрики и метрики данных."""
    data = load_data()
    model = load_model()
    _, X_test, _, y_test = make_train_test_split(data)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Метрики модели
    model_roc_auc.set(roc_auc_score(y_test, y_proba))
    model_f1_score.set(f1_score(y_test, y_pred, zero_division=0))
    model_recall_score.set(recall_score(y_test, y_pred, zero_division=0))
    avg_churn_probability.set(float(y_proba.mean()))

    # Метрики данных
    dataset_rows_total.set(len(data))
    dataset_missing_values_total.set(int(data.isna().sum().sum()))

    if TARGET_COLUMN in data.columns:
        churn_positive_rate.set(float(data[TARGET_COLUMN].mean()))


@app.on_event("startup")
def startup_event():
    """Считает стартовые метрики при запуске сервиса."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    update_monitoring_metrics()


@app.get("/")
def root():
    """Простая главная страница сервиса."""
    return {"service": "bank-churn-monitoring", "metrics": "/metrics"}


@app.get("/health")
def health():
    """Проверка, что сервис жив."""
    return {"status": "ok"}


@app.post("/predict")
def predict(client: dict):
    """Делает предсказание для одного клиента и обновляет request-метрики."""
    start_time = time.time()
    prediction_requests_total.inc()

    model = load_model()
    data = pd.DataFrame([client])
    X = prepare_data(data)

    predicted_class = int(model.predict(X)[0])
    churn_probability = float(model.predict_proba(X)[:, 1][0])

    prediction_latency_seconds.observe(time.time() - start_time)

    return {
        "predicted_class": predicted_class,
        "churn_probability": churn_probability,
    }


@app.post("/refresh")
def refresh_metrics():
    """Ручной пересчёт метрик модели и данных."""
    update_monitoring_metrics()
    return {"status": "metrics refreshed"}


# Prometheus будет забирать метрики с /metrics
app.mount("/metrics", make_asgi_app())
