import joblib
import mlflow
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

try:
    from src.config import METRICS_PATH, MODEL_PATH, RANDOM_STATE
    from src.data import load_data, make_train_test_split
    from src.features import build_preprocessor
except ImportError:
    from config import METRICS_PATH, MODEL_PATH, RANDOM_STATE
    from data import load_data, make_train_test_split
    from features import build_preprocessor


def get_models():
    """Возвращает набор моделей для автоматического сравнения."""
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier(
            random_state=RANDOM_STATE,
        ),
    }


def calculate_metrics(model, X_test, y_test):
    """Считает основные метрики качества модели."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def setup_mlflow():
    """Настраивает локальный MLflow experiment."""
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("bank_churn_automl")


def train_model():
    """Обучает несколько моделей, логирует MLflow и сохраняет лучшую."""
    setup_mlflow()

    data = load_data()
    X_train, X_test, y_train, y_test = make_train_test_split(data)

    results = []
    run_ids = {}
    best_model = None
    best_model_name = None
    best_run_id = None
    best_roc_auc = -1

    for model_name, model in get_models().items():
        # Для каждой модели собираем отдельный Pipeline
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", model),
            ]
        )

        with mlflow.start_run(run_name=model_name) as run:
            pipeline.fit(X_train, y_train)
            metrics = calculate_metrics(pipeline, X_test, y_test)

            # Логируем название, параметры и метрики модели
            mlflow.set_tag("model_name", model_name)
            mlflow.log_param("model_name", model_name)
            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)

            run_ids[model_name] = run.info.run_id

        results.append({"model": model_name, **metrics})

        # Выбираем лучшую модель по ROC-AUC
        if metrics["roc_auc"] > best_roc_auc:
            best_roc_auc = metrics["roc_auc"]
            best_model = pipeline
            best_model_name = model_name
            best_run_id = run_ids[model_name]

    # Сохраняем таблицу метрик
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    metrics_df.to_csv(METRICS_PATH, index=False)

    # Сохраняем лучшую модель локально
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    # Логируем metrics.csv во все MLflow runs
    for run_id in run_ids.values():
        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifact(METRICS_PATH)

    # Логируем лучшую модель как MLflow artifact
    with mlflow.start_run(run_id=best_run_id):
        mlflow.set_tag("best_model", "true")
        mlflow.log_artifact(MODEL_PATH, artifact_path="best_model")

    return best_model, X_test, y_test, best_model_name, best_roc_auc


if __name__ == "__main__":
    _, _, _, best_model_name, best_roc_auc = train_model()

    print(f"Лучшая модель: {best_model_name}")
    print(f"ROC-AUC: {best_roc_auc:.4f}")
