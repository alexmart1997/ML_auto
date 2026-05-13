# Bank Churn AutoML Project

Учебный ML-проект по предмету **«Автоматизация машинного обучения»**.

Проект решает задачу бинарной классификации: нужно предсказать, уйдёт ли клиент банка.

Целевая переменная:

```text
Exited
```

`Exited = 1` означает, что клиент ушёл.  
`Exited = 0` означает, что клиент остался.

## Что сделано

В проекте реализован полный простой ML-пайплайн:

- загрузка данных из GitHub;
- сохранение датасета локально в `data/train.csv`;
- очистка технических колонок;
- train/test split со stratify;
- preprocessing числовых и категориальных признаков;
- обучение нескольких моделей;
- автоматический выбор лучшей модели по ROC-AUC;
- сохранение модели в `models/best_model.pkl`;
- сохранение метрик в `reports/metrics.csv`;
- оценка модели и построение графиков;
- инференс по CSV-файлу;
- MLflow tracking;
- локальный monitoring report;
- FastAPI ML-сервис с Prometheus-метриками;
- Prometheus + Grafana через Docker Compose;
- GitHub Actions CI.

## Данные

Данные загружаются из:

```text
https://github.com/alexmart1997/Case_1/blob/main/train.csv?raw=true
```

При первом запуске файл скачивается и сохраняется сюда:

```text
data/train.csv
```

В Git датасет не добавляется, потому что он игнорируется через `.gitignore`.

## Структура проекта

```text
ML_auto/
  data/
    .gitkeep
  models/
    .gitkeep
  reports/
    .gitkeep
  src/
    __init__.py
    config.py
    data.py
    features.py
    train.py
    evaluate.py
    predict.py
    monitoring.py
    monitoring_app.py
  tests/
    conftest.py
    test_data.py
    test_features.py
  monitoring/
    prometheus.yml
    grafana/
      dashboards/
        bank_churn_dashboard.json
      provisioning/
        datasources/
          prometheus.yml
        dashboards/
          dashboards.yml
  .github/
    workflows/
      ci.yml
  Dockerfile
  docker-compose.yml
  requirements.txt
  README.md
```

## Основные скрипты

### `src/config.py`

Хранит пути и основные настройки проекта:

- путь к данным;
- путь к модели;
- путь к отчётам;
- имя целевой переменной;
- колонки, которые нужно удалить;
- `test_size`;
- `random_state`.

### `src/data.py`

Отвечает за данные:

- загружает CSV;
- сохраняет локальную копию;
- удаляет лишние признаки;
- делит данные на train/test.

### `src/features.py`

Создаёт preprocessing:

- числовые признаки:
  - `SimpleImputer(strategy="median")`;
  - `StandardScaler`;
- категориальные признаки:
  - `SimpleImputer(strategy="most_frequent")`;
  - `OneHotEncoder(handle_unknown="ignore")`.

### `src/train.py`

Обучает несколько моделей:

- `LogisticRegression`;
- `RandomForestClassifier`;
- `HistGradientBoostingClassifier`.

Для каждой модели считается:

- accuracy;
- precision;
- recall;
- f1;
- roc_auc.

Лучшая модель выбирается по `ROC-AUC`.

### `src/evaluate.py`

Загружает сохранённую модель и строит отчёты:

- `reports/classification_report.txt`;
- `reports/confusion_matrix.png`;
- `reports/roc_curve.png`.

### `src/predict.py`

Делает предсказания по CSV-файлу.

Результат сохраняется в:

```text
reports/predictions.csv
```

### `src/monitoring.py`

Делает простой локальный мониторинг:

- проверяет размер датасета;
- считает пропуски;
- проверяет наличие нужных признаков;
- считает распределение `Exited`;
- считает ROC-AUC, F1, Recall;
- сохраняет monitoring report и графики.

### `src/monitoring_app.py`

FastAPI-приложение для Prometheus и Grafana.

Отдаёт ML-метрики на:

```text
http://localhost:8000/metrics
```

## Установка локально

Перейти в папку проекта:

```powershell
cd "C:\Users\79034\Documents\New project\ML_auto"
```

Создать виртуальное окружение:

```powershell
python -m venv .venv
```

Активировать окружение:

```powershell
.\.venv\Scripts\activate
```

Установить зависимости:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Быстрый запуск ML-пайплайна

### 1. Обучить модель

```powershell
python src/train.py
```

После запуска появятся:

```text
models/best_model.pkl
reports/metrics.csv
```

Пример вывода:

```text
Лучшая модель: HistGradientBoostingClassifier
ROC-AUC: 0.8897
```

### 2. Оценить модель

```powershell
python src/evaluate.py
```

После запуска появятся:

```text
reports/classification_report.txt
reports/confusion_matrix.png
reports/roc_curve.png
```

### 3. Сделать предсказания

```powershell
python src/predict.py data/train.csv
```

После запуска появится:

```text
reports/predictions.csv
```

В файл добавляются колонки:

```text
predicted_class
churn_probability
```

### 4. Запустить локальный мониторинг

```powershell
python src/monitoring.py
```

После запуска появятся:

```text
reports/monitoring_report.txt
reports/target_distribution.png
reports/churn_probability_distribution.png
reports/missing_values.png
```

## MLflow

В `src/train.py` добавлен MLflow tracking.

При обучении логируются:

- название модели;
- параметры модели;
- accuracy;
- precision;
- recall;
- f1;
- roc_auc;
- файл `reports/metrics.csv`;
- лучшая модель как artifact.

Локальная папка MLflow:

```text
mlruns/
```

Запуск MLflow UI:

```powershell
mlflow ui
```

Если команда `mlflow` не находится:

```powershell
python -m mlflow ui
```

Открыть в браузере:

```text
http://127.0.0.1:5000
```

## FastAPI ML-сервис

Локальный запуск без Docker:

```powershell
python src/train.py
uvicorn src.monitoring_app:app --host 0.0.0.0 --port 8000
```

Проверить сервис:

```text
http://localhost:8000
```

Проверить health endpoint:

```text
http://localhost:8000/health
```

Проверить Prometheus-метрики:

```text
http://localhost:8000/metrics
```

Пример запроса на предсказание:

```powershell
curl -X POST http://localhost:8000/predict `
  -H "Content-Type: application/json" `
  -d "{\"CreditScore\":650,\"Geography\":\"France\",\"Gender\":\"Male\",\"Age\":35,\"Tenure\":5,\"Balance\":50000,\"NumOfProducts\":2,\"HasCrCard\":1,\"IsActiveMember\":1,\"EstimatedSalary\":100000}"
```

Пример ответа:

```json
{
  "predicted_class": 0,
  "churn_probability": 0.03
}
```

## Prometheus и Grafana

Проект содержит готовый `docker-compose.yml`, который запускает:

- ML-сервис;
- Prometheus;
- Grafana.

Запуск:

```powershell
docker compose up --build
```

Адреса:

```text
ML-сервис:  http://localhost:8000
Prometheus: http://localhost:9090
Grafana:    http://localhost:3000
```

Логин и пароль Grafana:

```text
admin / admin
```

Dashboard:

```text
Bank Churn ML Monitoring
```

## Метрики Prometheus

FastAPI-сервис отдаёт следующие метрики:

```text
model_roc_auc
model_f1_score
model_recall_score
dataset_rows_total
dataset_missing_values_total
churn_positive_rate
prediction_requests_total
prediction_latency_seconds
avg_churn_probability
```

В Prometheus их можно проверить на странице:

```text
http://localhost:9090
```

Примеры запросов:

```text
model_roc_auc
model_f1_score
dataset_rows_total
avg_churn_probability
prediction_requests_total
```

Проверка targets:

```text
http://localhost:9090/targets
```

Target `bank-churn-app` должен быть в статусе `UP`.

## Что видно в Grafana

В Grafana dashboard отображаются:

- ROC-AUC модели;
- F1-score;
- Recall;
- количество строк в датасете;
- количество пропусков;
- доля клиентов с `Exited = 1`;
- средняя вероятность оттока;
- количество запросов к `/predict`;
- p95 latency предсказаний.

Чтобы графики запросов начали меняться, нужно отправить несколько запросов на `/predict`.

## Тесты

Запуск тестов:

```powershell
python -m pytest
```

Проверяется:

- данные загружаются;
- данные не пустые;
- есть целевая переменная `Exited`;
- удаляются `id`, `CustomerId`, `Surname`;
- preprocessing создаётся без ошибок;
- train/test split возвращает непустые выборки.

## GitHub Actions CI

В проект добавлен workflow:

```text
.github/workflows/ci.yml
```

CI запускается на:

- `push`;
- `pull_request`.

Pipeline делает:

```text
install dependencies
run pytest
run python src/train.py
run python src/evaluate.py
```

## Docker

Сборка Docker-образа:

```powershell
docker build -t bank-churn-ml .
```

Запуск контейнера:

```powershell
docker run --rm bank-churn-ml
```

Полный monitoring stack:

```powershell
docker compose up --build
```

Если Docker ругается на `.pytest_cache`, в проекте есть `.dockerignore`, который исключает кэш и локальные артефакты из Docker build context.

## Возможные проблемы

### `docker` не найден

Нужно установить и запустить Docker Desktop:

```text
https://www.docker.com/products/docker-desktop/
```

Проверка:

```powershell
docker --version
docker compose version
```

### Docker Engine не запущен

Если ошибка похожа на:

```text
failed to connect to the docker API
```

нужно открыть Docker Desktop и дождаться, пока он полностью запустится.

### Конфликт `protobuf` и `streamlit`

Лучше использовать отдельное виртуальное окружение:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### `ModuleNotFoundError: No module named 'src'`

Для тестов добавлен файл:

```text
tests/conftest.py
```

Он добавляет корень проекта в `PYTHONPATH`.

## Итог

Этот проект показывает базовый вариант автоматизации ML-пайплайна:

```text
данные -> preprocessing -> обучение моделей -> выбор лучшей модели
-> оценка -> инференс -> мониторинг -> CI -> Docker
```

Код специально сделан простым и понятным, без сложной production-архитектуры.
