import argparse

import joblib
import pandas as pd

try:
    from src.config import DROP_COLUMNS, MODEL_PATH, REPORTS_DIR
except ImportError:
    from config import DROP_COLUMNS, MODEL_PATH, REPORTS_DIR


PREDICTIONS_PATH = REPORTS_DIR / "predictions.csv"
EXTRA_COLUMNS = DROP_COLUMNS + ["Exited"]


def load_model(path=MODEL_PATH):
    """Загружает сохранённую модель."""
    return joblib.load(path)


def prepare_data(data):
    """Удаляет лишние колонки перед предсказанием."""
    columns_to_drop = [col for col in EXTRA_COLUMNS if col in data.columns]
    return data.drop(columns=columns_to_drop)


def make_predictions(input_path, output_path=PREDICTIONS_PATH):
    """Делает предсказания для клиентов из CSV-файла."""
    model = load_model()
    data = pd.read_csv(input_path)
    X = prepare_data(data)

    # Предсказываем класс и вероятность оттока
    predicted_class = model.predict(X)
    churn_probability = model.predict_proba(X)[:, 1]

    result = data.copy()
    result["predicted_class"] = predicted_class
    result["churn_probability"] = churn_probability

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    return result


def parse_args():
    """Читает путь к CSV-файлу из аргументов командной строки."""
    parser = argparse.ArgumentParser(description="Предсказание оттока клиентов банка")
    parser.add_argument("input_csv", help="Путь к CSV-файлу с клиентами")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predictions = make_predictions(args.input_csv)

    print(f"Предсказания сохранены: {PREDICTIONS_PATH}")
    print(f"Количество строк: {len(predictions)}")
