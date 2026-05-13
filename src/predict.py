import joblib
import pandas as pd

from src.config import MODEL_PATH


def load_model(path=MODEL_PATH):
    """Загружает обученную модель."""
    return joblib.load(path)


def predict(data, model_path=MODEL_PATH):
    """Возвращает предсказания для новых клиентов."""
    model = load_model(model_path)

    if isinstance(data, dict):
        data = pd.DataFrame([data])

    return model.predict(data)


if __name__ == "__main__":
    # Пример одного клиента для быстрой проверки
    client = {
        "CreditScore": 650,
        "Geography": "France",
        "Gender": "Male",
        "Age": 35,
        "Tenure": 5,
        "Balance": 50000,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 100000,
    }

    result = predict(client)[0]
    print(f"Прогноз Exited: {result}")
