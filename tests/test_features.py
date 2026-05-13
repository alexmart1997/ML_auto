import pandas as pd

from src.features import make_preprocessor, split_features_target


def test_split_features_target_removes_service_columns():
    """Проверяем отделение целевой переменной и удаление ID-полей."""
    data = pd.DataFrame(
        {
            "id": [1, 2],
            "CustomerId": [100, 200],
            "Surname": ["Ivanov", "Petrov"],
            "CreditScore": [650, 720],
            "Exited": [0, 1],
        }
    )

    X, y = split_features_target(data)

    assert "Exited" not in X.columns
    assert "id" not in X.columns
    assert "CustomerId" not in X.columns
    assert "Surname" not in X.columns
    assert y.tolist() == [0, 1]


def test_make_preprocessor_transforms_data():
    """Проверяем, что препроцессор обрабатывает числовые и текстовые признаки."""
    X = pd.DataFrame(
        {
            "CreditScore": [650, 720],
            "Age": [35, 42],
            "Geography": ["France", "Spain"],
            "Gender": ["Male", "Female"],
        }
    )

    preprocessor = make_preprocessor(X)
    transformed = preprocessor.fit_transform(X)

    assert transformed.shape[0] == 2
