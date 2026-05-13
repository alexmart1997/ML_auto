import pandas as pd

from src.data import load_data, make_train_test_split, split_features_target


def make_sample_data():
    """Создаёт маленький датасет для тестов."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "CustomerId": [101, 102, 103, 104],
            "Surname": ["Ivanov", "Petrov", "Sidorov", "Smirnov"],
            "CreditScore": [650, 720, 590, 810],
            "Geography": ["France", "Spain", "Germany", "France"],
            "Gender": ["Male", "Female", "Male", "Female"],
            "Age": [35, 42, 28, 50],
            "Tenure": [3, 5, 1, 8],
            "Balance": [0.0, 10000.0, 50000.0, 75000.0],
            "NumOfProducts": [1, 2, 1, 2],
            "HasCrCard": [1, 0, 1, 1],
            "IsActiveMember": [1, 1, 0, 0],
            "EstimatedSalary": [100000.0, 90000.0, 70000.0, 120000.0],
            "Exited": [0, 0, 1, 1],
        }
    )


def test_load_data_not_empty_and_has_target(tmp_path):
    """Проверяем, что данные загружаются, не пустые и содержат Exited."""
    data_path = tmp_path / "train.csv"
    make_sample_data().to_csv(data_path, index=False)

    data = load_data(path=data_path)

    assert not data.empty
    assert "Exited" in data.columns


def test_split_features_target_removes_extra_columns():
    """Проверяем удаление технических и идентификационных колонок."""
    data = make_sample_data()

    X, y = split_features_target(data)

    assert "id" not in X.columns
    assert "CustomerId" not in X.columns
    assert "Surname" not in X.columns
    assert "Exited" not in X.columns
    assert not X.empty
    assert not y.empty


def test_train_test_split_returns_not_empty_parts():
    """Проверяем, что train/test split возвращает непустые выборки."""
    data = make_sample_data()

    X_train, X_test, y_train, y_test = make_train_test_split(data, test_size=0.5)

    assert not X_train.empty
    assert not X_test.empty
    assert not y_train.empty
    assert not y_test.empty
