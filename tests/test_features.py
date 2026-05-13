import pandas as pd
from sklearn.compose import ColumnTransformer

from src.features import build_preprocessor, get_feature_columns


def make_sample_features():
    """Создаёт маленькую таблицу признаков для проверки препроцессора."""
    return pd.DataFrame(
        {
            "CreditScore": [650, 720],
            "Age": [35, 42],
            "Tenure": [3, 5],
            "Balance": [0.0, 10000.0],
            "NumOfProducts": [1, 2],
            "HasCrCard": [1, 0],
            "IsActiveMember": [1, 1],
            "EstimatedSalary": [100000.0, 90000.0],
            "Geography": ["France", "Spain"],
            "Gender": ["Male", "Female"],
        }
    )


def test_get_feature_columns_returns_expected_columns():
    """Проверяем, что списки признаков содержат нужные колонки."""
    numerical_features, categorical_features = get_feature_columns()

    assert "CreditScore" in numerical_features
    assert "EstimatedSalary" in numerical_features
    assert "Geography" in categorical_features
    assert "Gender" in categorical_features


def test_build_preprocessor_created_without_errors():
    """Проверяем, что build_preprocessor создаёт рабочий ColumnTransformer."""
    X = make_sample_features()

    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)

    assert isinstance(preprocessor, ColumnTransformer)
    assert transformed.shape[0] == len(X)
