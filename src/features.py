from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import DROP_COLUMNS, TARGET_COLUMN


def get_feature_columns():
    """Возвращает списки числовых и категориальных признаков."""
    numerical_features = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
    ]

    categorical_features = [
        "Geography",
        "Gender",
    ]

    return numerical_features, categorical_features


def build_preprocessor():
    """Создаёт препроцессор для подготовки признаков."""
    numerical_features, categorical_features = get_feature_columns()

    # Обработка числовых признаков: заполняем пропуски и масштабируем
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Обработка категориальных признаков: заполняем пропуски и кодируем
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def split_features_target(data, target_column=TARGET_COLUMN):
    """Разделяет данные на признаки X и целевую переменную y."""
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Удаляем технические и идентификационные признаки
    columns_to_drop = [col for col in DROP_COLUMNS if col in X.columns]
    X = X.drop(columns=columns_to_drop)

    return X, y


def make_preprocessor(X=None):
    """Оставлено для совместимости со старым кодом проекта."""
    if X is None:
        return build_preprocessor()

    numerical_features, categorical_features = get_feature_columns()
    numerical_features = [col for col in numerical_features if col in X.columns]
    categorical_features = [col for col in categorical_features if col in X.columns]

    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
