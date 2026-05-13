from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import TARGET_COLUMN


# Признаки, которые не помогают модели или являются идентификаторами
DROP_COLUMNS = ["id", "CustomerId", "Surname"]


def split_features_target(data, target_column=TARGET_COLUMN):
    """Разделяет таблицу на признаки X и целевую переменную y."""
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Удаляем идентификаторы только если они есть в данных
    X = X.drop(columns=[col for col in DROP_COLUMNS if col in X.columns])
    return X, y


def make_preprocessor(X):
    """Создаёт обработчик числовых и категориальных признаков."""
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object", "string"]).columns

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )
