import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .config import CATEGORICAL_COLUMNS, ID_COLUMNS, TARGET


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """Prepare model features in a small, readable ETL step."""
    frame = data.drop(columns=[col for col in ID_COLUMNS if col in data.columns], errors="ignore").copy()

    if "Balance" in frame.columns and "EstimatedSalary" in frame.columns:
        salary = frame["EstimatedSalary"].replace(0, np.nan)
        frame["BalanceToSalary"] = (frame["Balance"] / salary).replace([np.inf, -np.inf], np.nan).fillna(0)

    if "Age" in frame.columns and "Tenure" in frame.columns:
        frame["TenureByAge"] = (frame["Tenure"] / frame["Age"].replace(0, np.nan)).fillna(0)

    return frame.drop(columns=[TARGET], errors="ignore")


def build_preprocessor(data: pd.DataFrame) -> ColumnTransformer:
    prepared = add_features(data)
    categorical_columns = [col for col in CATEGORICAL_COLUMNS if col in prepared.columns]
    numeric_columns = [col for col in prepared.columns if col not in categorical_columns]

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_columns),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ]
    )


def build_feature_pipeline(data: pd.DataFrame) -> Pipeline:
    return Pipeline(
        steps=[
            ("features", FunctionTransformer(add_features, validate=False)),
            ("preprocess", build_preprocessor(data)),
        ]
    )
