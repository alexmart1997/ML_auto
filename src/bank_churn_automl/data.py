from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import TARGET


def load_train_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    if TARGET not in data.columns:
        raise ValueError(f"Target column '{TARGET}' was not found in {path}")
    return data


def load_inference_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)
