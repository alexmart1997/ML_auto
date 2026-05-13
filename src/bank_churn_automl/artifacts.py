from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


class TrainedModel:
    def __init__(self, pipeline: Pipeline, model_name: str, validation_metrics: dict[str, float]):
        self.pipeline = pipeline
        self.model_name = model_name
        self.validation_metrics = validation_metrics

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(data)[:, 1]
