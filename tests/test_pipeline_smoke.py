import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from bank_churn_automl.features import build_feature_pipeline


def test_feature_pipeline_and_model_smoke():
    data = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "CustomerId": [11, 12, 13, 14],
            "Surname": ["A", "B", "C", "D"],
            "CreditScore": [600, 700, 650, 720],
            "Geography": ["France", "Spain", "France", "Germany"],
            "Gender": ["Female", "Male", "Female", "Male"],
            "Age": [30.0, 45.0, 38.0, 52.0],
            "Tenure": [2, 4, 3, 7],
            "Balance": [0.0, 1000.0, 0.0, 5000.0],
            "NumOfProducts": [2, 1, 2, 1],
            "HasCrCard": [1.0, 1.0, 0.0, 1.0],
            "IsActiveMember": [1.0, 0.0, 1.0, 0.0],
            "EstimatedSalary": [10000.0, 20000.0, 12000.0, 25000.0],
            "Exited": [0, 1, 0, 1],
        }
    )

    x = data.drop(columns=["Exited"])
    y = data["Exited"]
    pipeline = Pipeline(
        steps=[
            ("features", build_feature_pipeline(data)),
            ("model", LogisticRegression(max_iter=200)),
        ]
    )
    pipeline.fit(x, y)

    assert pipeline.predict_proba(x).shape == (len(data), 2)
