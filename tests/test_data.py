import pandas as pd
import pytest

from bank_churn_automl.data import load_train_data


def test_load_train_data_requires_target_column(tmp_path):
    path = tmp_path / "train.csv"
    pd.DataFrame({"CreditScore": [600, 700]}).to_csv(path, index=False)

    with pytest.raises(ValueError):
        load_train_data(path)
