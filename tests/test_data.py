import pandas as pd

from src.data import load_data


def test_load_data_from_local_file(tmp_path):
    """Проверяем, что данные читаются из локального CSV."""
    data_path = tmp_path / "train.csv"
    expected = pd.DataFrame(
        {
            "CreditScore": [650, 720],
            "Geography": ["France", "Spain"],
            "Exited": [0, 1],
        }
    )
    expected.to_csv(data_path, index=False)

    actual = load_data(path=data_path)

    assert actual.shape == (2, 3)
    assert "Exited" in actual.columns
