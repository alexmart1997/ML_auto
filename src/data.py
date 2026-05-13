import pandas as pd

from src.config import DATA_PATH, DATA_URL


def load_data(path=DATA_PATH, url=DATA_URL):
    """Загружает датасет из локального файла или скачивает его по ссылке."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        return pd.read_csv(path)

    data = pd.read_csv(url)
    data.to_csv(path, index=False)
    return data
