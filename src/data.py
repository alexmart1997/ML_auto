import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from src.config import (
        DATA_PATH,
        DATA_URL,
        DROP_COLUMNS,
        RANDOM_STATE,
        TARGET_COLUMN,
        TEST_SIZE,
    )
except ImportError:
    from config import (
        DATA_PATH,
        DATA_URL,
        DROP_COLUMNS,
        RANDOM_STATE,
        TARGET_COLUMN,
        TEST_SIZE,
    )


def load_data(path=DATA_PATH, url=DATA_URL):
    """Загружает данные из CSV-файла или скачивает их по ссылке."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Если файл уже есть локально, просто читаем его
    if path.exists():
        return pd.read_csv(path)

    # Если файла нет, скачиваем датасет и сохраняем в data/train.csv
    data = pd.read_csv(url)
    data.to_csv(path, index=False)
    return data


def split_features_target(data, target_column=TARGET_COLUMN, drop_columns=DROP_COLUMNS):
    """Разделяет данные на признаки X и целевую переменную y."""
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Удаляем технические и идентификационные признаки
    columns_to_drop = [col for col in drop_columns if col in X.columns]
    X = X.drop(columns=columns_to_drop)

    return X, y


def make_train_test_split(
    data,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
):
    """Делит данные на train/test с сохранением баланса классов."""
    X, y = split_features_target(data)

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


if __name__ == "__main__":
    # Быстрая проверка загрузки и разбиения данных
    df = load_data()
    X_train, X_test, y_train, y_test = make_train_test_split(df)

    print(f"Данные загружены: {df.shape}")
    print(f"Train: X={X_train.shape}, y={y_train.shape}")
    print(f"Test: X={X_test.shape}, y={y_test.shape}")
    print(f"Целевая переменная: {TARGET_COLUMN}")
