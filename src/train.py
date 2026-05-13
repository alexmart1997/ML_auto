import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import MODEL_PATH, RANDOM_STATE, TEST_SIZE
from src.data import load_data
from src.features import make_preprocessor, split_features_target


def train_model():
    """Обучает модель и сохраняет её в папку models."""
    data = load_data()
    X, y = split_features_target(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(X_train)),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )

    model.fit(X_train, y_train)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    return model, X_test, y_test


if __name__ == "__main__":
    train_model()
    print(f"Модель сохранена: {MODEL_PATH}")
