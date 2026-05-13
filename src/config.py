from pathlib import Path


# Корневая папка проекта
ROOT_DIR = Path(__file__).resolve().parents[1]

# Основные папки проекта
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

# Пути к файлам
DATA_URL = "https://github.com/alexmart1997/Case_1/blob/main/train.csv?raw=true"
DATA_PATH = DATA_DIR / "train.csv"
MODEL_PATH = MODELS_DIR / "model.joblib"
METRICS_PATH = REPORTS_DIR / "metrics.txt"

# Настройки обучения
TARGET_COLUMN = "Exited"
TEST_SIZE = 0.2
RANDOM_STATE = 42
