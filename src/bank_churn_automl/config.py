from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

TRAIN_PATH = RAW_DATA_DIR / "train.csv"
TEST_PATH = RAW_DATA_DIR / "test.csv"
MODEL_PATH = MODELS_DIR / "best_model.pkl"
METRICS_PATH = REPORTS_DIR / "metrics.json"
MONITORING_PATH = REPORTS_DIR / "monitoring_report.json"

TARGET = "Exited"
ID_COLUMNS = ["id", "CustomerId", "Surname"]
CATEGORICAL_COLUMNS = ["Geography", "Gender"]
