from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd

from .config import MODEL_PATH, TEST_PATH


def predict(input_path: Path = TEST_PATH, output_path: Path = Path("reports/predictions.csv")) -> pd.DataFrame:
    with MODEL_PATH.open("rb") as file:
        model = pickle.load(file)
    data = pd.read_csv(input_path)
    result = pd.DataFrame({"id": data["id"], "Exited": model.predict_proba(data)})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=Path, default=TEST_PATH)
    parser.add_argument("--output-path", type=Path, default=Path("reports/predictions.csv"))
    args = parser.parse_args()
    output = predict(args.input_path, args.output_path)
    print(f"Saved {len(output)} predictions to {args.output_path}")


if __name__ == "__main__":
    main()
