import json
from pathlib import Path

import pandas as pd
from joblib import load
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

HERE = Path(__file__).resolve().parent          # .../src
ROOT = HERE.parent                              # project root

TEST_CSV = ROOT / "data" / "test.csv"
RUN_DIR  = ROOT / "runs" / "xgb_run_01"
MODEL_PATH = RUN_DIR / "model.joblib" # this one is from train.py
COLS_PATH  = RUN_DIR / "train_columns.json"

TARGET = "oil_consumption" 


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not COLS_PATH.exists():
        raise FileNotFoundError(f"Train columns not found: {COLS_PATH}")
    if not TEST_CSV.exists():
        raise FileNotFoundError(f"Test CSV not found: {TEST_CSV}")

    # Load artifacts
    model = load(MODEL_PATH)
    with open(COLS_PATH) as f:
        train_columns = json.load(f)

    # Load & align test
    test = pd.read_csv(TEST_CSV)
    if TARGET not in test.columns:
        raise ValueError(f"Target column '{TARGET}' not found in test CSV.")

    y_test = test[TARGET].astype(float)
    X_test_all = test.drop(columns=[TARGET], errors="ignore")
    # Align to training schema: add missing cols as 0, drop extras
    X_test = X_test_all.reindex(columns=train_columns, fill_value=0)

    preds = model.predict(X_test)

    # Metrics
    rmse = root_mean_squared_error(y_test, preds)
    mae  = mean_absolute_error(y_test, preds)
    r2   = r2_score(y_test, preds)

    # Save outputs
    out_preds = test.copy()
    out_preds["prediction"] = preds
    out_preds.to_csv(RUN_DIR / "test_predictions.csv", index=False)

    pd.DataFrame([{"rmse": rmse, "mae": mae, "r2": r2}]).to_csv(
        RUN_DIR / "test_metrics.csv", index=False
    )

    print({"rmse": rmse, "mae": mae, "r2": r2})
    print(f"Saved predictions -> {RUN_DIR/'test_predictions.csv'}")
    print(f"Saved metrics -> {RUN_DIR/'test_metrics.csv'}")

if __name__ == "__main__":
    main()