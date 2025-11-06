import os, json
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import dump
import xgboost as xgb
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent            # .../src
ROOT = HERE.parent                                # project root

TRAIN_CSV = ROOT / "data" / "train.csv"
VAL_CSV = ROOT / "data" / "val.csv"
RUN_DIR = ROOT / "runs" / "xgb_run_01"
BEST_HP = ROOT / "runs" / "xgb_run_01" / "best_params.json"

with open(BEST_HP) as f:
    XGB_PARAMS = json.load(f)

EARLY_STOPPING_ROUNDS = 50

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_xy(train_path=TRAIN_CSV, val_path=VAL_CSV, target="oil_consumption"):

    train = pd.read_csv(train_path)
    X_train = train.drop(columns=[target])
    y_train = train[target].astype(float)

    val = pd.read_csv(val_path)
    X_val = val.drop(columns=[target])
    y_val = val[target].astype(float)

    return X_train, y_train, X_val, y_val


def fit_model(X_train, y_train, X_val=None, y_val=None):

    model = xgb.XGBRegressor(**XGB_PARAMS, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    eval_set = [(X_train, y_train), (X_val, y_val)]
    model.fit(X_train, y_train, eval_set=eval_set, verbose=True)
    return model, model.evals_result()


def plot_history(metric = 'rmse'):

    X_train, y_train, X_val, y_val = _load_xy()
    model, res = fit_model(X_train, y_train, X_val, y_val)

    epochs = len(res['validation_0'][metric])
    x_axis = range(0, epochs)

    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, res['validation_0'][metric], label=f'Train {metric}')
    plt.plot(x_axis, res['validation_1'][metric], label=f'Validation {metric}')
    plt.xlabel("Boosting Round")
    plt.ylabel(f"{metric}")
    plt.title(f"Training vs Validation {metric}")
    plt.legend()
    plt.show()


def training_summary(metric="rmse"):

    X_train, y_train, X_val, y_val = _load_xy()
    model, res = fit_model(X_train, y_train, X_val, y_val)

    tr = res.get("validation_0", {}).get(metric, [np.nan])[-1]
    va = res.get("validation_1", {}).get(metric, [np.nan])[-1]

    print(pd.DataFrame([{"train_"+metric: tr, "val_"+metric: va}]))



def main():
    ensure_dir(RUN_DIR)

    X_train, y_train, X_val, y_val = _load_xy()
    model, evals_result = fit_model(X_train, y_train, X_val, y_val)

    # model.joblib contains all model trees, learned weights, hyperparameters etc
    # this is what will be used in evaluate.py, production deployments etc
    dump(model, os.path.join(RUN_DIR, "model.joblib"))
    # train_columns.json = the exact list of feature columns
    with open(os.path.join(RUN_DIR, "train_columns.json"), "w") as f:
        json.dump(list(X_train.columns), f, indent=2)

    # evals_result.json includes per-iteration training & validation metrics
    with open(os.path.join(RUN_DIR, "evals_result.json"), "w") as f:
        json.dump(evals_result, f, indent=2)

    print(f"Model saved -> {os.path.join(RUN_DIR,'model.joblib')}")
    print(f"Columns saved -> {os.path.join(RUN_DIR,'train_columns.json')}")

if __name__ == "__main__":
    main()