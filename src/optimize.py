import os, json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args

HERE = Path(__file__).resolve().parent            # .../src
ROOT = HERE.parent                                # project root

TRAIN_CSV = ROOT / "data" / "train.csv"
VAL_CSV = ROOT / "data" / "val.csv"
RUN_DIR = ROOT / "runs" / "xgb_run_01"

RANDOM_SEED = 42
N_CALLS = 25                        # number of trials for gp_minimize
N_INITIAL_POINTS = 10               # random starts before GP takes over
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


SPACE = [
    Integer(300, 800, name="n_estimators"),
    Integer(6, 8, name="max_depth"),
    Real(0.01, 0.2, "log-uniform", name="learning_rate"),
    Real(0.7, 1.0, name="colsample_bytree"),
    Real(0.1, 5.0, "log-uniform", name="reg_lambda"),
    Real(0.1, 1.5, "log-uniform",  name="gamma")
]


def main():
    ensure_dir(RUN_DIR)
    X_tr, y_tr, X_va, y_va = _load_xy()

    # XGB fixed args that shouldn't change during search
    fixed = dict(
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    @use_named_args(SPACE)
    def objective(**hp):
        model = xgb.XGBRegressor(**fixed, **hp,
                                 early_stopping_rounds=EARLY_STOPPING_ROUNDS)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_tr, y_tr), (X_va, y_va)],
            verbose=False
        )
        
        preds = model.predict(X_va)
        rmse = float(np.sqrt(((y_va - preds) ** 2).mean()))
        return rmse

    res = gp_minimize(
        func=objective,
        dimensions=SPACE,
        n_calls=N_CALLS,
        n_initial_points=N_INITIAL_POINTS,
        random_state=RANDOM_SEED,
        verbose=True,
    )

    best = {
        "n_estimators":      int(res.x[0]),
        "max_depth":         int(res.x[1]),
        "learning_rate":     float(res.x[2]),
        "colsample_bytree":  float(res.x[3]),
        "reg_lambda":        float(res.x[4]),
        "gamma":             float(res.x[5]),
        "objective":         "reg:squarederror",
        "eval_metric":       "rmse",
        "random_state":      RANDOM_SEED,
    }

    # Save artifacts
    best_path_json = os.path.join(RUN_DIR, "best_params.json") 
    best_path_csv  = os.path.join(RUN_DIR, "best_params.csv")
    hist_path_csv  = os.path.join(RUN_DIR, "opt_history.csv")

    with open(best_path_json, "w") as f: # will be used by train.py
        json.dump(best, f, indent=2)

    # CSV versions for quick viewing
    pd.DataFrame([best]).to_csv(best_path_csv, index=False)
    pd.DataFrame({"iteration": np.arange(len(res.func_vals)), "val_rmse": res.func_vals}).to_csv(
        hist_path_csv, index=False
    )

    print("\nOptimization complete")
    print(f"Best val RMSE: {res.fun:.6f}")
    print(f"Saved best params -> {best_path_json}")
    print(f"Trials history -> {hist_path_csv}")


if __name__ == "__main__":
    main()