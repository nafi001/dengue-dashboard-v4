"""
Dengue Forecasting — Fixed SVR Pipeline for Streamlit Dashboard
==============================================================
Single model family: Support Vector Regression with fixed hyperparameters
(C=100, gamma='scale', epsilon=0.05). Trained separately
for two horizons: 7 days ahead and 28 days ahead.

What this script does, in order:
  1. Load the CSV, set Date as index.
  2. Build features (lags, rolling stats, calendar, weather).
  3. For each horizon (7, 28):
       - build the supervised (X, y) dataset
       - chronological train/test split
       - fit SVR on the training set, evaluate on the test set
       - plot actual vs predicted at the requested steps
       - refit SVR on the FULL dataset (production model)
       - save every file the Streamlit dashboard needs
  4. Save a combined metadata.json describing everything that was saved.
"""

import argparse
import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
TARGET = "confirm_dengue"
HORIZONS = [7, 28]

# which forecast steps to score / plot, per horizon
STEPS_TO_REPORT = {
    7: [1, 3, 7],
    28: [1, 3, 7, 14, 21, 28],
}

TARGET_LAGS = [1, 2, 3, 7, 14]
TARGET_ROLL_WINDOWS = [3, 7, 14]

WEATHER_COLS_CANDIDATES = ["ta", "rha", "ra", "max_ta", "min_ta", "ws", "sr"]
WEATHER_LAGS = [1, 2, 3, 7]
WEATHER_ROLL_WINDOWS = [3, 7, 14]

TEST_FRACTION = 0.20
MIN_TRAIN_SIZE = 60

OUTPUT_DIR = Path("dashboard_artifacts")
OUTPUT_DIR.mkdir(exist_ok=True)
PLOTS_DIR = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


# ==========================================================================
# 1. LOAD
# ==========================================================================
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.set_index("date")
    return df


# ==========================================================================
# 2. FEATURES
# ==========================================================================
def add_features(data: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    d = data.copy()

    # --- target lags ---
    for l in TARGET_LAGS:
        d[f"lag{l}"] = d[target].shift(l)

    # --- target rolling stats (mean/std/min/max/sum) ---
    s = d[target].shift(1)
    for w in TARGET_ROLL_WINDOWS:
        d[f"roll_mean_{w}"] = s.rolling(w).mean()
        d[f"roll_std_{w}"] = s.rolling(w).std()
        d[f"roll_min_{w}"] = s.rolling(w).min()
        d[f"roll_max_{w}"] = s.rolling(w).max()
        d[f"roll_sum_{w}"] = s.rolling(w).sum()

    # --- EMAs / diffs on target ---
    d["ema_7"] = s.ewm(span=7, adjust=False).mean()
    d["ema_14"] = s.ewm(span=14, adjust=False).mean()
    d["diff_1"] = s.diff(1)
    d["diff_7"] = s.diff(7)

    # --- calendar features ---
    d["dayofweek"] = d.index.dayofweek
    d["month"] = d.index.month
    d["quarter"] = d.index.quarter
    d["is_monsoon"] = d.index.month.isin([6, 7, 8, 9, 10]).astype(int)
    d["is_friday"] = (d.index.dayofweek == 4).astype(int)

    # --- weather features: lags + rolling stats for every weather col present ---
    weather_cols_found = [c for c in WEATHER_COLS_CANDIDATES if c in d.columns]
    for wcol in weather_cols_found:
        ws = d[wcol].shift(1)
        for l in WEATHER_LAGS:
            d[f"{wcol}_lag{l}"] = d[wcol].shift(l)
        for w in WEATHER_ROLL_WINDOWS:
            d[f"{wcol}_rmean_{w}"] = ws.rolling(w).mean()
            d[f"{wcol}_rstd_{w}"] = ws.rolling(w).std()

    print(f"  Weather columns found & expanded: {weather_cols_found}")
    return d


# ==========================================================================
# 3. SUPERVISED DATASET
# ==========================================================================
def make_supervised_dataset(df_features: pd.DataFrame, horizon: int, target: str = TARGET):
    feat_df = df_features.dropna().copy()
    fcols = [c for c in feat_df.columns if c != target]
    X = feat_df[fcols].values
    y = np.array([
        feat_df[target].iloc[i: i + horizon].values
        for i in range(len(feat_df) - horizon)
    ])
    X = X[: len(y)]
    dates = feat_df.index[: len(y)]
    return dict(X=X, y=y, dates=dates, fcols=fcols)


def chrono_split(n_rows: int, test_fraction: float = TEST_FRACTION):
    split_idx = int(n_rows * (1 - test_fraction))
    tr = np.zeros(n_rows, dtype=bool)
    te = np.zeros(n_rows, dtype=bool)
    tr[:split_idx] = True
    te[split_idx:] = True
    return tr, te


def clip_preds(arr):
    return np.clip(np.asarray(arr), 0, None)


# ==========================================================================
# 4. METRICS + PLOTTING
# ==========================================================================
def eval_regression(y_true, y_pred, label=""):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    if label:
        print(f"    {label}: MAE={mae:.2f} RMSE={rmse:.2f} R2={r2:.3f}")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def plot_actual_vs_predicted(dates, y_true, y_pred, horizon, step):
    plt.figure(figsize=(12, 5))
    plt.plot(dates, y_true, label="Actual", linewidth=2)
    plt.plot(dates, y_pred, label="Predicted", linewidth=2, linestyle="--")
    plt.title(f"Fixed-SVR — H={horizon} — step {step} — Actual vs Predicted")
    plt.xlabel("Date")
    plt.ylabel("Confirmed dengue cases")
    plt.legend()
    plt.tight_layout()
    fname = PLOTS_DIR / f"Fixed_SVR_H{horizon}_S{step}_actual_vs_pred.png"
    plt.savefig(fname, dpi=130)
    plt.close()


# ==========================================================================
# 5. MAIN
# ==========================================================================
def main(csv_path: str):
    print("Loading and cleaning data...")
    df = load_data(csv_path)
    print(f"  Data range: {df.index.min().date()} -> {df.index.max().date()} ({len(df)} days)")

    print("Engineering features (target lags/rolling + expanded weather lags/rolling)...")
    df_feat = add_features(df)

    all_metrics = []
    best_overall = {}
    
    # Fixed parameters being used
    fixed_params_named = {"C": 100, "gamma": "scale", "epsilon": 0.05}

    for H in HORIZONS:
        print(f"\n{'=' * 70}\nHORIZON = {H} days\n{'=' * 70}")

        ds = make_supervised_dataset(df_feat, horizon=H)
        n = len(ds["dates"])
        if n < MIN_TRAIN_SIZE:
            print(f"  Skipping horizon={H}, not enough rows ({n})")
            continue

        tr_mask, te_mask = chrono_split(n)
        X_tr, X_te = ds["X"][tr_mask], ds["X"][te_mask]
        y_tr, y_te = ds["y"][tr_mask], ds["y"][te_mask]
        dates_te = ds["dates"][te_mask]

        # ---------------- fit on train, evaluate on test ----------------
        print("\n  Training Fixed SVR on Training split...")
        svr_model = MultiOutputRegressor(Pipeline([
            ("scaler", StandardScaler()),
            ("svr", SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.05))
        ]), n_jobs=-1)
        
        svr_model.fit(X_tr, y_tr)
        pred_te = clip_preds(svr_model.predict(X_te))

        steps = STEPS_TO_REPORT[H]
        for step in steps:
            step_idx = step - 1
            if step_idx >= H:
                continue
            yt = y_te[:, step_idx]
            yp = pred_te[:, step_idx]
            m = eval_regression(yt, yp, label=f"Fixed-SVR H={H} step={step}")
            all_metrics.append({"horizon": H, "step": step, "model": "Fixed_SVR",
                                 "mae": m["MAE"], "rmse": m["RMSE"], "r2": m["R2"]})
            plot_actual_vs_predicted(dates_te, yt, yp, H, step)

        mean_mae = np.mean([m["mae"] for m in all_metrics if m["horizon"] == H])
        print(f"  Mean test MAE across reported steps (H={H}): {mean_mae:.3f}")

        # residual std per step (from the held-out test set), for CI bands on the dashboard
        residuals = y_te - pred_te
        residual_std = residuals.std(axis=0)

        best_overall[H] = {
            "params": fixed_params_named,
            "mean_mae": float(mean_mae),
            "feature_cols": ds["fcols"],
        }

        # ---------------- refit on FULL dataset (production model) ----------------
        print(f"  Refitting Fixed SVR on full dataset for production (H={H})...")
        prod_model = MultiOutputRegressor(Pipeline([
            ("scaler", StandardScaler()),
            ("svr", SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.05))
        ]), n_jobs=-1)
        prod_model.fit(ds["X"], ds["y"])

        # ---------------- save artifacts for the dashboard ----------------
        model_path = OUTPUT_DIR / f"production_model_h{H}.pkl"
        joblib.dump(prod_model, model_path, compress=('xz', 9))
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"  Saved {model_path.name} ({size_mb:.1f} MB)")
        if size_mb > 25:
            print("    WARNING: still over 25MB. Try compress=9 or compress=('xz',3), or use Git LFS.")

        pd.DataFrame({
            "step": range(1, H + 1),
            "residual_std": residual_std,
        }).to_csv(OUTPUT_DIR / f"residual_std_h{H}.csv", index=False)

        with open(OUTPUT_DIR / f"feature_columns_h{H}.json", "w") as f:
            json.dump(ds["fcols"], f, indent=2)

        # Replaced the GA params JSON with the fixed params JSON
        with open(OUTPUT_DIR / f"model_params_h{H}.json", "w") as f:
            json.dump(fixed_params_named, f, indent=2)

        # seed row for live forecasting starting "today"
        df_feat.dropna().iloc[[-1]][ds["fcols"]].to_csv(
            OUTPUT_DIR / f"latest_feature_row_h{H}.csv", index=False
        )

    # ---------------- combined metrics + metadata ----------------
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(OUTPUT_DIR / "test_metrics_fixed_svr.csv", index=False)

    meta = {
        "target": TARGET,
        "model": "Fixed-SVR (RBF kernel, C=100, gamma=scale, eps=0.05, MultiOutputRegressor)",
        "horizons": HORIZONS,
        "steps_reported": STEPS_TO_REPORT,
        "params_by_horizon": {
            str(H): info["params"] for H, info in best_overall.items()
        },
        "mean_mae_by_horizon": {
            str(H): info["mean_mae"] for H, info in best_overall.items()
        },
        "last_training_date": str(df.index.max().date()),
        "n_rows_total": len(df),
        "files_per_horizon": [
            "production_model_h{H}.pkl",
            "residual_std_h{H}.csv",
            "feature_columns_h{H}.json",
            "model_params_h{H}.json",
            "latest_feature_row_h{H}.csv",
        ],
    }
    with open(OUTPUT_DIR / "model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print("\n=== DONE ===")
    print(f"Artifacts saved under: {OUTPUT_DIR.resolve()}")
    print(" Per horizon (7, 28): production_model_h{H}.pkl, residual_std_h{H}.csv,")
    print("   feature_columns_h{H}.json, model_params_h{H}.json, latest_feature_row_h{H}.csv")
    print(" Combined: test_metrics_fixed_svr.csv, model_metadata.json")
    print(f" Plots saved under: {PLOTS_DIR.resolve()}")
    print("\nThese files are everything a Streamlit app needs:")
    print("  - load production_model_h{H}.pkl with joblib to predict")
    print("  - use feature_columns_h{H}.json to know the exact input column order")
    print("  - use latest_feature_row_h{H}.csv as the input row to forecast from 'today'")
    print("  - use residual_std_h{H}.csv to draw confidence bands around the forecast")
    print("  - use model_metadata.json / model_params_h{H}.json to show model info on the dashboard")


if __name__ == "__main__":
    csv_path = "/content/current_data.csv"
    main(csv_path)
