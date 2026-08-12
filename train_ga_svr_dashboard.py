"""
Dengue Forecasting — GA-SVR Pipeline for Streamlit Dashboard
==============================================================
Single model family: Support Vector Regression, hyperparameters
(C, gamma, epsilon) tuned with a genetic algorithm. Trained separately
for two horizons: 7 days ahead and 28 days ahead.

What this script does, in order:
  1. Load the CSV, set Date as index.
  2. Build features:
       - lags of the target (confirmed dengue cases)
       - rolling mean/std/min/max/sum of the target
       - lags AND rolling stats of every weather column found in the data
         (this is the "increase weather features" step)
       - calendar features (day of week, month, quarter, monsoon, Friday)
  3. For each horizon (7, 28):
       - build the supervised (X, y) dataset
       - chronological train/test split
       - run the GA to find the best SVR hyperparameters, print them
       - fit SVR on the training set, evaluate on the test set
       - plot actual vs predicted at the requested steps
       - refit SVR on the FULL dataset (production model)
       - save every file the Streamlit dashboard needs
  4. Save a combined metadata.json describing everything that was saved.

Run:
    pip install pandas numpy scikit-learn joblib matplotlib
    python train_ga_svr_dashboard.py --csv your_data.csv
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

# GA settings
GA_POP_SIZE = 25
GA_N_GEN = 10
SVR_BOUNDS = [(1.0, 900.0), (1e-4, 1.0), (0.001, 0.2)]  # C, gamma, epsilon

OUTPUT_DIR = Path("dashboard_artifacts")
OUTPUT_DIR.mkdir(exist_ok=True)
PLOTS_DIR = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


# ==========================================================================
# 1. LOAD
# ==========================================================================
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
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
# 3. SUPERVISED DATASET (multi-output, unscaled — SVR pipeline scales internally)
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
# 4. GENETIC ALGORITHM FOR SVR HYPERPARAMETERS
# ==========================================================================
class GeneticAlgorithmOptimiser:
    """Real-valued GA: tournament selection, single-point crossover,
    Gaussian mutation with annealed scale. Minimizes RMSE on a validation split."""

    def __init__(self, param_bounds, pop_size=25, n_gen=30,
                 crossover_prob=0.8, mutation_prob=0.15, tournament_k=3, seed=SEED):
        self.bounds = param_bounds
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.cr = crossover_prob
        self.mr = mutation_prob
        self.k = tournament_k
        self.rng = np.random.RandomState(seed)

    def _rand_individual(self):
        return [self.rng.uniform(lo, hi) for lo, hi in self.bounds]

    def _fitness(self, ind, X_tr, y_tr, X_va, y_va, model_fn):
        try:
            m = model_fn(ind)
            m.fit(X_tr, y_tr)
            p = np.nan_to_num(m.predict(X_va), nan=1e9, posinf=1e9, neginf=1e9)
            rmse = np.sqrt(mean_squared_error(np.asarray(y_va).flatten(), np.asarray(p).flatten()))
            return rmse if np.isfinite(rmse) else 1e9
        except Exception:
            return 1e9

    def _tournament(self, pop, fits):
        idx = self.rng.choice(len(pop), self.k, replace=False)
        best = idx[int(np.argmin([fits[i] for i in idx]))]
        return pop[best][:]

    def _crossover(self, p1, p2):
        if self.rng.rand() < self.cr:
            pt = self.rng.randint(1, len(p1))
            return p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]
        return p1[:], p2[:]

    def _mutate(self, ind, gen):
        scale = 1.0 - 0.7 * gen / max(self.n_gen, 1)
        for k in range(len(ind)):
            if self.rng.rand() < self.mr:
                lo, hi = self.bounds[k]
                ind[k] = float(np.clip(ind[k] + (hi - lo) * scale * self.rng.randn() * 0.2, lo, hi))
        return ind

    def optimise(self, X_tr, y_tr, X_va, y_va, model_fn, verbose=True):
        pop = [self._rand_individual() for _ in range(self.pop_size)]
        best_ind, best_fit = pop[0][:], np.inf

        for gen in range(self.n_gen):
            fits = [self._fitness(p, X_tr, y_tr, X_va, y_va, model_fn) for p in pop]
            order = np.argsort(fits)
            pop = [pop[i] for i in order]
            fits = [fits[i] for i in order]

            if fits[0] < best_fit:
                best_fit, best_ind = fits[0], pop[0][:]

            if verbose and (gen % 5 == 0 or gen == self.n_gen - 1):
                print(f"    GA gen {gen + 1}/{self.n_gen}  best RMSE so far: {best_fit:.4f}")

            new_pop = pop[:2]  # elitism
            while len(new_pop) < self.pop_size:
                p1, p2 = self._tournament(pop, fits), self._tournament(pop, fits)
                c1, c2 = self._crossover(p1, p2)
                new_pop += [self._mutate(c1, gen), self._mutate(c2, gen)]
            pop = new_pop[: self.pop_size]

        return best_ind, best_fit


def make_ga_svr(params):
    C, gamma, eps = params
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf", C=C, gamma=gamma, epsilon=eps)),
    ])


# ==========================================================================
# 5. METRICS + PLOTTING
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
    plt.title(f"GA-SVR — H={horizon} — step {step} — Actual vs Predicted")
    plt.xlabel("Date")
    plt.ylabel("Confirmed dengue cases")
    plt.legend()
    plt.tight_layout()
    fname = PLOTS_DIR / f"GA_SVR_H{horizon}_S{step}_actual_vs_pred.png"
    plt.savefig(fname, dpi=130)
    plt.close()


# ==========================================================================
# 6. MAIN
# ==========================================================================
def main(csv_path: str):
    print("Loading and cleaning data...")
    df = load_data(csv_path)
    print(f"  Data range: {df.index.min().date()} -> {df.index.max().date()} ({len(df)} days)")

    print("Engineering features (target lags/rolling + expanded weather lags/rolling)...")
    df_feat = add_features(df)

    all_metrics = []
    best_overall = {}

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

        # inner validation carve-out from the tail of train, used only by the GA
        ga_val_frac = 0.15
        ga_split = int(len(X_tr) * (1 - ga_val_frac))
        X_trg, X_vag = X_tr[:ga_split], X_tr[ga_split:]
        y_trg, y_vag = y_tr[:ga_split], y_tr[ga_split:]

        # ---------------- GA search for SVR hyperparameters ----------------
        print("\n  Running GA search for SVR hyperparameters...")
        ga = GeneticAlgorithmOptimiser(SVR_BOUNDS, pop_size=GA_POP_SIZE, n_gen=GA_N_GEN, seed=SEED)

        def svr_fn(params):
            return MultiOutputRegressor(make_ga_svr(params), n_jobs=-1)

        best_params, best_rmse = ga.optimise(X_trg, y_trg, X_vag, y_vag, svr_fn)
        best_params_named = {"C": best_params[0], "gamma": best_params[1], "epsilon": best_params[2]}
        print(f"  >>> Best SVR params for H={H}: {best_params_named}  (val RMSE={best_rmse:.4f})")

        # ---------------- fit on train, evaluate on test ----------------
        svr_model = MultiOutputRegressor(make_ga_svr(best_params), n_jobs=-1)
        svr_model.fit(X_tr, y_tr)
        pred_te = clip_preds(svr_model.predict(X_te))

        steps = STEPS_TO_REPORT[H]
        for step in steps:
            step_idx = step - 1
            if step_idx >= H:
                continue
            yt = y_te[:, step_idx]
            yp = pred_te[:, step_idx]
            m = eval_regression(yt, yp, label=f"GA-SVR H={H} step={step}")
            all_metrics.append({"horizon": H, "step": step, "model": "GA_SVR",
                                 "mae": m["MAE"], "rmse": m["RMSE"], "r2": m["R2"]})
            plot_actual_vs_predicted(dates_te, yt, yp, H, step)

        mean_mae = np.mean([m["mae"] for m in all_metrics if m["horizon"] == H])
        print(f"  Mean test MAE across reported steps (H={H}): {mean_mae:.3f}")

        # residual std per step (from the held-out test set), for CI bands on the dashboard
        residuals = y_te - pred_te
        residual_std = residuals.std(axis=0)

        best_overall[H] = {
            "params": best_params_named,
            "mean_mae": float(mean_mae),
            "feature_cols": ds["fcols"],
        }

        # ---------------- refit on FULL dataset (production model) ----------------
        print(f"  Refitting GA-SVR on full dataset for production (H={H})...")
        prod_model = MultiOutputRegressor(make_ga_svr(best_params), n_jobs=-1)
        prod_model.fit(ds["X"], ds["y"])

        # ---------------- save artifacts for the dashboard ----------------
        model_path = OUTPUT_DIR / f"production_model_h{H}.pkl"
        joblib.dump(prod_model, model_path, compress=3)
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

        with open(OUTPUT_DIR / f"ga_best_params_h{H}.json", "w") as f:
            json.dump(best_params_named, f, indent=2)

        # seed row for live forecasting starting "today"
        df_feat.dropna().iloc[[-1]][ds["fcols"]].to_csv(
            OUTPUT_DIR / f"latest_feature_row_h{H}.csv", index=False
        )

    # ---------------- combined metrics + metadata ----------------
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(OUTPUT_DIR / "test_metrics_ga_svr.csv", index=False)

    meta = {
        "target": TARGET,
        "model": "GA-SVR (RBF kernel, GA-tuned C/gamma/epsilon, MultiOutputRegressor)",
        "horizons": HORIZONS,
        "steps_reported": STEPS_TO_REPORT,
        "best_params_by_horizon": {
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
            "ga_best_params_h{H}.json",
            "latest_feature_row_h{H}.csv",
        ],
    }
    with open(OUTPUT_DIR / "model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print("\n=== DONE ===")
    print(f"Artifacts saved under: {OUTPUT_DIR.resolve()}")
    print(" Per horizon (7, 28): production_model_h{H}.pkl, residual_std_h{H}.csv,")
    print("   feature_columns_h{H}.json, ga_best_params_h{H}.json, latest_feature_row_h{H}.csv")
    print(" Combined: test_metrics_ga_svr.csv, model_metadata.json")
    print(f" Plots saved under: {PLOTS_DIR.resolve()}")
    print("\nThese files are everything a Streamlit app needs:")
    print("  - load production_model_h{H}.pkl with joblib to predict")
    print("  - use feature_columns_h{H}.json to know the exact input column order")
    print("  - use latest_feature_row_h{H}.csv as the input row to forecast from 'today'")
    print("  - use residual_std_h{H}.csv to draw confidence bands around the forecast")
    print("  - use model_metadata.json / ga_best_params_h{H}.json to show model info on the dashboard")


if __name__ == "__main__":
    csv_path = "/content/nasapower_final_avg (1).csv"
    main(csv_path)
