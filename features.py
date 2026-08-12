"""
features.py
===========
Feature engineering shared between the training script and the Streamlit
dashboard. This MUST stay in lockstep with train_ga_svr_dashboard.py —
if you change one, change the other, or the saved models will receive
inputs they weren't trained on.
"""

import pandas as pd

TARGET = "confirm_dengue"

TARGET_LAGS = [1, 2, 3, 7, 14]
TARGET_ROLL_WINDOWS = [3, 7, 14]

WEATHER_COLS_CANDIDATES = ["ta", "rha", "ra", "max_ta", "min_ta", "ws", "sr"]
WEATHER_LAGS = [1, 2, 3, 7]
WEATHER_ROLL_WINDOWS = [3, 7, 14]


def load_data(csv_path_or_buffer, target: str = TARGET) -> pd.DataFrame:
    """Load a CSV with a 'Date' column (any case) and the target column.
    Works with a file path or an in-memory buffer (e.g. Streamlit upload)."""
    df = pd.read_csv(csv_path_or_buffer)
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" not in df.columns:
        raise ValueError("CSV must contain a 'Date' column.")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.set_index("date")
    if target not in df.columns:
        raise ValueError(f"CSV must contain the target column '{target}'.")
    return df


def add_features(data: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    """Identical logic to train_ga_svr_dashboard.py's add_features()."""
    d = data.copy()

    for l in TARGET_LAGS:
        d[f"lag{l}"] = d[target].shift(l)

    s = d[target].shift(1)
    for w in TARGET_ROLL_WINDOWS:
        d[f"roll_mean_{w}"] = s.rolling(w).mean()
        d[f"roll_std_{w}"] = s.rolling(w).std()
        d[f"roll_min_{w}"] = s.rolling(w).min()
        d[f"roll_max_{w}"] = s.rolling(w).max()
        d[f"roll_sum_{w}"] = s.rolling(w).sum()

    d["ema_7"] = s.ewm(span=7, adjust=False).mean()
    d["ema_14"] = s.ewm(span=14, adjust=False).mean()
    d["diff_1"] = s.diff(1)
    d["diff_7"] = s.diff(7)

    d["dayofweek"] = d.index.dayofweek
    d["month"] = d.index.month
    d["quarter"] = d.index.quarter
    d["is_monsoon"] = d.index.month.isin([6, 7, 8, 9, 10]).astype(int)
    d["is_friday"] = (d.index.dayofweek == 4).astype(int)

    weather_cols_found = [c for c in WEATHER_COLS_CANDIDATES if c in d.columns]
    for wcol in weather_cols_found:
        ws = d[wcol].shift(1)
        for l in WEATHER_LAGS:
            d[f"{wcol}_lag{l}"] = d[wcol].shift(l)
        for w in WEATHER_ROLL_WINDOWS:
            d[f"{wcol}_rmean_{w}"] = ws.rolling(w).mean()
            d[f"{wcol}_rstd_{w}"] = ws.rolling(w).std()

    return d


def build_latest_feature_row(df_raw: pd.DataFrame, feature_cols: list, target: str = TARGET):
    """
    Given the FULL raw dataframe (date-indexed, including target + weather
    columns, freshly updated with today's row), rebuild features and return
    the single latest row aligned to `feature_cols` (the exact column order
    the production model was trained on).

    Returns (row_df, last_date, warnings) — warnings lists any requested
    feature_cols not found (e.g. if a weather column is missing from new data).
    """
    df_feat = add_features(df_raw, target=target)
    df_feat_clean = df_feat.dropna()

    if df_feat_clean.empty:
        raise ValueError(
            "Not enough history to compute all lag/rolling features "
            "(need at least ~14+ prior days of continuous data)."
        )

    last_row = df_feat_clean.iloc[[-1]]
    last_date = df_feat_clean.index[-1]

    missing = [c for c in feature_cols if c not in last_row.columns]
    row = last_row.reindex(columns=feature_cols)  # missing -> NaN, order enforced

    return row, last_date, missing
