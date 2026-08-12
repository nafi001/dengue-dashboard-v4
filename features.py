"""
features.py
===========
Feature engineering shared between the training script and the Streamlit
dashboard. This MUST stay in lockstep with train_ga_svr_dashboard.py —
if you change one, change the other, or the saved models will receive
inputs they weren't trained on.

Column names match the actual data source:
    Date, YEAR, MONTH, DAY, SHA, RHA, RA, PSA, WSA, WD, TA, DA,
    Max_TA, Min_TA, Confirm_Dengue
(lowercased on load, so: date, year, month, day, sha, rha, ra, psa, wsa,
 wd, ta, da, max_ta, min_ta, confirm_dengue)
"""

import warnings

import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

TARGET = "confirm_dengue"

TARGET_LAGS = [1, 2, 3, 7, 14]
TARGET_ROLL_WINDOWS = [3, 7, 14]

# Full weather variable set as it actually appears in the data.
# sha = sunshine hours, rha = relative humidity, ra = rainfall, psa = pressure,
# wsa = wind speed, wd = wind direction, ta = avg temp, da = dew point,
# max_ta / min_ta = max/min temp.
WEATHER_COLS_CANDIDATES = ["sha", "rha", "ra", "psa", "wsa", "wd", "ta", "da", "max_ta", "min_ta"]
WEATHER_LAGS = [1, 2, 3, 7]
WEATHER_ROLL_WINDOWS = [3, 7, 14]

# Human-readable labels for the UI (with units)
WEATHER_LABELS = {
    "sha": "Sunshine hours (hr)",
    "rha": "Relative humidity (%)",
    "ra": "Rainfall (mm)",
    "psa": "Surface pressure (kPa)",
    "wsa": "Wind speed (m/s)",
    "wd": "Wind direction (deg)",
    "ta": "Avg temperature (\u00b0C)",
    "da": "Dew point (\u00b0C)",
    "max_ta": "Max temperature (\u00b0C)",
    "min_ta": "Min temperature (\u00b0C)",
}


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


# Raw passthrough columns that sometimes leak into feature_cols from older
# training runs (because the training script selected "every column that
# isn't the target" rather than an explicit feature list). These carry no
# real predictive signal and, worse, are exactly the columns a live/appended
# row is most likely to arrive without — so a stale/incomplete value here
# must never block using the row for forecasting.
NON_PREDICTIVE_PASSTHROUGH_COLS = {"year", "day"}


def build_latest_feature_row(df_raw: pd.DataFrame, feature_cols: list, target: str = TARGET):
    """
    Given the FULL raw dataframe (date-indexed, including target + weather
    columns, freshly updated with today's row), rebuild features and return
    the single latest row aligned to `feature_cols` (the exact column order
    the production model was trained on).

    Returns (row_df, last_date, missing) — missing lists any requested
    feature_cols not found (e.g. if a weather column is absent from the
    current data / wasn't in this dataset when the model was trained).
    """
    df_feat = add_features(df_raw, target=target)

    # Only require the columns the model actually consumes to be non-null —
    # and even among those, skip known non-predictive passthrough columns
    # (year/day) that shouldn't gate whether a fresh row can be forecast.
    cols_to_check = [
        c for c in feature_cols
        if c in df_feat.columns and c not in NON_PREDICTIVE_PASSTHROUGH_COLS
    ]
    df_feat_clean = df_feat.dropna(subset=cols_to_check) if cols_to_check else df_feat.dropna()

    if df_feat_clean.empty:
        raise ValueError(
            "Not enough history to compute all lag/rolling features "
            "(need at least ~14+ prior days of continuous data)."
        )

    last_row = df_feat_clean.iloc[[-1]].copy()

    # Backfill non-predictive passthrough columns from the row's own date if
    # they came through as NaN (e.g. a live-appended row that never had a
    # raw year/day value). The model still expects *some* numeric value in
    # that column slot since it was part of training's column list.
    last_date_val = last_row.index[-1]
    if "year" in last_row.columns and last_row["year"].isna().any():
        last_row["year"] = last_date_val.year
    if "day" in last_row.columns and last_row["day"].isna().any():
        last_row["day"] = last_date_val.day
    last_date = df_feat_clean.index[-1]

    missing = [c for c in feature_cols if c not in last_row.columns]
    row = last_row.reindex(columns=feature_cols)  # missing -> NaN, order enforced

    return row, last_date, missing


def monthly_climatology(df_raw: pd.DataFrame, weather_cols: list) -> pd.DataFrame:
    """Historical average of each weather variable, grouped by calendar month
    (1-12), computed from all available history. Used to auto-fill a new
    day's weather entry when the user doesn't have today's reading yet."""
    cols_present = [c for c in weather_cols if c in df_raw.columns]
    if not cols_present:
        return pd.DataFrame()
    return df_raw.groupby(df_raw.index.month)[cols_present].mean()


def autofill_weather_for_date(df_raw: pd.DataFrame, target_date, weather_cols: list) -> dict:
    """Return {col: historical_monthly_avg} for the month of target_date,
    using all data currently in df_raw. Empty dict if no history exists yet
    for that month or no weather columns are present."""
    clim = monthly_climatology(df_raw, weather_cols)
    if clim.empty:
        return {}
    month = pd.Timestamp(target_date).month
    if month not in clim.index:
        return {}
    row = clim.loc[month]
    return {c: float(row[c]) for c in row.index if pd.notna(row[c])}
