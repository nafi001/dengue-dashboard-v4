"""
Dengue Forecasting Dashboard — Continuous Forecasting (GA-SVR)
================================================================
Streamlit app for decision makers. Loads production models trained by
train_ga_svr_dashboard.py, lets the user add daily data via input boxes
(with a one-click historical-average autofill for weather), and
regenerates the 7-day / 14-day forecast using the latest features.

Directory layout expected (relative to this file):
    app.py
    features.py
    dashboard_artifacts/
        production_model_h7.pkl
        production_model_h14.pkl
        feature_columns_h7.json
        feature_columns_h14.json
        residual_std_h7.csv
        residual_std_h14.csv
        ga_best_params_h7.json
        ga_best_params_h14.json
        model_metadata.json
    data/
        current_data.csv      <- the "live" dataset the app reads/appends to
"""

import json
from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from features import (
    TARGET,
    WEATHER_COLS_CANDIDATES,
    WEATHER_LABELS,
    add_features,
    autofill_weather_for_date,
    build_latest_feature_row,
    load_data,
)

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
APP_DIR = Path(__file__).parent
ARTIFACT_DIR = APP_DIR / "dashboard_artifacts"
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
LIVE_DATA_PATH = DATA_DIR / "current_data.csv"

HORIZONS = [7, 14]

# z-scores for selectable confidence intervals (normal approximation on residual std)
CONF_Z_LOOKUP = {80: 1.282, 90: 1.645, 95: 1.960, 99: 2.576}
DEFAULT_CONF_LEVEL = 90

st.set_page_config(
    page_title="Dengue Forecasting Dashboard",
    page_icon="🦟",
    layout="wide",
)


# --------------------------------------------------------------------------
# CACHED LOADERS — artifacts don't change during a session unless retrained
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(horizon: int):
    path = ARTIFACT_DIR / f"production_model_h{horizon}.pkl"
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def load_feature_cols(horizon: int):
    path = ARTIFACT_DIR / f"feature_columns_h{horizon}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_residual_std(horizon: int):
    path = ARTIFACT_DIR / f"residual_std_h{horizon}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)["residual_std"].values


@st.cache_data(show_spinner=False)
def load_metadata():
    path = ARTIFACT_DIR / "model_metadata.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_step_metrics(horizon: int):
    """Reads step-level test metrics (step, mae, rmse, r2) saved by the
    training script as step_metrics_h{H}.csv, if present."""
    path = ARTIFACT_DIR / f"step_metrics_h{horizon}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def artifacts_available() -> bool:
    """The 7-day model is the minimum requirement for any forecast at all.
    The 14-day model is optional — generate_blended_forecast() falls back
    to a 7-day-only forecast with a warning if it's missing."""
    return (ARTIFACT_DIR / "production_model_h7.pkl").exists()


# --------------------------------------------------------------------------
# LIVE DATA HANDLING
# --------------------------------------------------------------------------
def get_live_data() -> pd.DataFrame:
    """The dataset the app treats as 'current'. Falls back to session_state
    so uploads/appends persist across reruns within a session."""
    if "live_df" in st.session_state:
        return st.session_state["live_df"]
    if LIVE_DATA_PATH.exists():
        df = load_data(LIVE_DATA_PATH)
        st.session_state["live_df"] = df
        return df
    return pd.DataFrame()


def save_live_data(df: pd.DataFrame):
    st.session_state["live_df"] = df
    df.reset_index().to_csv(LIVE_DATA_PATH, index=False)


def append_new_row(df: pd.DataFrame, new_row: dict) -> pd.DataFrame:
    new_date = pd.to_datetime(new_row["date"])
    row_df = pd.DataFrame([new_row])
    row_df["date"] = pd.to_datetime(row_df["date"])
    row_df = row_df.set_index("date")
    row_df.columns = [c.strip().lower() for c in row_df.columns]

    # Auto-derive any calendar columns the original CSV carried (year/month/day)
    # so manually-added rows don't end up with blanks in columns the model
    # doesn't use but that were present in the source data.
    for cal_col, val in [("year", new_date.year), ("month", new_date.month), ("day", new_date.day)]:
        if cal_col in df.columns:
            row_df[cal_col] = val

    updated = df.copy()
    if new_date in updated.index:
        updated.loc[new_date] = row_df.loc[new_date]
    else:
        updated = pd.concat([updated, row_df])
    updated = updated.sort_index()
    return updated


# --------------------------------------------------------------------------
# FORECASTING
# --------------------------------------------------------------------------
def generate_forecast(df_raw: pd.DataFrame, horizon: int, conf_z: float = CONF_Z_LOOKUP[DEFAULT_CONF_LEVEL]):
    model = load_model(horizon)
    feature_cols = load_feature_cols(horizon)
    residual_std = load_residual_std(horizon)

    if model is None or feature_cols is None:
        return None

    row, last_date, missing = build_latest_feature_row(df_raw, feature_cols)

    if row.isna().any(axis=None):
        na_cols = row.columns[row.isna().any()].tolist()
        raise ValueError(
            f"Cannot forecast H={horizon}: the latest row is missing values for "
            f"{na_cols[:6]}{'...' if len(na_cols) > 6 else ''}. "
            "This usually means there's a gap in recent daily data, or a weather "
            "column the model was trained on is absent from the current dataset."
        )

    X = row.values
    y_pred = model.predict(X)[0]  # shape (horizon,)
    y_pred = np.clip(y_pred, 0, None)

    future_dates = [last_date + timedelta(days=i) for i in range(1, horizon + 1)]

    lower, upper = None, None
    if residual_std is not None and len(residual_std) == horizon:
        lower = np.clip(y_pred - conf_z * residual_std, 0, None)
        upper = y_pred + conf_z * residual_std

    return {
        "dates": future_dates,
        "pred": y_pred,
        "lower": lower,
        "upper": upper,
        "as_of": last_date,
        "missing_weather_cols": missing,
    }


def generate_blended_forecast(df_raw: pd.DataFrame, conf_z: float = CONF_Z_LOOKUP[DEFAULT_CONF_LEVEL]):
    """
    Produces one consistent 14-day forecast instead of two disagreeing ones.

    Decision makers get confused when the 7-day model says day 3 is 560 but
    the 14-day model says 682 for that same day — both are separately-trained
    SVRs and will never agree exactly. Since the 7-day model is trained
    specifically for near-term accuracy, we always use ITS predictions for
    days 1-7, and only fall back to the 14-day model for days 8-14 (which
    the 7-day model can't produce at all). The result is a single number
    per date, everywhere.
    """
    res7 = generate_forecast(df_raw, 7, conf_z=conf_z)
    res14 = generate_forecast(df_raw, 14, conf_z=conf_z) if 14 in HORIZONS else None

    if res7 is None:
        return None

    dates = list(res7["dates"])
    pred = list(res7["pred"])
    lower = list(res7["lower"]) if res7["lower"] is not None else None
    upper = list(res7["upper"]) if res7["upper"] is not None else None
    source = ["7-day model"] * len(dates)

    warnings = []
    if res7["missing_weather_cols"]:
        warnings.append(f"7-day model: missing {res7['missing_weather_cols'][:6]}")

    if res14 is not None:
        # Append only steps 8-14 from the 14-day model — steps 1-7 are
        # discarded even though the model produced them, specifically to
        # avoid ever showing two different numbers for the same date.
        extra_dates = res14["dates"][7:]
        extra_pred = list(res14["pred"][7:])
        dates += extra_dates
        pred += extra_pred
        source += ["14-day model"] * len(extra_dates)
        if lower is not None and res14["lower"] is not None:
            lower += list(res14["lower"][7:])
        elif lower is not None:
            lower += [np.nan] * len(extra_dates)
        if upper is not None and res14["upper"] is not None:
            upper += list(res14["upper"][7:])
        elif upper is not None:
            upper += [np.nan] * len(extra_dates)
        if res14["missing_weather_cols"]:
            warnings.append(f"14-day model: missing {res14['missing_weather_cols'][:6]}")
        as_of_14 = res14["as_of"]
    else:
        as_of_14 = None
        warnings.append("14-day model artifacts not found — showing 7-day forecast only.")

    return {
        "dates": dates,
        "pred": np.array(pred),
        "lower": np.array(lower) if lower is not None else None,
        "upper": np.array(upper) if upper is not None else None,
        "source": source,
        "as_of_7": res7["as_of"],
        "as_of_14": as_of_14,
        "warnings": warnings,
    }


# --------------------------------------------------------------------------
# UI — SIDEBAR
# --------------------------------------------------------------------------
st.sidebar.title("🦟 Dengue Forecast")
st.sidebar.caption("GA-SVR continuous forecasting dashboard")

if not artifacts_available():
    st.sidebar.error("Model artifacts not found in `dashboard_artifacts/`.")
else:
    meta = load_metadata()
    st.sidebar.success("Models loaded")
    if meta:
        st.sidebar.markdown(f"**Model:** {meta.get('model', 'GA-SVR')}")
        st.sidebar.markdown(f"**Last trained on data through:** {meta.get('last_training_date', '—')}")

st.sidebar.divider()
st.sidebar.subheader("Add today's data")

live_df = get_live_data()
weather_cols_present_in_data = [c for c in WEATHER_COLS_CANDIDATES if live_df.empty or c in live_df.columns] \
    if not live_df.empty else WEATHER_COLS_CANDIDATES

# Track autofilled values across the date-picker changing, via session_state
if "autofill_values" not in st.session_state:
    st.session_state["autofill_values"] = {}

entry_date = st.sidebar.date_input("Date", value=pd.Timestamp.today().normalize())

today = pd.Timestamp.today().normalize().date()
if entry_date > today:
    days_ahead = (entry_date - today).days
    st.sidebar.warning(
        f"⚠️ This date is {days_ahead} day(s) in the future. If that's not "
        "intentional, double-check the date picker before submitting."
    )
if not live_df.empty and pd.Timestamp(entry_date) in live_df.index:
    existing_val = live_df.loc[pd.Timestamp(entry_date), TARGET]
    st.sidebar.info(
        f"A row for **{entry_date}** already exists (confirmed cases: "
        f"**{int(existing_val)}**). Submitting will overwrite it."
    )

autofill_clicked = st.sidebar.button(
    "🌦️ Autofill weather (historical daily avg)",
    help="Fills the weather boxes below using the historical average for "
         "this exact calendar day (e.g. Aug 12th across all prior years) "
         "computed from the data currently loaded. If that exact day has "
         "no history yet, uses the nearest day that does.",
    use_container_width=True,
)
if autofill_clicked:
    if live_df.empty:
        st.sidebar.warning("No historical data loaded yet — nothing to average.")
    else:
        vals = autofill_weather_for_date(live_df, entry_date, WEATHER_COLS_CANDIDATES)
        if vals:
            # Write straight into each widget's own session_state key.
            # Once a number_input has a `key`, Streamlit ignores value= on
            # rerun and keeps whatever's already in session_state[key] — so
            # value= alone can never update an already-rendered box.
            for k, v in vals.items():
                st.session_state[f"input_{k}"] = round(float(v), 4)
            st.sidebar.success(
                f"Filled {len(vals)} variable(s) using {pd.Timestamp(entry_date).strftime('%b %d')}'s "
                "historical average (or nearest available day)."
            )
            st.rerun()
        else:
            st.sidebar.warning("No historical weather data available yet to average from.")

with st.sidebar.form("manual_entry"):
    confirm_dengue = st.number_input("Confirmed dengue cases", min_value=0, step=1)

    st.caption("Weather (leave 0.0 / click Autofill above to use historical daily average)")
    weather_inputs = {}
    c1, c2 = st.columns(2)
    weather_keys = list(WEATHER_LABELS.items())
    for i, (key, label) in enumerate(weather_keys):
        col = c1 if i % 2 == 0 else c2
        weather_inputs[key] = col.number_input(
            label, value=0.0, format="%.4f", key=f"input_{key}"
        )

    submitted = st.form_submit_button("Add / update row", use_container_width=True)

if submitted:
    new_row = {"date": entry_date, "confirm_dengue": confirm_dengue, **weather_inputs}
    live_df = append_new_row(live_df if not live_df.empty else pd.DataFrame(), new_row)
    save_live_data(live_df)
    for k in WEATHER_LABELS:
        st.session_state.pop(f"input_{k}", None)
    st.sidebar.success(f"Row for {entry_date} saved. Dataset now has {len(live_df)} rows.")
    st.rerun()

st.sidebar.divider()
with st.sidebar.expander("Delete a row"):
    if live_df.empty:
        st.caption("No data loaded yet.")
    else:
        delete_date = st.date_input(
            "Date to delete",
            value=live_df.index.max().date(),
            min_value=live_df.index.min().date(),
            max_value=live_df.index.max().date(),
            key="delete_date_picker",
        )
        delete_ts = pd.Timestamp(delete_date)
        if delete_ts in live_df.index:
            row_preview = live_df.loc[delete_ts]
            st.caption(
                f"Confirmed cases on {delete_date}: **{int(row_preview[TARGET])}**"
            )
            confirm_delete = st.checkbox(
                f"I understand this permanently removes the {delete_date} row",
                key="confirm_delete_checkbox",
            )
            if st.button("🗑️ Delete this row", disabled=not confirm_delete, use_container_width=True):
                live_df = live_df.drop(index=delete_ts)
                save_live_data(live_df)
                st.session_state.pop("confirm_delete_checkbox", None)
                st.success(f"Deleted row for {delete_date}.")
                st.rerun()
        else:
            st.caption(f"No row exists for {delete_date}.")

st.sidebar.divider()
with st.sidebar.expander("Bulk upload CSV instead"):
    uploaded = st.file_uploader("CSV with Date + confirm_dengue + weather cols", type=["csv"])
    merge_mode = st.radio("If data already loaded:", ["Replace entirely", "Merge (append/overwrite by date)"])
    if uploaded is not None:
        try:
            new_df = load_data(uploaded)
            if live_df.empty or merge_mode == "Replace entirely":
                live_df = new_df
            else:
                live_df = pd.concat([live_df, new_df])
                live_df = live_df[~live_df.index.duplicated(keep="last")].sort_index()
            save_live_data(live_df)
            st.success(f"Loaded {len(new_df)} rows. Dataset now has {len(live_df)} rows.")
            st.rerun()
        except Exception as e:
            st.error(f"Could not load CSV: {e}")

st.sidebar.divider()
if not live_df.empty:
    st.sidebar.metric("Rows in current dataset", len(live_df))
    st.sidebar.caption(f"Range: {live_df.index.min().date()} → {live_df.index.max().date()}")
    csv_bytes = live_df.reset_index().to_csv(index=False).encode()
    st.sidebar.download_button("Download current dataset", csv_bytes, "current_data.csv", "text/csv")

    with st.sidebar.expander("Recent data health check"):
        last14 = live_df.tail(14)
        full_range = pd.date_range(last14.index.min(), live_df.index.max(), freq="D")
        missing_dates = full_range.difference(live_df.index)
        if len(missing_dates) > 0:
            st.warning(f"{len(missing_dates)} missing date(s) in the last 14 days: "
                       + ", ".join(d.strftime('%Y-%m-%d') for d in missing_dates[:5])
                       + (" ..." if len(missing_dates) > 5 else ""))
        else:
            st.success("No missing dates in the last 14 days.")

        relevant_cols = [TARGET] + [c for c in WEATHER_COLS_CANDIDATES if c in last14.columns]
        na_cols = [c for c in relevant_cols if last14[c].isna().any()]
        zero_weather = [c for c in WEATHER_COLS_CANDIDATES
                        if c in last14.columns and (last14[c] == 0).any()]
        if na_cols:
            st.warning(f"Blank values in: {', '.join(na_cols)} — these will block the forecast "
                      "from using recent rows until filled in.")
        if zero_weather:
            st.caption(f"Zero-valued weather entries recently: {', '.join(zero_weather)} "
                      "— confirm these are real readings, not unfilled boxes.")

# --------------------------------------------------------------------------
# MAIN — TABS
# --------------------------------------------------------------------------
st.title("Dengue Forecasting Dashboard")
st.caption("Decision-support view — descriptive trends, current risk, and forward-looking forecasts.")

if live_df.empty:
    st.info("👈 Add today's data or upload a CSV in the sidebar to get started.")
    st.stop()

tab_overview, tab_corr, tab_reliability = st.tabs(
    ["📊 Overview & Forecast", "🔗 Weather correlation", "📈 Model reliability"]
)

# ---- OVERVIEW & FORECAST TAB ----------------------------------------------
with tab_overview:
    latest_val = live_df[TARGET].iloc[-1]
    latest_date = live_df.index[-1]

    # 7-day vs prior 7-day trend, as a percentage (more decision-legible than a raw delta)
    last_7 = live_df[TARGET].iloc[-7:].mean() if len(live_df) >= 7 else np.nan
    prev_7 = live_df[TARGET].iloc[-14:-7].mean() if len(live_df) >= 14 else np.nan
    if pd.notna(last_7) and pd.notna(prev_7) and prev_7 > 0:
        pct_change_7 = (last_7 - prev_7) / prev_7 * 100
    else:
        pct_change_7 = np.nan

    # 14-day cumulative burden — smooths day-to-day noise, shows recent load
    cum_14 = live_df[TARGET].iloc[-14:].sum() if len(live_df) >= 14 else live_df[TARGET].sum()

    # Same-month historical average — is this month running hot or normal for the season?
    current_month = latest_date.month
    hist_mask = (live_df.index.month == current_month) & (live_df.index.year < latest_date.year)
    hist_month_avg = live_df.loc[hist_mask, TARGET].mean() if hist_mask.any() else np.nan
    if pd.notna(hist_month_avg) and hist_month_avg > 0:
        vs_seasonal_pct = (last_7 - hist_month_avg) / hist_month_avg * 100 if pd.notna(last_7) else np.nan
    else:
        vs_seasonal_pct = np.nan

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Latest confirmed cases", int(latest_val), help=f"As of {latest_date.date()}")

    col2.metric(
        "7-day trend",
        f"{pct_change_7:+.0f}%" if pd.notna(pct_change_7) else "—",
        delta=f"{pct_change_7:+.0f}% vs prior week" if pd.notna(pct_change_7) else None,
        delta_color="inverse",  # rising cases = bad, so red for positive
        help="Average daily cases this week vs. the week before.",
    )

    col3.metric(
        "14-day cumulative cases",
        int(cum_14),
        help="Total confirmed cases over the last 14 days — smooths daily noise, shows recent burden.",
    )

    col4.metric(
        "Vs. seasonal average",
        f"{vs_seasonal_pct:+.0f}%" if pd.notna(vs_seasonal_pct) else "—",
        delta=f"{vs_seasonal_pct:+.0f}% vs typical {latest_date.strftime('%B')}" if pd.notna(vs_seasonal_pct) else None,
        delta_color="inverse",
        help=f"This week's average vs. the historical average for {latest_date.strftime('%B')} "
             "in prior years — a quick read on whether this is a normal season or an unusual spike."
             if pd.notna(vs_seasonal_pct) else
             "Needs at least one prior year of data for this month to compute.",
    )

    st.divider()

    # ---------------- FORECAST ----------------
    st.subheader("🔮 14-day forecast")

    if not artifacts_available():
        st.warning("No trained model artifacts found. Run the training script first and commit "
                    "`dashboard_artifacts/` to the repo.")
    else:
        fcol1, fcol2 = st.columns([1, 3])
        with fcol1:
            conf_level = st.selectbox(
                "Confidence interval", options=[80, 90, 95, 99],
                index=[80, 90, 95, 99].index(DEFAULT_CONF_LEVEL),
                format_func=lambda x: f"{x}%",
                help="Wider intervals (99%) are more likely to contain the true value "
                     "but give a less precise range. Based on historical test-set error, "
                     "not live-recalculated.",
            )
        conf_z = CONF_Z_LOOKUP[conf_level]

        latest_row_date = live_df.index.max().date()
        st.caption(f"Latest date in current dataset: **{latest_row_date}**")

        if st.button("🔄 Regenerate forecast now", type="primary"):
            st.session_state["run_forecast"] = True

        if st.session_state.get("run_forecast"):
            try:
                blended = generate_blended_forecast(live_df, conf_z=conf_z)
            except Exception as e:
                blended = None
                st.error(f"Could not generate forecast: {e}")

            if blended is not None:
                for w in blended["warnings"]:
                    st.warning(w)

                stale_dates = []
                if blended["as_of_7"].date() < latest_row_date:
                    stale_dates.append(f"7-day model (as of {blended['as_of_7'].date()})")
                if blended["as_of_14"] is not None and blended["as_of_14"].date() < latest_row_date:
                    stale_dates.append(f"14-day model (as of {blended['as_of_14'].date()})")
                if stale_dates:
                    st.warning(
                        f"⚠️ {' and '.join(stale_dates)} used data older than your dataset's latest "
                        f"row ({latest_row_date}). This usually means a gap or missing field in a "
                        "recent entry — check the sidebar's data health check."
                    )

                ffig = go.Figure()
                hist_tail = live_df[TARGET].tail(30)
                ffig.add_trace(go.Scatter(x=hist_tail.index, y=hist_tail.values, mode="lines", name="Recent actual"))

                # split forecast trace by source so days 1-7 vs 8-14 are visually distinguishable
                n7 = sum(1 for s in blended["source"] if s == "7-day model")
                ffig.add_trace(go.Scatter(
                    x=blended["dates"][:n7], y=blended["pred"][:n7], mode="lines+markers",
                    name="Forecast (days 1\u20137)", line=dict(color="firebrick"),
                ))
                if len(blended["dates"]) > n7:
                    # connect the two segments visually with an overlapping point
                    ffig.add_trace(go.Scatter(
                        x=blended["dates"][n7 - 1:], y=blended["pred"][n7 - 1:], mode="lines+markers",
                        name="Forecast (days 8\u201314)", line=dict(color="firebrick", dash="dot"),
                    ))
                if blended["lower"] is not None:
                    valid = ~np.isnan(blended["lower"]) & ~np.isnan(blended["upper"])
                    d_valid = [d for d, v in zip(blended["dates"], valid) if v]
                    lo_valid = blended["lower"][valid]
                    up_valid = blended["upper"][valid]
                    ffig.add_trace(go.Scatter(
                        x=d_valid + d_valid[::-1],
                        y=list(up_valid) + list(lo_valid[::-1]),
                        fill="toself", fillcolor="rgba(178,34,34,0.15)",
                        line=dict(color="rgba(255,255,255,0)"), name=f"~{conf_level}% interval", showlegend=True,
                    ))
                ffig.update_layout(height=400, margin=dict(t=20, b=20), hovermode="x unified")
                st.plotly_chart(ffig, use_container_width=True)

                fc_table = pd.DataFrame({
                    "Date": [d.date() for d in blended["dates"]],
                    "Forecast": np.round(blended["pred"], 1),
                })
                if blended["lower"] is not None:
                    fc_table[f"Lower (~{conf_level}%)"] = np.round(blended["lower"], 1)
                    fc_table[f"Upper (~{conf_level}%)"] = np.round(blended["upper"], 1)
                st.dataframe(fc_table, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download forecast CSV",
                    fc_table.to_csv(index=False).encode(),
                    f"forecast_14day_{latest_row_date}.csv",
                    "text/csv",
                )
        else:
            st.info("Click **Regenerate forecast** to run the models on the current dataset.")

    st.divider()

    # ---------------- CASE TREND ----------------
    st.subheader("Case trend")
    window_options = sorted(set([30, 60, 90, 180, 365, len(live_df)]))
    window = st.select_slider("Show last N days", options=window_options, value=min(90, len(live_df)))
    recent = live_df.tail(window)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recent.index, y=recent[TARGET], mode="lines", name="Confirmed cases", line=dict(width=2)))
    if len(recent) >= 7:
        fig.add_trace(go.Scatter(
            x=recent.index, y=recent[TARGET].rolling(7).mean(),
            mode="lines", name="7-day rolling mean", line=dict(dash="dash"),
        ))
    fig.update_layout(height=380, margin=dict(t=20, b=20), hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # ---------------- YEAR-OVER-YEAR COMPARISON ----------------
    years_present = sorted(live_df.index.year.unique())
    if len(years_present) >= 2:
        st.subheader("Year-over-year comparison")
        st.caption(
            "Each year plotted against the same Jan\u2013Dec calendar axis, so "
            "seasons line up regardless of which calendar year they fell in \u2014 "
            "makes it easy to see if this year is running above, below, or in "
            "line with previous years."
        )
        yoy_default = years_present[-3:] if len(years_present) > 3 else years_present
        yoy_years = st.multiselect("Years to compare", years_present, default=yoy_default)
        if yoy_years:
            yfig = go.Figure()
            # Map every year onto a single common reference year (a leap year,
            # so Feb 29 always has somewhere to go) so the x-axis shows real
            # calendar dates ("Jan 15", "Feb 1"...) instead of raw day-of-year
            # numbers, while still overlaying different years on one axis.
            REF_YEAR = 2020
            for yr in sorted(yoy_years):
                yr_data = live_df[live_df.index.year == yr]
                ref_dates = pd.to_datetime({
                    "year": REF_YEAR, "month": yr_data.index.month, "day": yr_data.index.day
                })
                yfig.add_trace(go.Scatter(
                    x=ref_dates, y=yr_data[TARGET].values,
                    mode="lines", name=str(yr),
                    line=dict(width=2.5 if yr == latest_date.year else 1.5),
                ))
            yfig.update_layout(
                height=400, margin=dict(t=20, b=20), hovermode="x unified",
                xaxis_title="Date (month/day)", yaxis_title="Confirmed dengue cases",
                xaxis=dict(tickformat="%b %d"),
            )
            st.plotly_chart(yfig, use_container_width=True)

        # monthly comparison table — this year vs each prior year
        st.markdown("**Monthly totals by year**")
        monthly = live_df.groupby([live_df.index.year, live_df.index.month])[TARGET].sum().unstack(level=0)
        monthly.index.name = "Month"
        monthly.columns.name = None
        monthly = monthly.rename(index={i: pd.Timestamp(2000, i, 1).strftime("%B") for i in range(1, 13)})
        st.dataframe(monthly.style.format("{:.0f}", na_rep="—"), use_container_width=True)

    # ---------------- WEATHER CONDITIONS ----------------
    weather_present = [c for c in WEATHER_COLS_CANDIDATES if c in live_df.columns]
    if weather_present:
        st.subheader("Weather conditions")
        st.caption("Each variable in its own subplot — units differ (°C, %, mm, hPa, m/s, hours, degrees), so they aren't comparable on one shared axis.")
        wsel = st.multiselect(
            "Show variables",
            weather_present,
            default=weather_present[:4],
            format_func=lambda c: WEATHER_LABELS.get(c, c),
        )
        if wsel:
            n = len(wsel)
            wfig = make_subplots(
                rows=n, cols=1, shared_xaxes=True,
                subplot_titles=[WEATHER_LABELS.get(c, c) for c in wsel],
                vertical_spacing=min(0.08, 1.0 / max(n * 2, 1)),
            )
            for i, w in enumerate(wsel, start=1):
                wfig.add_trace(
                    go.Scatter(x=recent.index, y=recent[w], mode="lines", name=WEATHER_LABELS.get(w, w),
                               showlegend=False, line=dict(width=1.8)),
                    row=i, col=1,
                )
            wfig.update_layout(height=220 * n, margin=dict(t=40, b=20), hovermode="x unified")
            st.plotly_chart(wfig, use_container_width=True)

    # ---------------- RECENT DATA TABLE ----------------
    st.subheader("Recent daily data")
    st.caption("Last 14 days — cases and weather side by side, for a quick manual scan.")
    display_cols = [TARGET] + weather_present
    recent_table = live_df[display_cols].tail(14).reset_index()
    recent_table.columns = ["Date"] + [
        "Confirmed cases" if c == TARGET else WEATHER_LABELS.get(c, c) for c in display_cols
    ]
    recent_table["Date"] = recent_table["Date"].dt.date
    st.dataframe(recent_table, use_container_width=True, hide_index=True)

# ---- WEATHER CORRELATION TAB ---------------------------------------------
with tab_corr:
    weather_present = [c for c in WEATHER_COLS_CANDIDATES if c in live_df.columns]

    if not weather_present:
        st.info("No weather columns found in the current dataset.")
    else:
        st.subheader("Lagged correlation: weather variable vs. dengue cases")
        st.caption(
            "Pick a weather variable and a lag (days earlier than the case count) to see "
            "how strongly they're related — useful for questions like 'does rainfall "
            "2 weeks ago predict today's cases?'"
        )

        cc1, cc2 = st.columns([2, 1])
        with cc1:
            corr_var = st.selectbox(
                "Weather variable", weather_present,
                format_func=lambda c: WEATHER_LABELS.get(c, c), key="corr_var",
            )
        with cc2:
            corr_lag = st.slider("Lag (days)", min_value=0, max_value=60, value=7, key="corr_lag")

        merged = pd.DataFrame({
            "weather": live_df[corr_var].shift(corr_lag),
            "dengue": live_df[TARGET],
        }).dropna()

        if len(merged) < 3:
            st.warning("Not enough overlapping data to compute a correlation at this lag.")
        else:
            r = merged["weather"].corr(merged["dengue"])
            scatter_fig = go.Figure()
            scatter_fig.add_trace(go.Scatter(
                x=merged["weather"], y=merged["dengue"], mode="markers",
                marker=dict(size=6, opacity=0.6), name="Days",
            ))
            # simple linear trend line
            if merged["weather"].std() > 0:
                coeffs = np.polyfit(merged["weather"], merged["dengue"], 1)
                xs = np.linspace(merged["weather"].min(), merged["weather"].max(), 50)
                ys = coeffs[0] * xs + coeffs[1]
                scatter_fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Linear trend", line=dict(color="firebrick", dash="dash")))

            scatter_fig.update_layout(
                height=420, margin=dict(t=30, b=20),
                xaxis_title=f"{WEATHER_LABELS.get(corr_var, corr_var)} (lag {corr_lag}d)",
                yaxis_title="Confirmed dengue cases",
                title=f"Pearson r = {r:.3f}  (n={len(merged)})",
            )
            st.plotly_chart(scatter_fig, use_container_width=True)

            # correlation across a range of lags, for context
            with st.expander("See correlation across all lags 0–60 days"):
                lag_range = range(0, 61)
                rs = []
                for lg in lag_range:
                    m = pd.DataFrame({
                        "weather": live_df[corr_var].shift(lg),
                        "dengue": live_df[TARGET],
                    }).dropna()
                    rs.append(m["weather"].corr(m["dengue"]) if len(m) >= 3 else np.nan)
                lag_fig = go.Figure()
                lag_fig.add_trace(go.Bar(x=list(lag_range), y=rs, marker_color="steelblue"))
                lag_fig.add_hline(y=0, line_color="gray", line_width=1)
                lag_fig.add_vline(x=corr_lag, line_color="firebrick", line_dash="dash")
                lag_fig.update_layout(
                    height=300, margin=dict(t=20, b=20),
                    xaxis_title="Lag (days)", yaxis_title="Correlation (r)",
                )
                st.plotly_chart(lag_fig, use_container_width=True)

        st.divider()

        st.subheader("Time series comparison")
        st.caption(
            "Select a date range and weather variable to see the two series stacked — "
            "weather on top, dengue cases below — to visually inspect how they move together over time."
        )

        min_date, max_date = live_df.index.min().date(), live_df.index.max().date()
        default_start = max(min_date, (live_df.index.max() - pd.Timedelta(days=180)).date())
        ts_var = st.selectbox(
            "Weather variable", weather_present,
            format_func=lambda c: WEATHER_LABELS.get(c, c), key="ts_var",
        )
        date_range = st.slider(
            "Date range", min_value=min_date, max_value=max_date,
            value=(default_start, max_date), key="ts_range",
        )

        window_df = live_df.loc[str(date_range[0]):str(date_range[1])]

        if window_df.empty:
            st.warning("No data in the selected range.")
        else:
            ts_fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                subplot_titles=[WEATHER_LABELS.get(ts_var, ts_var), "Confirmed dengue cases"],
                vertical_spacing=0.1,
            )
            ts_fig.add_trace(
                go.Scatter(x=window_df.index, y=window_df[ts_var], mode="lines",
                           name=WEATHER_LABELS.get(ts_var, ts_var), line=dict(color="steelblue")),
                row=1, col=1,
            )
            ts_fig.add_trace(
                go.Scatter(x=window_df.index, y=window_df[TARGET], mode="lines",
                           name="Dengue cases", line=dict(color="firebrick")),
                row=2, col=1,
            )
            ts_fig.update_layout(height=520, margin=dict(t=40, b=20), hovermode="x unified", showlegend=False)
            st.plotly_chart(ts_fig, use_container_width=True)

# ---- MODEL RELIABILITY TAB -----------------------------------------------
with tab_reliability:
    st.subheader("Forecast reliability by step-ahead")
    st.caption(
        "How accurate the model was on held-out test data at each individual "
        "day of the forecast horizon (step 1 = tomorrow, step 7 = a week out, "
        "etc.). Lower MAE/RMSE and higher R\u00b2 mean a more reliable step."
    )

    metrics_by_h = {h: load_step_metrics(h) for h in HORIZONS}
    available = {h: df for h, df in metrics_by_h.items() if df is not None}

    if not available:
        st.info(
            "No `step_metrics_h{H}.csv` files found in `dashboard_artifacts/`. "
            "These are produced by the training script's per-step evaluation "
            "loop \u2014 save one CSV per horizon with columns `step, mae, rmse, r2` "
            "and this tab will pick them up automatically."
        )
    else:
        missing = [h for h in HORIZONS if h not in available]
        if missing:
            st.warning(f"Missing step metrics for horizon(s): {missing} "
                      f"\u2014 showing what's available.")

        colors = {7: "#4C9AFF", 14: "#E5484D", 28: "#F5A623"}

        metric_tabs = st.tabs(["MAE", "RMSE", "R\u00b2"])
        metric_specs = [("mae", "Mean Absolute Error (cases)"), ("rmse", "RMSE (cases)"), ("r2", "R\u00b2")]

        for mtab, (col, ylabel) in zip(metric_tabs, metric_specs):
            with mtab:
                fig = go.Figure()
                for h, df in available.items():
                    fig.add_trace(go.Scatter(
                        x=df["step"], y=df[col], mode="lines+markers",
                        name=f"H={h}", line=dict(color=colors.get(h)),
                    ))
                if col == "r2":
                    fig.add_hline(y=0, line_color="gray", line_width=1, line_dash="dot")
                fig.update_layout(
                    height=380, margin=dict(t=20, b=20),
                    xaxis_title="Step ahead (days)", yaxis_title=ylabel,
                    hovermode="x unified",
                )
                st.plotly_chart(fig, use_container_width=True)
