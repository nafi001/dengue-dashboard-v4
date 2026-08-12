"""
Dengue Forecasting Dashboard — Continuous Forecasting (GA-SVR)
================================================================
Streamlit app for decision makers. Loads production models trained by
train_ga_svr_dashboard.py, lets the user add daily data via input boxes
(with a one-click historical-average autofill for weather), and
regenerates the 7-day / 28-day forecast using the latest features.

Directory layout expected (relative to this file):
    app.py
    features.py
    dashboard_artifacts/
        production_model_h7.pkl
        production_model_h28.pkl
        feature_columns_h7.json
        feature_columns_h28.json
        residual_std_h7.csv
        residual_std_h28.csv
        ga_best_params_h7.json
        ga_best_params_h28.json
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

HORIZONS = [7, 28]
CONF_Z = 1.645  # ~90% interval using residual std (normal approx)

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
def load_ga_params(horizon: int):
    path = ARTIFACT_DIR / f"ga_best_params_h{horizon}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def artifacts_available() -> bool:
    return all((ARTIFACT_DIR / f"production_model_h{h}.pkl").exists() for h in HORIZONS)


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
def generate_forecast(df_raw: pd.DataFrame, horizon: int):
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
        lower = np.clip(y_pred - CONF_Z * residual_std, 0, None)
        upper = y_pred + CONF_Z * residual_std

    return {
        "dates": future_dates,
        "pred": y_pred,
        "lower": lower,
        "upper": upper,
        "as_of": last_date,
        "missing_weather_cols": missing,
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

autofill_clicked = st.sidebar.button(
    "🌦️ Autofill weather (historical monthly avg)",
    help="Fills the weather boxes below using the average of each variable "
         "for this calendar month, computed from the data currently loaded.",
    use_container_width=True,
)
if autofill_clicked:
    if live_df.empty:
        st.sidebar.warning("No historical data loaded yet — nothing to average.")
    else:
        vals = autofill_weather_for_date(live_df, entry_date, WEATHER_COLS_CANDIDATES)
        if vals:
            st.session_state["autofill_values"] = vals
            st.sidebar.success(f"Filled {len(vals)} variable(s) from month-{pd.Timestamp(entry_date).month} history.")
        else:
            st.sidebar.warning("No historical data for this month yet.")

with st.sidebar.form("manual_entry"):
    confirm_dengue = st.number_input("Confirmed dengue cases", min_value=0, step=1)

    st.caption("Weather (leave 0.0 / click Autofill above to use monthly history)")
    af = st.session_state.get("autofill_values", {})
    weather_inputs = {}
    c1, c2 = st.columns(2)
    weather_keys = list(WEATHER_LABELS.items())
    for i, (key, label) in enumerate(weather_keys):
        col = c1 if i % 2 == 0 else c2
        weather_inputs[key] = col.number_input(
            label, value=float(af.get(key, 0.0)), format="%.2f", key=f"input_{key}"
        )

    submitted = st.form_submit_button("Add / update row", use_container_width=True)

if submitted:
    new_row = {"date": entry_date, "confirm_dengue": confirm_dengue, **weather_inputs}
    live_df = append_new_row(live_df if not live_df.empty else pd.DataFrame(), new_row)
    save_live_data(live_df)
    st.session_state["autofill_values"] = {}
    st.sidebar.success(f"Row for {entry_date} saved. Dataset now has {len(live_df)} rows.")
    st.rerun()

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

# --------------------------------------------------------------------------
# MAIN — TABS
# --------------------------------------------------------------------------
st.title("Dengue Forecasting Dashboard")
st.caption("Decision-support view — descriptive trends, current risk, and forward-looking forecasts.")

if live_df.empty:
    st.info("👈 Add today's data or upload a CSV in the sidebar to get started.")
    st.stop()

tab_overview, tab_forecast, tab_corr, tab_model = st.tabs(
    ["📊 Overview", "🔮 Forecast", "🔗 Weather correlation", "⚙️ Model info"]
)

# ---- OVERVIEW TAB ---------------------------------------------------------
with tab_overview:
    col1, col2, col3, col4 = st.columns(4)
    latest_val = live_df[TARGET].iloc[-1]
    latest_date = live_df.index[-1]
    prev_7 = live_df[TARGET].iloc[-8:-1].mean() if len(live_df) > 8 else np.nan
    trend_7 = latest_val - prev_7 if pd.notna(prev_7) else np.nan

    col1.metric("Latest confirmed cases", int(latest_val), help=f"As of {latest_date.date()}")
    col2.metric(
        "7-day avg change",
        f"{trend_7:+.1f}" if pd.notna(trend_7) else "—",
        delta=f"{trend_7:+.1f}" if pd.notna(trend_7) else None,
    )
    col3.metric("Total in dataset", int(live_df[TARGET].sum()))
    col4.metric("Days of data", len(live_df))

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

# ---- FORECAST TAB ----------------------------------------------------------
with tab_forecast:
    if not artifacts_available():
        st.warning("No trained model artifacts found. Run the training script first and commit "
                    "`dashboard_artifacts/` to the repo.")
    else:
        st.subheader("Generate forecast from the latest data")
        st.caption(
            "Forecasts are regenerated from the most recent rows in your current dataset — "
            "lag and rolling-window features (the biggest contributors) update automatically "
            "every time you add a new day."
        )

        if st.button("🔄 Regenerate forecast now", type="primary"):
            st.session_state["run_forecast"] = True

        if st.session_state.get("run_forecast"):
            results = {}
            errors = {}
            for h in HORIZONS:
                try:
                    results[h] = generate_forecast(live_df, h)
                except Exception as e:
                    errors[h] = str(e)

            for h in HORIZONS:
                st.markdown(f"#### {h}-day horizon")
                if h in errors:
                    st.error(errors[h])
                    continue

                res = results[h]
                if res["missing_weather_cols"]:
                    st.warning(
                        "Some weather features the model expects weren't found in the current "
                        f"data columns: {res['missing_weather_cols'][:6]}. Predictions may be degraded."
                    )
                st.caption(f"Forecast generated from data as of **{res['as_of'].date()}**")

                ffig = go.Figure()
                hist_tail = live_df[TARGET].tail(30)
                ffig.add_trace(go.Scatter(x=hist_tail.index, y=hist_tail.values, mode="lines", name="Recent actual"))
                ffig.add_trace(go.Scatter(x=res["dates"], y=res["pred"], mode="lines+markers", name="Forecast", line=dict(color="firebrick")))
                if res["lower"] is not None:
                    ffig.add_trace(go.Scatter(
                        x=res["dates"] + res["dates"][::-1],
                        y=list(res["upper"]) + list(res["lower"][::-1]),
                        fill="toself", fillcolor="rgba(178,34,34,0.15)",
                        line=dict(color="rgba(255,255,255,0)"), name="~90% interval", showlegend=True,
                    ))
                ffig.update_layout(height=380, margin=dict(t=20, b=20), hovermode="x unified")
                st.plotly_chart(ffig, use_container_width=True)

                fc_table = pd.DataFrame({
                    "Date": [d.date() for d in res["dates"]],
                    "Forecast": np.round(res["pred"], 1),
                })
                if res["lower"] is not None:
                    fc_table["Lower (~90%)"] = np.round(res["lower"], 1)
                    fc_table["Upper (~90%)"] = np.round(res["upper"], 1)
                st.dataframe(fc_table, use_container_width=True, hide_index=True)
                st.download_button(
                    f"Download {h}-day forecast CSV",
                    fc_table.to_csv(index=False).encode(),
                    f"forecast_h{h}_{res['as_of'].date()}.csv",
                    "text/csv",
                    key=f"dl_{h}",
                )
        else:
            st.info("Click **Regenerate forecast** to run the models on the current dataset.")

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

# ---- MODEL INFO TAB ----------------------------------------------------
with tab_model:
    st.subheader("Model details")
    meta = load_metadata()
    if meta:
        st.json(meta)
    for h in HORIZONS:
        params = load_ga_params(h)
        if params:
            st.markdown(f"**GA-tuned SVR params, H={h}:**")
            st.json(params)
