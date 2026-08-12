"""
Dengue Forecasting Dashboard — Continuous Forecasting (GA-SVR)
================================================================
Streamlit app for decision makers. Loads production models trained by
train_ga_svr_dashboard.py, lets the user upload/append daily data, and
regenerates the forecast (7-day and 28-day horizons) using the latest
autoregressive + weather features.

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

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

import json
from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from features import TARGET, add_features, build_latest_feature_row, load_data

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
st.sidebar.subheader("1. Update data")

data_mode = st.sidebar.radio(
    "How do you want to update the dataset?",
    ["Upload a CSV (replace/merge)", "Add today's row manually"],
    label_visibility="collapsed",
)

live_df = get_live_data()

if data_mode == "Upload a CSV (replace/merge)":
    uploaded = st.sidebar.file_uploader("CSV with Date + confirm_dengue + weather cols", type=["csv"])
    merge_mode = st.sidebar.radio("If data already loaded:", ["Replace entirely", "Merge (append/overwrite by date)"])
    if uploaded is not None:
        try:
            new_df = load_data(uploaded)
            if live_df.empty or merge_mode == "Replace entirely":
                live_df = new_df
            else:
                live_df = pd.concat([live_df, new_df])
                live_df = live_df[~live_df.index.duplicated(keep="last")].sort_index()
            save_live_data(live_df)
            st.sidebar.success(f"Loaded {len(new_df)} rows. Dataset now has {len(live_df)} rows.")
        except Exception as e:
            st.sidebar.error(f"Could not load CSV: {e}")

else:
    st.sidebar.caption("Enter today's confirmed cases and weather readings.")
    with st.sidebar.form("manual_entry"):
        entry_date = st.date_input("Date", value=pd.Timestamp.today().normalize())
        confirm_dengue = st.number_input("Confirmed dengue cases", min_value=0, step=1)
        c1, c2 = st.columns(2)
        with c1:
            ta = st.number_input("Avg temp (ta)", value=0.0, format="%.2f")
            rha = st.number_input("Rel. humidity (rha)", value=0.0, format="%.2f")
            ra = st.number_input("Rainfall (ra)", value=0.0, format="%.2f")
            max_ta = st.number_input("Max temp (max_ta)", value=0.0, format="%.2f")
        with c2:
            min_ta = st.number_input("Min temp (min_ta)", value=0.0, format="%.2f")
            ws = st.number_input("Wind speed (ws)", value=0.0, format="%.2f")
            sr = st.number_input("Solar radiation (sr)", value=0.0, format="%.2f")
        submitted = st.form_submit_button("Add / update row")

    if submitted:
        new_row = {
            "date": entry_date, "confirm_dengue": confirm_dengue,
            "ta": ta, "rha": rha, "ra": ra, "max_ta": max_ta,
            "min_ta": min_ta, "ws": ws, "sr": sr,
        }
        live_df = append_new_row(live_df if not live_df.empty else pd.DataFrame(), new_row)
        save_live_data(live_df)
        st.sidebar.success(f"Row for {entry_date} saved. Dataset now has {len(live_df)} rows.")

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
    st.info("👈 Upload a CSV or add today's row in the sidebar to get started.")
    st.stop()

tab_overview, tab_forecast, tab_data, tab_model = st.tabs(
    ["📊 Overview", "🔮 Forecast", "📁 Data", "⚙️ Model info"]
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
    window = st.select_slider("Show last N days", options=[30, 60, 90, 180, 365, len(live_df)], value=min(90, len(live_df)))
    recent = live_df.tail(window)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recent.index, y=recent[TARGET], mode="lines", name="Confirmed cases", line=dict(width=2)))
    if len(recent) >= 7:
        fig.add_trace(go.Scatter(
            x=recent.index, y=recent[TARGET].rolling(7).mean(),
            mode="lines", name="7-day rolling mean", line=dict(dash="dash"),
        ))
    fig.update_layout(height=400, margin=dict(t=20, b=20), hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    weather_present = [c for c in ["ta", "rha", "ra", "max_ta", "min_ta", "ws", "sr"] if c in live_df.columns]
    if weather_present:
        st.subheader("Weather conditions")
        wsel = st.multiselect("Show variables", weather_present, default=weather_present[:2])
        if wsel:
            wfig = go.Figure()
            for w in wsel:
                wfig.add_trace(go.Scatter(x=recent.index, y=recent[w], mode="lines", name=w))
            wfig.update_layout(height=320, margin=dict(t=20, b=20), hovermode="x unified")
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

# ---- DATA TAB ---------------------------------------------------------
with tab_data:
    st.subheader("Current dataset")
    st.dataframe(live_df.reset_index(), use_container_width=True, height=450)
    st.caption(
        "This is the dataset the forecast is generated from. Add new rows via the sidebar "
        "(upload or manual entry) — the forecast tab always uses the most recent row as the "
        "starting point for the autoregressive features."
    )

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
