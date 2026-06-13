"""
Rajshahi Hourly Load Forecast — Streamlit web app

Anyone opens the URL, picks a date from the calendar (today .. today+2 in
Asia/Dhaka), clicks "Generate Forecast", and gets a 24-hour hourly load
forecast: chart + table + downloadable CSV.

Deployment: Streamlit Community Cloud (free), from a GitHub repo containing:
    app.py
    requirements.txt
    artifacts/
        config.json
        best_model.pkl        (or best_model.keras if a DL model won)
        feat_scaler.pkl
        target_scaler.pkl
        history_tail.csv

After each weekly retrain in Colab, download the new artifacts from
Google Drive (MyDrive/nesco_load_forecast/) and replace the files in
artifacts/ on GitHub. Streamlit Cloud redeploys automatically on commit.

The forecasting logic is identical to forecast.ipynb:
  - seasonal-naive bridge between last actual and the target day
  - iterative hour-by-hour prediction (prediction committed to history
    before the next hour's features are computed)
  - XGBoost gets RAW features; DL models get scaled features
  - bias correction from config.json added to every prediction
"""

import json
import math
import os
from datetime import datetime, timedelta, date, timezone

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

# ============================================================
# Configuration
# ============================================================
ARTIFACT_DIR = ARTIFACT_DIR = os.path.dirname(os.path.abspath(__file__))

LAT, LON = 24.3636, 88.6241
TZ_OFFSET_HOURS = 6  # Asia/Dhaka is UTC+6, no DST
WEATHER_VARS = ["temperature_2m", "relative_humidity_2m", "precipitation",
                "wind_speed_10m", "cloud_cover"]

st.set_page_config(
    page_title="Load Forecast by Planning",
    page_icon="⚡",
    layout="wide",
)


# ============================================================
# Login gate
# ============================================================
# Credentials are NOT stored in this file. They live in Streamlit's encrypted
# "Secrets" box (App settings -> Secrets on share.streamlit.io), in this format:
#
#     [users]
#     "zobair.buet@gmail.com" = "the-real-password"
#     "operator1" = "another-password"
#
# Add as many users as you like. To change/revoke access, edit Secrets and
# save — no code change, no redeploy needed.
#
# For LOCAL testing only, you can instead create a file
#     streamlit_app/.streamlit/secrets.toml
# with the same [users] block. DO NOT commit that file to GitHub (it is listed
# in .gitignore).
def check_login() -> bool:
    """Render a login form. Returns True once the user is authenticated."""
    if st.session_state.get("authenticated"):
        return True

    try:
        users = dict(st.secrets["users"])
    except Exception:
        st.error(
            "No login credentials configured. The app owner must add a "
            "`[users]` section under **App settings → Secrets** on "
            "Streamlit Cloud. See the comment at the top of app.py."
        )
        st.stop()

    st.title("⚡ Load Forecast by Planning")
    st.subheader("Please sign in")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", type="primary")

    if submitted:
        if username in users and password == str(users[username]):
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.rerun()
        else:
            st.error("Incorrect username or password.")
    return False


if not check_login():
    st.stop()


def today_dhaka() -> date:
    return (datetime.now(timezone.utc) + timedelta(hours=TZ_OFFSET_HOURS)).date()


# ============================================================
# Artifact loading (cached — survives across users/sessions)
# ============================================================
@st.cache_resource(show_spinner="Loading model artifacts ...")
def load_artifacts():
    cfg_path = os.path.join(ARTIFACT_DIR, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"config.json not found in {ARTIFACT_DIR}. "
            "Upload the artifacts from Google Drive (nesco_load_forecast/) "
            "into the artifacts/ folder of this repo."
        )
    with open(cfg_path) as f:
        config = json.load(f)

    feat_scaler = joblib.load(os.path.join(ARTIFACT_DIR, "feat_scaler.pkl"))
    target_scaler = joblib.load(os.path.join(ARTIFACT_DIR, "target_scaler.pkl"))

    if config["model_kind"] == "xgboost":
        model = joblib.load(os.path.join(ARTIFACT_DIR, "best_model.pkl"))
    else:
        # Lazy import: TensorFlow only needed when a DL model won training.
        # If this raises, add tensorflow-cpu to requirements.txt.
        import tensorflow as tf  # noqa: F401
        from tensorflow import keras
        from tensorflow.keras import layers

        class AttentionPooling(layers.Layer):
            def __init__(self, units=64, **kwargs):
                super().__init__(**kwargs)
                self.units = units

            def build(self, input_shape):
                self.W = self.add_weight(shape=(input_shape[-1], self.units),
                                         initializer="glorot_uniform", name="att_W")
                self.b = self.add_weight(shape=(self.units,),
                                         initializer="zeros", name="att_b")
                self.v = self.add_weight(shape=(self.units, 1),
                                         initializer="glorot_uniform", name="att_v")
                super().build(input_shape)

            def call(self, x):
                score = tf.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
                score = tf.tensordot(score, self.v, axes=1)
                weights = tf.nn.softmax(score, axis=1)
                return tf.reduce_sum(x * weights, axis=1)

            def get_config(self):
                cfg = super().get_config()
                cfg.update({"units": self.units})
                return cfg

        model = keras.models.load_model(
            os.path.join(ARTIFACT_DIR, "best_model.keras"),
            custom_objects={"AttentionPooling": AttentionPooling},
        )

    history = pd.read_csv(os.path.join(ARTIFACT_DIR, "history_tail.csv"),
                          parse_dates=["Time"]).sort_values("Time").reset_index(drop=True)
    if len(history) < 168:
        raise ValueError(
            f"history_tail.csv only has {len(history)} rows — need >= 168."
        )
    return config, feat_scaler, target_scaler, model, history


# ============================================================
# Weather (cached 30 min so repeated clicks don't re-hit the API)
# ============================================================
@st.cache_data(ttl=1800, show_spinner="Fetching weather forecast ...")
def fetch_weather_forecast(start_date: str, end_date: str) -> pd.DataFrame:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LAT, "longitude": LON,
        "start_date": start_date, "end_date": end_date,
        "hourly": ",".join(WEATHER_VARS),
        "timezone": "Asia/Dhaka",
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    wx = pd.DataFrame(r.json()["hourly"])
    wx["time"] = pd.to_datetime(wx["time"])
    return (wx.rename(columns={"time": "Time"})
              .sort_values("Time").reset_index(drop=True))


# ============================================================
# Feature engineering — MUST match train.ipynb exactly
# ============================================================
def build_row_features(t, history_df, weather_row, holiday_dates):
    row = {}
    row["hour"] = t.hour
    row["dow"] = t.dayofweek
    row["month"] = t.month
    row["doy"] = t.dayofyear
    row["is_weekend"] = int(t.dayofweek in (4, 5))  # Friday + Saturday
    row["hour_sin"] = math.sin(2 * math.pi * t.hour / 24)
    row["hour_cos"] = math.cos(2 * math.pi * t.hour / 24)
    row["dow_sin"] = math.sin(2 * math.pi * t.dayofweek / 7)
    row["dow_cos"] = math.cos(2 * math.pi * t.dayofweek / 7)
    row["month_sin"] = math.sin(2 * math.pi * t.month / 12)
    row["month_cos"] = math.cos(2 * math.pi * t.month / 12)
    row["doy_sin"] = math.sin(2 * math.pi * t.dayofyear / 366)
    row["doy_cos"] = math.cos(2 * math.pi * t.dayofyear / 366)
    row["is_holiday"] = int(t.date() in holiday_dates)

    for v in WEATHER_VARS:
        row[v] = float(weather_row[v])

    s = history_df["Demand"]
    for k in (1, 2, 3, 24, 48, 72, 168):
        ts = t - pd.Timedelta(hours=k)
        if ts not in s.index:
            raise KeyError(f"Missing lag-{k} value at {ts}.")
        row[f"demand_lag_{k}"] = float(s.loc[ts])

    past_6 = s.loc[(s.index >= t - pd.Timedelta(hours=6)) & (s.index < t)]
    past_12 = s.loc[(s.index >= t - pd.Timedelta(hours=12)) & (s.index < t)]
    past_24 = s.loc[(s.index >= t - pd.Timedelta(hours=24)) & (s.index < t)]
    past_168 = s.loc[(s.index >= t - pd.Timedelta(hours=168)) & (s.index < t)]
    row["demand_roll6_mean"] = float(past_6.mean())
    row["demand_roll12_mean"] = float(past_12.mean())
    row["demand_roll24_mean"] = float(past_24.mean())
    row["demand_roll24_std"] = float(past_24.std())
    row["demand_roll168_mean"] = float(past_168.mean())
    row["demand_roll168_std"] = float(past_168.std())

    for col in ("temperature_2m", "relative_humidity_2m", "precipitation"):
        ts = t - pd.Timedelta(hours=24)
        if ts in history_df.index and col in history_df.columns and \
                not pd.isna(history_df.at[ts, col]):
            row[f"{col}_lag_24"] = float(history_df.at[ts, col])
        else:
            row[f"{col}_lag_24"] = float(weather_row[col])

    row["temp_squared"] = row["temperature_2m"] ** 2
    row["temp_hour_sin"] = row["temperature_2m"] * row["hour_sin"]
    row["temp_hour_cos"] = row["temperature_2m"] * row["hour_cos"]
    row["humidex_proxy"] = row["temperature_2m"] + 0.1 * row["relative_humidity_2m"]
    return row


def build_lookback_window(t, history_df, weather_lookup, holiday_dates,
                          feature_cols, lookback):
    rows = []
    for k in range(lookback, 0, -1):
        ts = t - pd.Timedelta(hours=k)
        if ts in history_df.index and \
                all(c in history_df.columns for c in WEATHER_VARS) and \
                not history_df.loc[ts, WEATHER_VARS].isna().any():
            w = {v: history_df.at[ts, v] for v in WEATHER_VARS}
        else:
            w_row = weather_lookup.loc[weather_lookup["Time"] == ts]
            if w_row.empty:
                raise KeyError(f"No weather available at {ts}.")
            w = {v: w_row.iloc[0][v] for v in WEATHER_VARS}
        rows.append(build_row_features(ts, history_df, w, holiday_dates))
    return pd.DataFrame(rows)[feature_cols].values.astype(np.float32)


# ============================================================
# Iterative 24-hour forecast (same logic as forecast.ipynb)
# ============================================================
def run_forecast(target_day, config, feat_scaler, target_scaler, model,
                 history, weather_fc, progress_callback=None):
    feature_cols = config["feature_columns"]
    model_kind = config["model_kind"]
    lookback = config["lookback"]
    bias_corr = float(config.get("bias_correction", 0.0))
    holiday_dates = set(pd.to_datetime(list(config["holidays"].keys())).date)

    hist = history.copy().set_index("Time").sort_index()
    for v in WEATHER_VARS:
        if v not in hist.columns:
            hist[v] = np.nan

    weather_lookup = pd.concat([
        hist.reset_index()[["Time"] + WEATHER_VARS],
        weather_fc,
    ], axis=0).drop_duplicates(subset="Time", keep="last").sort_values("Time")

    start_t = pd.Timestamp(target_day.strftime("%Y-%m-%d") + " 00:00:00")
    target_hours = pd.date_range(start_t, periods=24, freq="h")

    # Seasonal-naive bridge between last actual and target day
    bridged_hours = 0
    last_hist_t = hist.index.max()
    if last_hist_t < target_hours[0] - pd.Timedelta(hours=1):
        gap_idx = pd.date_range(last_hist_t + pd.Timedelta(hours=1),
                                target_hours[0] - pd.Timedelta(hours=1), freq="h")
        gap_df = pd.DataFrame(index=gap_idx, columns=hist.columns)
        gap_df.index.name = "Time"

        how_median = hist.groupby(
            [hist.index.dayofweek, hist.index.hour])["Demand"].median()
        vals = []
        for ts in gap_idx:
            val = np.nan
            for days_back in (7, 14, 21):
                ref = ts - pd.Timedelta(days=days_back)
                if ref in hist.index and not pd.isna(hist.at[ref, "Demand"]):
                    val = float(hist.at[ref, "Demand"])
                    break
            if pd.isna(val):
                val = float(how_median.loc[(ts.dayofweek, ts.hour)])
            vals.append(val)
        gap_df["Demand"] = vals
        for v in WEATHER_VARS:
            gap_df[v] = weather_lookup.set_index("Time").reindex(gap_idx)[v].values
        hist = pd.concat([hist, gap_df]).sort_index()
        bridged_hours = len(gap_idx)

    preds = []
    for i, t in enumerate(target_hours):
        wrow = weather_fc.loc[weather_fc["Time"] == t]
        if wrow.empty:
            raise KeyError(f"No forecast weather at {t}.")
        wdict = {v: wrow.iloc[0][v] for v in WEATHER_VARS}

        if model_kind == "xgboost":
            # XGBoost was trained on RAW features — no scaling.
            row_feats = build_row_features(t, hist, wdict, holiday_dates)
            X = np.array([[row_feats[c] for c in feature_cols]], dtype=np.float32)
            yhat = float(model.predict(X)[0])
        else:
            win = build_lookback_window(t, hist, weather_lookup, holiday_dates,
                                        feature_cols, lookback)
            win_s = feat_scaler.transform(win)
            yhat_s = float(model.predict(win_s[np.newaxis, :, :], verbose=0)[0, 0])
            yhat = float(target_scaler.inverse_transform([[yhat_s]])[0, 0])

        yhat = yhat + bias_corr
        yhat = float(np.clip(yhat, 50.0, config.get("max_valid_demand", 700)))

        preds.append((t, yhat))
        # Commit prediction to working history BEFORE the next hour.
        hist.loc[t, "Demand"] = yhat
        for v in WEATHER_VARS:
            hist.loc[t, v] = wdict[v]
        hist = hist.sort_index()

        if progress_callback:
            progress_callback((i + 1) / 24)

    out = pd.DataFrame(preds, columns=["datetime", "forecasted_load"])
    return out, bridged_hours


# ============================================================
# UI
# ============================================================
st.title("⚡ Load Forecast by Planning")
st.caption("NESCO System Planning · Rajshahi Zone hourly load | powered by Open-Meteo weather + ML")

# Load artifacts up front so errors show immediately
try:
    config, feat_scaler, target_scaler, model, history = load_artifacts()
except Exception as e:
    st.error(f"Could not load model artifacts: {e}")
    st.stop()

last_actual = history["Time"].max()
_today = today_dhaka()

with st.sidebar:
    st.success(f"Signed in as **{st.session_state.get('username', '')}**")
    if st.button("Sign out", use_container_width=True):
        st.session_state.clear()
        st.rerun()
    st.divider()
    st.header("Model info")
    st.markdown(
        f"""
- **Model:** {config['model_name']}
- **Trained:** {config.get('trained_at', 'unknown')[:10]}
- **Training data through:** {config.get('data_range', {}).get('end', 'unknown')[:10]}
- **Last actual in history:** {last_actual:%Y-%m-%d %H:%M}
- **Holdout MAPE:** {min(r['MAPE'] for r in config.get('leaderboard', [{'MAPE': float('nan')}])):.2f}%
"""
    )
    staleness = (_today - last_actual.date()).days
    if staleness > 10:
        st.warning(
            f"History is {staleness} days old. Retrain with fresh data "
            "for best accuracy."
        )

col1, col2 = st.columns([1, 2])
with col1:
    # Calendar widget, limited to the allowed window (today .. today+2 Dhaka)
    target_day = st.date_input(
        "Forecast date (Asia/Dhaka)",
        value=_today + timedelta(days=1),
        min_value=_today,
        max_value=_today + timedelta(days=2),
        format="YYYY-MM-DD",
        help="Forecasts are available for today, tomorrow, and the day after "
             "tomorrow only (weather forecast reliability limit).",
    )
    go = st.button("🔮 Generate Forecast", type="primary", use_container_width=True)

if go:
    try:
        lookback = config["lookback"]
        fetch_start = (datetime.combine(target_day, datetime.min.time())
                       - timedelta(hours=lookback + 12)).date()
        weather_fc = fetch_weather_forecast(str(fetch_start), str(target_day))

        pbar = st.progress(0.0, text="Forecasting hour by hour ...")
        forecast_df, bridged = run_forecast(
            target_day, config, feat_scaler, target_scaler, model,
            history, weather_fc,
            progress_callback=lambda p: pbar.progress(p, text=f"Forecasting ... {int(p*24)}/24 hours"),
        )
        pbar.empty()

        if bridged > 0:
            st.info(
                f"Note: {bridged} hours between the last actual reading "
                f"({last_actual:%Y-%m-%d %H:%M}) and the forecast day were "
                "estimated with seasonal-naive fill. Accuracy improves when "
                "the model is retrained with recent data."
            )

        # ---- Summary metrics ----
        peak_row = forecast_df.loc[forecast_df["forecasted_load"].idxmax()]
        min_row = forecast_df.loc[forecast_df["forecasted_load"].idxmin()]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Peak load", f"{peak_row['forecasted_load']:.1f} MW",
                  f"at {peak_row['datetime']:%H:%M}")
        m2.metric("Minimum load", f"{min_row['forecasted_load']:.1f} MW",
                  f"at {min_row['datetime']:%H:%M}")
        m3.metric("Average load", f"{forecast_df['forecasted_load'].mean():.1f} MW")
        m4.metric("Daily energy", f"{forecast_df['forecasted_load'].sum():.0f} MWh")

        # ---- Chart ----
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(forecast_df["datetime"], forecast_df["forecasted_load"],
                marker="o", linewidth=1.8, color="#1f77b4")
        ax.fill_between(forecast_df["datetime"], forecast_df["forecasted_load"],
                        alpha=0.15, color="#1f77b4")
        ax.set_title(f"Rajshahi hourly load forecast — {target_day}")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Forecasted load (MW)")
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()
        st.pyplot(fig)

        # ---- Table + download ----
        show_df = forecast_df.copy()
        show_df["datetime"] = show_df["datetime"].dt.strftime("%Y-%m-%d %H:%M")
        show_df["forecasted_load"] = show_df["forecasted_load"].round(2)

        tcol, dcol = st.columns([2, 1])
        with tcol:
            st.dataframe(show_df, use_container_width=True, height=420)
        with dcol:
            st.download_button(
                "⬇️ Download CSV",
                data=show_df.to_csv(index=False).encode("utf-8"),
                file_name=f"forecast_{target_day}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    except Exception as e:
        st.error(f"Forecast failed: {e}")
        st.exception(e)
else:
    st.markdown(
        "👈 Pick a date and click **Generate Forecast** to produce the "
        "24-hour hourly load forecast."
    )
