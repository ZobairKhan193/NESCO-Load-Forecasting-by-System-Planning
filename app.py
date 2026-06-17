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

import io
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
# Artifacts may sit either in an `artifacts/` subfolder OR right next to app.py
# (repo root on Streamlit Cloud). Use whichever actually contains config.json,
# so it works regardless of how the files were uploaded.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ARTIFACT_CANDIDATES = [os.path.join(_HERE, "artifacts"), _HERE]
ARTIFACT_DIR = next(
    (d for d in _ARTIFACT_CANDIDATES if os.path.exists(os.path.join(d, "config.json"))),
    _ARTIFACT_CANDIDATES[0],
)

LAT, LON = 24.3636, 88.6241
TZ_OFFSET_HOURS = 6  # Asia/Dhaka is UTC+6, no DST
WEATHER_VARS = ["temperature_2m", "relative_humidity_2m", "precipitation",
                "wind_speed_10m", "cloud_cover"]

# ---- OIS output format ----
# OIS expects reactive power (MVAR) and half-hourly points in the evening peak.
POWER_FACTOR = 0.9                                   # cos(phi)
MVAR_FACTOR = math.sqrt(1 - POWER_FACTOR**2) / POWER_FACTOR   # tan(arccos(0.9)) ~= 0.48432
HALF_HOUR_SLOTS = ["18:30", "19:30"]                 # extra points OIS needs at evening peak


def build_ois_table(forecast_df, target_day):
    """Convert the 24-hour forecast into the OIS upload layout:
    columns [date_time(<date>), Forecast (MW), Forecast (MVAR)], hourly rows
    00:00..23:00 plus half-hourly rows (averaged from the adjacent hours),
    MVAR = MW * tan(arccos(power factor))."""
    mw = {ts.strftime("%H:%M"): float(v)
          for ts, v in zip(forecast_df["datetime"], forecast_df["forecasted_load"])}
    # Half-hourly = average of the immediate previous and next hourly values
    for slot in HALF_HOUR_SLOTS:
        h = int(slot[:2])
        mw[slot] = (mw[f"{h:02d}:00"] + mw[f"{h+1:02d}:00"]) / 2.0
    labels = sorted(mw, key=lambda x: (int(x[:2]), int(x[3:])))
    rows = []
    for lab in labels:
        v = round(mw[lab], 2)
        rows.append((lab, v, round(v * MVAR_FACTOR, 2)))
    return pd.DataFrame(
        rows, columns=[f"date_time({target_day})", "Forecast (MW)", "Forecast (MVAR)"]
    )


def models_table(config):
    """Build the per-model accuracy table from config (robust to old/new keys)."""
    holdout = {r["model"]: r
               for r in config.get("holdout_leaderboard", config.get("leaderboard", []))}
    backtest = config.get("backtest_leaderboard", {})
    names = list(dict.fromkeys(list(backtest.keys()) + list(holdout.keys())))
    rows = []
    for n in names:
        rows.append({
            "Model": n,
            "Day-ahead MAPE (%)": round(backtest[n], 2) if n in backtest else None,
            "Holdout MAPE (%)": round(holdout[n]["MAPE"], 2) if n in holdout else None,
        })
    df = pd.DataFrame(rows)
    sort_col = ("Day-ahead MAPE (%)" if df.get("Day-ahead MAPE (%)") is not None
                and df["Day-ahead MAPE (%)"].notna().any() else "Holdout MAPE (%)")
    return df.sort_values(sort_col, na_position="last").reset_index(drop=True)


def best_model_info(config):
    """Return (best_name, best_day_ahead_mape) from config."""
    best_name = config.get("model_name", "?")
    mape = config.get("selected_backtest_mape")
    if mape is None:
        mape = config.get("backtest_leaderboard", {}).get(best_name)
    return best_name, mape

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
            f"config.json not found. Looked in: {_ARTIFACT_CANDIDATES}. "
            "Upload the 5 artifacts from Google Drive (nesco_load_forecast/) "
            "to the repo — either directly at the root next to app.py, or in "
            "an 'artifacts/' folder. Either works."
        )
    with open(cfg_path) as f:
        config = json.load(f)

    feat_scaler = joblib.load(os.path.join(ARTIFACT_DIR, "feat_scaler.pkl"))
    target_scaler = joblib.load(os.path.join(ARTIFACT_DIR, "target_scaler.pkl"))

    if config["model_kind"] in ("xgboost", "lightgbm"):
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

        if model_kind in ("xgboost", "lightgbm"):
            # Tree models were trained on RAW features — no scaling.
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
st.markdown("""
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 2rem;}
.app-banner {
    background: linear-gradient(100deg, #0b3d2e 0%, #11705a 60%, #0d8f6f 100%);
    padding: 20px 26px; border-radius: 14px; margin-bottom: 14px;
    border: 1px solid rgba(255,255,255,0.10);
    box-shadow: 0 4px 18px rgba(0,0,0,0.25);
}
.app-banner h1 {margin: 0; font-size: 30px; color: #ffffff; font-weight: 800;
    letter-spacing: .3px;}
.app-banner p {margin: 6px 0 0 0; color: #d6efe7; font-size: 13.5px;}
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 12px; padding: 12px 16px;
}
.best-badge {display:inline-block; background:#0d8f6f; color:#fff;
    padding:3px 12px; border-radius:14px; font-size:13px; font-weight:700;}
.app-footer {text-align:center; color:#8595a1; font-size:12px;
    margin-top: 34px; padding-top: 14px;
    border-top: 1px solid rgba(255,255,255,0.08);}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-banner">
  <h1>⚡ Load Forecast by Planning</h1>
  <p>NESCO System Planning · Rajshahi Zone — 24-hour hourly load forecast &nbsp;|&nbsp;
     powered by Open-Meteo weather + Machine Learning</p>
</div>
""", unsafe_allow_html=True)

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

    _best_name, _best_mape = best_model_info(config)
    _mape_txt = f"{_best_mape:.2f}% MAPE" if _best_mape is not None else "n/a"
    st.subheader("🏆 Best model")
    st.markdown(f"<span class='best-badge'>{_best_name}</span> &nbsp; "
                f"<b>{_mape_txt}</b>", unsafe_allow_html=True)
    st.caption("Day-ahead accuracy, chosen by walk-forward backtest.")

    st.divider()
    st.subheader("ℹ️ Details")
    st.markdown(
        f"""
- **Trained:** {config.get('trained_at', 'unknown')[:10]}
- **Data through:** {config.get('data_range', {}).get('end', 'unknown')[:10]}
- **Last actual:** {last_actual:%Y-%m-%d %H:%M}
"""
    )

    with st.expander("📊 Models tested (accuracy)"):
        st.dataframe(models_table(config), hide_index=True, use_container_width=True)
        st.caption("Lower MAPE = better. *Day-ahead* = realistic iterative error; "
                   "*Holdout* = optimistic teacher-forced error.")

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

        # ---- OIS table + download (ready to upload to OIS) ----
        ois_df = build_ois_table(forecast_df, target_day)

        # Build an .xlsx in memory matching the OIS layout (sheet name "forecast")
        xbuf = io.BytesIO()
        with pd.ExcelWriter(xbuf, engine="openpyxl") as writer:
            ois_df.to_excel(writer, index=False, sheet_name="forecast")
        xbuf.seek(0)

        tcol, dcol = st.columns([2, 1])
        with tcol:
            st.dataframe(ois_df, use_container_width=True, height=460)
        with dcol:
            st.download_button(
                "⬇️ Download OIS file (.xlsx)",
                data=xbuf.getvalue(),
                file_name=f"NESCO-Raj_forecast_{target_day}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
            st.download_button(
                "⬇️ Plain CSV (hourly only)",
                data=ois_df.to_csv(index=False).encode("utf-8"),
                file_name=f"forecast_{target_day}.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.caption(f"MVAR at power factor {POWER_FACTOR} · "
                       f"half-hourly = average of adjacent hours")

    except Exception as e:
        st.error(f"Forecast failed: {e}")
        st.exception(e)
else:
    st.info(
        "👈 Pick a date and click **Generate Forecast** to produce the "
        "OIS-ready 24-hour forecast (MW + MVAR, with evening half-hourly points)."
    )
    _bn, _bm = best_model_info(config)
    _bm_txt = f"{_bm:.2f}% MAPE" if _bm is not None else "n/a"
    c1, c2 = st.columns([1, 1.3])
    with c1:
        st.markdown("#### 🏆 Selected model")
        st.markdown(
            f"<span class='best-badge'>{_bn}</span> &nbsp; "
            f"day-ahead accuracy ≈ <b>{_bm_txt}</b>", unsafe_allow_html=True)
        st.caption(
            "Chosen automatically from the models below by a walk-forward "
            "day-ahead backtest over the last 14 days (lower MAPE = better)."
        )
    with c2:
        st.markdown("#### 📊 Models evaluated")
        _mt = models_table(config)
        _sty = _mt.style.apply(
            lambda r: ['background-color: rgba(13,143,111,0.30)'
                       if r["Model"] == _bn else '' for _ in r], axis=1
        ).format({"Day-ahead MAPE (%)": "{:.2f}", "Holdout MAPE (%)": "{:.2f}"},
                 na_rep="—")
        st.dataframe(_sty, hide_index=True, use_container_width=True)

# ---- Copyright footer ----
st.markdown(
    f"<div class='app-footer'>© {_today.year} <b>Zobair Hossain Khan</b> · "
    f"Load Forecast by Planning — NESCO System Planning</div>",
    unsafe_allow_html=True,
)
