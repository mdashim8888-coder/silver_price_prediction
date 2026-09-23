"""
silver_price_app.py
-------------------
Silver Price Prediction — single-file combined application.
All modules (data_loader, feature_engineering, model_trainer, predictor)
are inlined here. The Streamlit dashboard is at the bottom.

Run with:
    streamlit run silver_price_app.py
"""

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════════════════════
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ══════════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════════
CSV_PATH   = Path(__file__).parent / "silver_price_500_dataset.csv"
MODEL_PATH = Path(__file__).parent / "models" / "silver_model.joblib"

# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADER
# ══════════════════════════════════════════════════════════════════════════════
def load_data(path=CSV_PATH):
    """Load CSV, parse dates, split training rows from the live inference row.

    Returns
    -------
    df_train : pd.DataFrame  — 499 rows with non-null Target_Next_Close
    df_live  : pd.DataFrame  — 1 row, Target_Next_Close is NaN (live prediction)
    """
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df.set_index("Date", inplace=True)

    numeric_cols = ["Open", "High", "Low", "Close", "Volume",
                    "Change_Pct", "MA7", "MA20", "RSI14", "Target_Next_Close"]
    df[numeric_cols] = df[numeric_cols].astype(float)

    df_train = df[df["Target_Next_Close"].notna()].copy()
    df_live  = df[df["Target_Next_Close"].isna()].copy()
    return df_train, df_live


# ══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
FEATURE_COLS = ["Open", "High", "Low", "Close", "Volume", "Change_Pct", "MA7", "MA20", "RSI14"]
TARGET_COL   = "Target_Next_Close"


def build_features(df):
    """Return (X, y) NumPy arrays for training/evaluation."""
    X = df[FEATURE_COLS].values.astype(np.float64)
    y = df[TARGET_COL].values.astype(np.float64)
    return X, y


def build_live_features(df_live):
    """Return a single-row feature matrix for inference."""
    return df_live[FEATURE_COLS].values.astype(np.float64)


# ══════════════════════════════════════════════════════════════════════════════
# 3. MODEL TRAINER
# ══════════════════════════════════════════════════════════════════════════════
def train_and_save(csv_path=CSV_PATH, model_path=MODEL_PATH):
    """Train XGBoost on an 80/20 chronological split, save model, return results.

    Returns
    -------
    dict: model, metrics {mae, rmse, r2}, X_test, y_test, y_pred, split_idx, test_dates
    """
    df_train, _ = load_data(csv_path)
    X, y = build_features(df_train)

    # Chronological 80/20 split — no shuffling (critical for time series)
    split      = int(len(X) * 0.80)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    test_dates = df_train.index[split:]

    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2   = r2_score(y_test, y_pred)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    return {
        "model":      model,
        "metrics":    {"mae": mae, "rmse": rmse, "r2": r2},
        "X_test":     X_test,
        "y_test":     y_test,
        "y_pred":     y_pred,
        "split_idx":  split,
        "test_dates": test_dates,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
def predict_next_close(csv_path=CSV_PATH, model_path=MODEL_PATH) -> float:
    """Load the saved model and return the predicted next-day close price."""
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model not found at '{model_path}'. "
            "The app will train and save it on first run."
        )
    model  = joblib.load(model_path)
    _, df_live = load_data(csv_path)
    X_live = build_live_features(df_live)
    return float(model.predict(X_live)[0])


# ══════════════════════════════════════════════════════════════════════════════
# 5. STREAMLIT DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Silver Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .prediction-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        border-radius: 12px;
        padding: 28px 32px;
        text-align: center;
        color: white;
        margin: 8px 0;
    }
    .prediction-card h2 { font-size: 2.4rem; margin: 0; font-weight: 700; }
    .prediction-card p  { margin: 4px 0 0; opacity: 0.85; font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)


# ── Cached model training & data loading ─────────────────────────────────────
@st.cache_resource(show_spinner="Training model...")
def get_model_result():
    return train_and_save()


@st.cache_data
def get_data():
    return load_data()


result             = get_model_result()
df_train, df_live  = get_data()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🥈 Silver Predictor")
    st.markdown("---")

    min_date = df_train.index.min().date()
    max_date = df_train.index.max().date()

    date_range = st.date_input(
        "📅 Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    st.markdown("---")
    st.subheader("📊 Chart Overlays")
    show_ma7    = st.checkbox("Show MA7",    value=True)
    show_ma20   = st.checkbox("Show MA20",   value=True)
    show_volume = st.checkbox("Show Volume", value=True)
    show_rsi    = st.checkbox("Show RSI14",  value=True)

    st.markdown("---")
    st.caption("Model: XGBoost Regressor\nSplit: 80 / 20 chronological")


# ── Date filter ───────────────────────────────────────────────────────────────
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_date = pd.Timestamp(date_range[0])
    end_date   = pd.Timestamp(date_range[1])
else:
    start_date = df_train.index.min()
    end_date   = df_train.index.max()

df_filtered = df_train.loc[start_date:end_date]


# ── Header ────────────────────────────────────────────────────────────────────
st.title("📈 Silver Price Prediction Dashboard")
st.caption(
    f"Dataset: {len(df_train)} trading days  ·  "
    f"{df_train.index.min().date()} to {df_train.index.max().date()}"
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Rows",    f"{len(df_train):,}")
c2.metric("Min Close",     f"{df_train['Close'].min():,.0f}")
c3.metric("Max Close",     f"{df_train['Close'].max():,.0f}")
c4.metric("Avg Volume",    f"{df_train['Volume'].mean():,.0f}")
c5.metric("Avg Daily Chg", f"{df_train['Change_Pct'].mean():.2f}%")

st.markdown("---")


# ── Live Prediction Card ───────────────────────────────────────────────────────
pred_price      = predict_next_close()
last_close      = float(df_live["Close"].iloc[0])
change_vs_last  = pred_price - last_close
pct_change      = (change_vs_last / last_close) * 100
arrow           = "▲" if change_vs_last >= 0 else "▼"
color           = "#4ade80" if change_vs_last >= 0 else "#f87171"

col_pred, col_ctx = st.columns([1, 2])
with col_pred:
    st.markdown(f"""
    <div class="prediction-card">
        <p>Predicted Close — 2026-09-24</p>
        <h2>{pred_price:,.2f}</h2>
        <p style="color:{color}; font-size:1.05rem; font-weight:600;">
            {arrow} {abs(change_vs_last):,.2f} ({pct_change:+.2f}%) vs last close
        </p>
        <p style="opacity:0.7; font-size:0.82rem;">
            Last known close (2026-09-23): {last_close:,.2f}
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_ctx:
    m = result["metrics"]
    st.markdown("#### Model Performance")
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("MAE",  f"{m['mae']:,.2f}")
    mc2.metric("RMSE", f"{m['rmse']:,.2f}")
    mc3.metric("R²",   f"{m['r2']:.4f}")
    st.info(
        f"Trained on **{result['split_idx']} rows**, tested on "
        f"**{len(result['y_test'])} rows** (chronological 80/20 split).",
        icon="ℹ️",
    )

st.markdown("---")


# ── Candlestick Chart ──────────────────────────────────────────────────────────
st.subheader("🕯️ Price History (OHLC)")

fig_candle = go.Figure()
fig_candle.add_trace(go.Candlestick(
    x=df_filtered.index,
    open=df_filtered["Open"],
    high=df_filtered["High"],
    low=df_filtered["Low"],
    close=df_filtered["Close"],
    name="OHLC",
    increasing_line_color="#4ade80",
    decreasing_line_color="#f87171",
))
if show_ma7:
    fig_candle.add_trace(go.Scatter(
        x=df_filtered.index, y=df_filtered["MA7"],
        mode="lines", name="MA7",
        line=dict(color="#60a5fa", width=1.5),
    ))
if show_ma20:
    fig_candle.add_trace(go.Scatter(
        x=df_filtered.index, y=df_filtered["MA20"],
        mode="lines", name="MA20",
        line=dict(color="#f59e0b", width=1.5),
    ))
fig_candle.update_layout(
    xaxis_rangeslider_visible=False,
    height=460,
    margin=dict(l=0, r=0, t=10, b=0),
    paper_bgcolor="white",
    plot_bgcolor="#f7f8fa",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    yaxis_title="Price",
)
st.plotly_chart(fig_candle, use_container_width=True)


# ── RSI & Volume ──────────────────────────────────────────────────────────────
if show_rsi or show_volume:
    rows           = sum([show_rsi, show_volume])
    row_heights    = []
    subplot_titles = []
    if show_rsi:
        row_heights.append(0.5)
        subplot_titles.append("RSI 14")
    if show_volume:
        row_heights.append(0.5)
        subplot_titles.append("Volume")

    fig_sub = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        subplot_titles=subplot_titles,
        row_heights=row_heights,
        vertical_spacing=0.08,
    )

    current_row = 1
    if show_rsi:
        fig_sub.add_trace(go.Scatter(
            x=df_filtered.index, y=df_filtered["RSI14"],
            mode="lines", name="RSI14",
            line=dict(color="#a78bfa", width=2),
        ), row=current_row, col=1)
        fig_sub.add_trace(go.Scatter(
            x=[df_filtered.index.min(), df_filtered.index.max()],
            y=[70, 70], mode="lines", name="Overbought 70",
            line=dict(color="#f87171", width=1, dash="dash"),
            showlegend=False,
        ), row=current_row, col=1)
        fig_sub.add_trace(go.Scatter(
            x=[df_filtered.index.min(), df_filtered.index.max()],
            y=[30, 30], mode="lines", name="Oversold 30",
            line=dict(color="#4ade80", width=1, dash="dash"),
            showlegend=False,
        ), row=current_row, col=1)
        current_row += 1

    if show_volume:
        colors = [
            "#4ade80" if c >= o else "#f87171"
            for c, o in zip(df_filtered["Close"], df_filtered["Open"])
        ]
        fig_sub.add_trace(go.Bar(
            x=df_filtered.index, y=df_filtered["Volume"],
            name="Volume", marker_color=colors, showlegend=False,
        ), row=current_row, col=1)

    fig_sub.update_layout(
        height=280 * rows,
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor="white",
        plot_bgcolor="#f7f8fa",
        showlegend=True,
    )
    st.plotly_chart(fig_sub, use_container_width=True)


# ── Actual vs Predicted ────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🎯 Actual vs Predicted — Test Split")

fig_pred = go.Figure()
fig_pred.add_trace(go.Scatter(
    x=result["test_dates"], y=result["y_test"],
    mode="lines", name="Actual",
    line=dict(color="#3b82f6", width=2),
))
fig_pred.add_trace(go.Scatter(
    x=result["test_dates"], y=result["y_pred"],
    mode="lines", name="Predicted",
    line=dict(color="#f59e0b", width=2, dash="dot"),
))
fig_pred.update_layout(
    height=350,
    margin=dict(l=0, r=0, t=10, b=0),
    paper_bgcolor="white",
    plot_bgcolor="#f7f8fa",
    yaxis_title="Price",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig_pred, use_container_width=True)


# ── Feature Importance ────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🔍 Feature Importance")

importances = result["model"].feature_importances_
feat_df = pd.DataFrame({
    "Feature":    FEATURE_COLS,
    "Importance": importances,
}).sort_values("Importance", ascending=True)

fig_feat = go.Figure(go.Bar(
    x=feat_df["Importance"],
    y=feat_df["Feature"],
    orientation="h",
    marker_color="#3b82f6",
))
fig_feat.update_layout(
    height=300,
    margin=dict(l=0, r=0, t=10, b=0),
    paper_bgcolor="white",
    plot_bgcolor="#f7f8fa",
    xaxis_title="Importance Score",
)
st.plotly_chart(fig_feat, use_container_width=True)


# ── Raw Data Table ────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📋 View Raw Dataset"):
    st.dataframe(
        df_train.reset_index().rename(columns={"index": "Date"}),
        use_container_width=True,
        height=350,
    )


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Silver Price Prediction · XGBoost Regression · Dataset: 500 trading days (Oct 2024 – Sep 2026)")
