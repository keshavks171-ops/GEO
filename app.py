"""
Stock Predictor Web App — Streamlit front-end
Run with:  streamlit run app.py
"""

import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Predictor · Random Forest",
    page_icon="📈",
    layout="wide",
)

# ── title ──────────────────────────────────────────────────────────────────────
st.title("📈 Stock Direction Predictor")
st.markdown(
    "Uses a **Random Forest Classifier** with walk-forward cross-validation to predict "
    "whether a stock's closing price will go **UP** or **DOWN** the next trading day."
)
st.divider()


# ── sidebar controls ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    ticker = st.text_input("Ticker symbol", value="AAPL").upper().strip()
    start_date = st.date_input("Start date", value=pd.Timestamp("2019-01-01"))
    end_date = st.date_input("End date", value=pd.Timestamp("2024-12-31"))
    n_splits = st.slider("CV folds (TimeSeriesSplit)", min_value=2, max_value=10, value=5)
    n_trees = st.slider("Number of trees", min_value=50, max_value=1000, value=300, step=50)
    run_btn = st.button("🚀 Run Prediction", type="primary", use_container_width=True)

    st.divider()
    st.caption("Data: Yahoo Finance · Model: scikit-learn")


# ── helpers (identical logic to stock_predictor.py) ───────────────────────────

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def compute_bollinger_bands(series, period=20, num_std=2.0):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    return sma + num_std * std, sma, sma - num_std * std


def build_features(df):
    close, high, low, volume = df["Close"], df["High"], df["Low"], df["Volume"]
    feat = pd.DataFrame(index=df.index)

    for d in [1, 3, 5, 10, 20]:
        feat[f"return_{d}d"] = close.pct_change(d)
    for w in [5, 10, 20, 50]:
        feat[f"sma_{w}"] = close.rolling(w).mean() / close - 1

    feat["ema_12"] = close.ewm(span=12, adjust=False).mean() / close - 1
    feat["ema_26"] = close.ewm(span=26, adjust=False).mean() / close - 1
    feat["volatility_5d"] = close.pct_change().rolling(5).std()
    feat["volatility_20d"] = close.pct_change().rolling(20).std()
    feat["rsi_14"] = compute_rsi(close, 14)
    feat["rsi_28"] = compute_rsi(close, 28)

    macd, macd_sig = compute_macd(close)
    feat["macd"] = macd / close
    feat["macd_signal"] = macd_sig / close
    feat["macd_hist"] = (macd - macd_sig) / close

    bb_upper, bb_mid, bb_lower = compute_bollinger_bands(close)
    bw = bb_upper - bb_lower
    feat["bb_position"] = (close - bb_lower) / bw.replace(0, np.nan)
    feat["bb_width"] = bw / bb_mid
    feat["hl_spread"] = (high - low) / close

    vol_ma10 = volume.rolling(10).mean()
    feat["volume_change"] = volume.pct_change()
    feat["volume_ratio"] = volume / vol_ma10

    feat["target"] = (close.shift(-1) > close).astype(int)
    return feat


@st.cache_data(show_spinner=False)
def load_data(ticker, start, end):
    raw = yf.download(ticker, start=str(start), end=str(end), auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return raw


def train_model(features, n_splits, n_trees):
    X, y = features.drop(columns=["target"]), features["target"]
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scaler = StandardScaler()
    rf = RandomForestClassifier(
        n_estimators=n_trees, max_depth=8, min_samples_leaf=20,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    fold_results, yt, yp, yprob = [], [], [], []
    progress = st.progress(0, text="Training…")

    for fold, (tr, te) in enumerate(tscv.split(X), 1):
        X_tr, X_te = scaler.fit_transform(X.iloc[tr]), scaler.transform(X.iloc[te])
        rf.fit(X_tr, y.iloc[tr])
        pred = rf.predict(X_te)
        prob = rf.predict_proba(X_te)[:, 1]
        fold_results.append({
            "Fold": fold,
            "Accuracy": accuracy_score(y.iloc[te], pred),
            "ROC-AUC": roc_auc_score(y.iloc[te], prob),
        })
        yt.extend(y.iloc[te].tolist())
        yp.extend(pred.tolist())
        yprob.extend(prob.tolist())
        progress.progress(fold / n_splits, text=f"Fold {fold}/{n_splits} done")

    progress.empty()
    # retrain on everything for next-day inference
    X_sc = scaler.fit_transform(X)
    rf.fit(X_sc, y)
    return rf, scaler, fold_results, yt, yp, yprob, X.columns.tolist()


# ── main content ───────────────────────────────────────────────────────────────

if run_btn:
    if start_date >= end_date:
        st.error("Start date must be before end date.")
        st.stop()

    # ── data ──
    with st.spinner(f"Downloading {ticker} data…"):
        raw = load_data(ticker, start_date, end_date)

    if raw.empty:
        st.error(f"No data found for **{ticker}**. Check the symbol and date range.")
        st.stop()

    st.success(f"Downloaded **{len(raw):,}** trading days for **{ticker}**")

    # ── features ──
    features_full = build_features(raw)
    features_full.dropna(inplace=True)
    features_train = features_full.iloc[:-1]   # exclude last row (no next-day target)

    # ── train ──
    with st.spinner("Training Random Forest…"):
        model, scaler, fold_results, yt, yp, yprob, feat_names = train_model(
            features_train, n_splits, n_trees
        )

    # ── next-day prediction banner ──────────────────────────────────────────
    st.subheader("🔮 Next-Day Prediction")
    last = features_full.drop(columns=["target"]).iloc[[-1]]
    last_sc = scaler.transform(last)
    pred_label = model.predict(last_sc)[0]
    prob_arr = model.predict_proba(last_sc)[0]
    direction = "UP 📈" if pred_label == 1 else "DOWN 📉"
    confidence = prob_arr[pred_label]
    p_up, p_down = prob_arr[1], prob_arr[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ticker", ticker)
    col2.metric("Predicted direction", direction)
    col3.metric("Confidence", f"{confidence:.1%}")
    col4.metric("P(UP) / P(DOWN)", f"{p_up:.1%} / {p_down:.1%}")

    st.divider()

    # ── CV summary ──────────────────────────────────────────────────────────
    st.subheader("📊 Cross-Validation Summary")
    df_folds = pd.DataFrame(fold_results)
    overall_acc = accuracy_score(yt, yp)
    overall_auc = roc_auc_score(yt, yprob)

    c1, c2 = st.columns(2)
    c1.metric("Overall Accuracy", f"{overall_acc:.4f}")
    c2.metric("Overall ROC-AUC", f"{overall_auc:.4f}")

    st.dataframe(
        df_folds.style.format({"Accuracy": "{:.4f}", "ROC-AUC": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )

    # ── classification report ───────────────────────────────────────────────
    with st.expander("📋 Classification Report"):
        report = classification_report(yt, yp, target_names=["DOWN", "UP"])
        st.code(report)

    st.divider()

    # ── plots ───────────────────────────────────────────────────────────────
    st.subheader("📉 Charts")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Random Forest Results — {ticker}", fontsize=13, fontweight="bold")

    # Confusion matrix
    cm = confusion_matrix(yt, yp)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=["DOWN", "UP"], yticklabels=["DOWN", "UP"])
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")
    axes[0].set_title("Confusion Matrix")

    # ROC curve
    fpr, tpr, _ = roc_curve(yt, yprob)
    axes[1].plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC={overall_auc:.4f}")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR")
    axes[1].set_title("ROC Curve"); axes[1].legend(loc="lower right")

    # Feature importances
    top_n = 15
    idx = np.argsort(model.feature_importances_)[-top_n:]
    axes[2].barh([feat_names[i] for i in idx], model.feature_importances_[idx], color="mediumseagreen")
    axes[2].set_title(f"Top {top_n} Feature Importances")

    plt.tight_layout()
    st.pyplot(fig)

    # ── price chart ─────────────────────────────────────────────────────────
    st.subheader("📅 Historical Close Price")
    st.line_chart(raw["Close"], use_container_width=True)

else:
    st.info("👈 Configure settings in the sidebar and click **Run Prediction** to start.")
