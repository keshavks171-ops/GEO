"""
Flask backend for the Stock Predictor website.
Run with:  python server.py
Then open:  http://localhost:5000
"""

import warnings
import numpy as np
import pandas as pd
import yfinance as yf

from flask import Flask, jsonify, request, render_template, send_file
import zipfile, io, os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

app = Flask(__name__)


# ── feature engineering ────────────────────────────────────────────────────────

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
    return macd_line, macd_line.ewm(span=signal, adjust=False).mean()


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
    feat["volume_change"] = volume.pct_change()
    feat["volume_ratio"] = volume / volume.rolling(10).mean()
    feat["target"] = (close.shift(-1) > close).astype(int)
    return feat


# ── API routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/download")
def download():
    base = os.path.dirname(os.path.abspath(__file__))
    buf = io.BytesIO()
    include = [
        "server.py", "stock_predictor.py", "app.py",
        "requirements.txt", "Procfile", "render.yaml",
        "templates/index.html",
        "static/css/style.css",
        "static/js/main.js",
    ]
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in include:
            full = os.path.join(base, rel)
            if os.path.exists(full):
                zf.write(full, os.path.join("stock_predictor", rel))
    buf.seek(0)
    return send_file(buf, as_attachment=True,
                     download_name="stock_predictor.zip",
                     mimetype="application/zip")


@app.route("/api/predict", methods=["POST"])
def predict():
    body = request.get_json(force=True)
    ticker = body.get("ticker", "AAPL").upper().strip()
    start = body.get("start", "2019-01-01")
    end = body.get("end", "2024-12-31")
    n_splits = int(body.get("n_splits", 5))
    n_trees = int(body.get("n_trees", 300))

    # ── download ──
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        return jsonify({"error": f"No data found for '{ticker}'"}), 400

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # ── features ──
    features_full = build_features(raw)
    features_full.dropna(inplace=True)
    features_train = features_full.iloc[:-1]

    X = features_train.drop(columns=["target"])
    y = features_train["target"]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    scaler = StandardScaler()
    rf = RandomForestClassifier(
        n_estimators=n_trees, max_depth=8, min_samples_leaf=20,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )

    fold_results, yt, yp, yprob = [], [], [], []

    for fold, (tr, te) in enumerate(tscv.split(X), 1):
        X_tr = scaler.fit_transform(X.iloc[tr])
        X_te = scaler.transform(X.iloc[te])
        rf.fit(X_tr, y.iloc[tr])
        pred = rf.predict(X_te)
        prob = rf.predict_proba(X_te)[:, 1]
        fold_results.append({
            "fold": fold,
            "accuracy": round(accuracy_score(y.iloc[te], pred), 4),
            "roc_auc": round(roc_auc_score(y.iloc[te], prob), 4),
        })
        yt.extend(y.iloc[te].tolist())
        yp.extend(pred.tolist())
        yprob.extend(prob.tolist())

    # retrain on all data
    X_sc = scaler.fit_transform(X)
    rf.fit(X_sc, y)

    # ── next-day prediction ──
    last = features_full.drop(columns=["target"]).iloc[[-1]]
    last_sc = scaler.transform(last)
    pred_label = int(rf.predict(last_sc)[0])
    prob_arr = rf.predict_proba(last_sc)[0].tolist()

    # ── confusion matrix ──
    cm = confusion_matrix(yt, yp).tolist()

    # ── ROC curve (downsample to ~200 pts) ──
    fpr, tpr, _ = roc_curve(yt, yprob)
    step = max(1, len(fpr) // 200)
    roc_data = {"fpr": fpr[::step].tolist(), "tpr": tpr[::step].tolist()}

    # ── feature importances (top 15) ──
    feat_names = X.columns.tolist()
    importances = rf.feature_importances_
    top_idx = np.argsort(importances)[-15:][::-1]
    feature_importance = [
        {"feature": feat_names[i], "importance": round(float(importances[i]), 5)}
        for i in top_idx
    ]

    # ── price history (last 252 days) ──
    price_history = {
        "dates": raw.index.strftime("%Y-%m-%d").tolist()[-252:],
        "close": [round(float(v), 2) for v in raw["Close"].tolist()[-252:]],
    }

    overall_acc = round(accuracy_score(yt, yp), 4)
    overall_auc = round(roc_auc_score(yt, yprob), 4)

    return jsonify({
        "ticker": ticker,
        "rows": len(raw),
        "prediction": {
            "direction": "UP" if pred_label == 1 else "DOWN",
            "confidence": round(float(prob_arr[pred_label]), 4),
            "p_up": round(float(prob_arr[1]), 4),
            "p_down": round(float(prob_arr[0]), 4),
        },
        "overall": {"accuracy": overall_acc, "roc_auc": overall_auc},
        "folds": fold_results,
        "confusion_matrix": cm,
        "roc_curve": roc_data,
        "feature_importance": feature_importance,
        "price_history": price_history,
    })


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
