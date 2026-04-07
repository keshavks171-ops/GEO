"""
Stock Predictor using Random Forest Classifier

Predicts whether a stock's closing price will go UP or DOWN the next trading day
based on technical indicators computed from historical price data.

Usage:
    python stock_predictor.py --ticker AAPL --start 2020-01-01 --end 2024-01-01
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def compute_bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return upper, sma, lower


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicator features and the binary target to the dataframe."""
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    feat = pd.DataFrame(index=df.index)

    # Price-based features
    feat["return_1d"] = close.pct_change(1)
    feat["return_3d"] = close.pct_change(3)
    feat["return_5d"] = close.pct_change(5)
    feat["return_10d"] = close.pct_change(10)
    feat["return_20d"] = close.pct_change(20)

    # Moving averages
    for w in [5, 10, 20, 50]:
        feat[f"sma_{w}"] = close.rolling(w).mean() / close - 1

    feat["ema_12"] = close.ewm(span=12, adjust=False).mean() / close - 1
    feat["ema_26"] = close.ewm(span=26, adjust=False).mean() / close - 1

    # Volatility
    feat["volatility_5d"] = close.pct_change().rolling(5).std()
    feat["volatility_20d"] = close.pct_change().rolling(20).std()

    # RSI
    feat["rsi_14"] = compute_rsi(close, 14)
    feat["rsi_28"] = compute_rsi(close, 28)

    # MACD
    macd, macd_signal = compute_macd(close)
    feat["macd"] = macd / close
    feat["macd_signal"] = macd_signal / close
    feat["macd_hist"] = (macd - macd_signal) / close

    # Bollinger Bands position
    bb_upper, bb_mid, bb_lower = compute_bollinger_bands(close)
    band_width = bb_upper - bb_lower
    feat["bb_position"] = (close - bb_lower) / band_width.replace(0, np.nan)
    feat["bb_width"] = band_width / bb_mid

    # High-Low spread
    feat["hl_spread"] = (high - low) / close

    # Volume features
    feat["volume_change"] = volume.pct_change()
    feat["volume_ma10"] = volume.rolling(10).mean()
    feat["volume_ratio"] = volume / feat["volume_ma10"]
    feat.drop(columns=["volume_ma10"], inplace=True)

    # Target: 1 if next day close > today close, else 0
    feat["target"] = (close.shift(-1) > close).astype(int)

    return feat


# ---------------------------------------------------------------------------
# Model training and evaluation
# ---------------------------------------------------------------------------

def train_evaluate(
    features: pd.DataFrame,
    n_splits: int = 5,
    n_estimators: int = 300,
    random_state: int = 42,
):
    """Walk-forward cross-validation with TimeSeriesSplit."""
    X = features.drop(columns=["target"])
    y = features["target"]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    scaler = StandardScaler()

    fold_results = []
    all_y_true, all_y_pred, all_y_prob = [], [], []

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=8,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)

        rf.fit(X_train_sc, y_train)
        y_pred = rf.predict(X_test_sc)
        y_prob = rf.predict_proba(X_test_sc)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        fold_results.append({"fold": fold, "accuracy": acc, "roc_auc": auc})
        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())
        all_y_prob.extend(y_prob.tolist())

        print(f"  Fold {fold}: Accuracy={acc:.4f}  ROC-AUC={auc:.4f}")

    # Final model on all data
    X_sc = scaler.fit_transform(X)
    rf.fit(X_sc, y)

    return rf, scaler, fold_results, all_y_true, all_y_pred, all_y_prob, X.columns.tolist()


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_results(
    ticker: str,
    fold_results: list,
    y_true: list,
    y_pred: list,
    y_prob: list,
    feature_names: list,
    importances: np.ndarray,
):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Random Forest Stock Predictor — {ticker}", fontsize=15, fontweight="bold")

    # 1. Fold metrics
    ax = axes[0, 0]
    df_folds = pd.DataFrame(fold_results)
    x = np.arange(len(df_folds))
    width = 0.35
    ax.bar(x - width / 2, df_folds["accuracy"], width, label="Accuracy", color="steelblue")
    ax.bar(x + width / 2, df_folds["roc_auc"], width, label="ROC-AUC", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {i}" for i in df_folds["fold"]])
    ax.set_ylim(0, 1)
    ax.set_title("Cross-Validation Metrics per Fold")
    ax.legend()
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)

    # 2. Confusion matrix
    ax = axes[0, 1]
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["DOWN", "UP"], yticklabels=["DOWN", "UP"],
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (all folds)")

    # 3. ROC curve
    ax = axes[1, 0]
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (all folds)")
    ax.legend(loc="lower right")

    # 4. Feature importances (top 15)
    ax = axes[1, 1]
    top_n = 15
    indices = np.argsort(importances)[-top_n:]
    ax.barh(
        [feature_names[i] for i in indices],
        importances[indices],
        color="mediumseagreen",
    )
    ax.set_title(f"Top {top_n} Feature Importances")
    ax.set_xlabel("Mean Decrease in Impurity")

    plt.tight_layout()
    out_path = f"{ticker}_results.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to: {out_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Prediction for next trading day
# ---------------------------------------------------------------------------

def predict_next_day(model: RandomForestClassifier, scaler: StandardScaler, features: pd.DataFrame, ticker: str):
    """Use the last available row (today) to predict tomorrow's direction."""
    X = features.drop(columns=["target"])
    last_row = X.iloc[[-1]]
    last_row_sc = scaler.transform(last_row)
    pred = model.predict(last_row_sc)[0]
    prob = model.predict_proba(last_row_sc)[0]

    direction = "UP" if pred == 1 else "DOWN"
    confidence = prob[pred]

    print("\n" + "=" * 50)
    print(f"Next-day prediction for {ticker}")
    print(f"  Direction : {direction}")
    print(f"  Confidence: {confidence:.2%}")
    print(f"  P(UP)     : {prob[1]:.2%}   P(DOWN): {prob[0]:.2%}")
    print("=" * 50)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stock direction predictor using Random Forest")
    parser.add_argument("--ticker", default="AAPL", help="Stock ticker symbol (default: AAPL)")
    parser.add_argument("--start", default="2019-01-01", help="Start date YYYY-MM-DD (default: 2019-01-01)")
    parser.add_argument("--end", default="2024-12-31", help="End date YYYY-MM-DD (default: 2024-12-31)")
    parser.add_argument("--splits", type=int, default=5, help="Number of TimeSeriesSplit folds (default: 5)")
    parser.add_argument("--trees", type=int, default=300, help="Number of trees in the forest (default: 300)")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    print(f"\nDownloading data for {ticker} ({args.start} → {args.end})...")
    raw = yf.download(ticker, start=args.start, end=args.end, auto_adjust=True, progress=False)

    if raw.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. Check the symbol and date range.")

    # Flatten multi-level columns that yfinance sometimes returns
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    print(f"  Downloaded {len(raw)} trading days.\n")

    print("Building features...")
    features = build_features(raw)
    features.dropna(inplace=True)
    # Drop the last row (target is NaN because there is no "next day" yet)
    features = features.iloc[:-1]
    print(f"  Feature matrix: {features.shape[0]} rows × {features.shape[1] - 1} features\n")

    print(f"Training Random Forest with {args.splits}-fold walk-forward CV...")
    model, scaler, fold_results, y_true, y_pred, y_prob, feat_names = train_evaluate(
        features, n_splits=args.splits, n_estimators=args.trees
    )

    # ---- Summary ----
    overall_acc = accuracy_score(y_true, y_pred)
    overall_auc = roc_auc_score(y_true, y_prob)
    print(f"\nOverall CV accuracy : {overall_acc:.4f}")
    print(f"Overall CV ROC-AUC  : {overall_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["DOWN", "UP"]))

    # ---- Next-day prediction ----
    # Rebuild full feature set including the last row (today's data)
    features_full = build_features(raw)
    features_full.dropna(inplace=True)
    predict_next_day(model, scaler, features_full, ticker)

    # ---- Plots ----
    plot_results(
        ticker=ticker,
        fold_results=fold_results,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        feature_names=feat_names,
        importances=model.feature_importances_,
    )


if __name__ == "__main__":
    main()
