"""
Quantitative Finance Research Project
Question: Can simple historical price features predict next-day NIFTY 50 direction?

This script downloads daily NIFTY 50 index data, constructs features using only
information available at the end of each day, performs a chronological train/test
split, compares logistic regression with a majority-class baseline, and saves
the actual results and plots.

Important:
- This is an educational prediction experiment, not investment advice.
- It does not claim that a model is profitable.
- Results are generated only when the script is run; do not invent or hand-edit them.
- Because the data provider can revise or change historical data, save the downloaded
  CSV and metadata with each run.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TICKER = "^NSEI"          # NIFTY 50 index symbol used by Yahoo Finance
PERIOD = "10y"
INTERVAL = "1d"
TRAIN_FRACTION = 0.80
RANDOM_STATE = 42
OUT = Path("outputs")
FIG = OUT / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "return_today",
    "return_previous_day",
    "mean_return_5d",
    "volatility_20d",
    "price_gap_sma5",
    "price_gap_sma20",
]


def get_close_series(raw: pd.DataFrame) -> pd.Series:
    """Extract a Close series from yfinance output, including MultiIndex formats."""
    if raw.empty:
        raise RuntimeError(
            "No data was downloaded. Check your internet connection, ticker symbol, "
            "and whether the data provider is available."
        )
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        if TICKER in close.columns:
            close = close[TICKER]
        else:
            close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors="coerce").dropna()
    close.name = "Close"
    if len(close) < 250:
        raise RuntimeError(
            f"Only {len(close)} valid price rows were downloaded. "
            "The experiment needs more history; try again later."
        )
    return close


def download_data() -> tuple[pd.DataFrame, pd.Series, dict]:
    """Download data and save the raw observations used for this run."""
    raw = yf.download(
        TICKER,
        period=PERIOD,
        interval=INTERVAL,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    close = get_close_series(raw)
    close.index = pd.to_datetime(close.index)
    close = close.sort_index()
    close = close[~close.index.duplicated(keep="last")]

    raw_path = OUT / "nifty50_daily_prices.csv"
    close.to_frame().to_csv(raw_path, index_label="Date")

    metadata = {
        "ticker": TICKER,
        "provider": "Yahoo Finance via yfinance",
        "period_requested": PERIOD,
        "interval": INTERVAL,
        "auto_adjust": True,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "first_observation": str(close.index.min().date()),
        "last_observation": str(close.index.max().date()),
        "number_of_price_rows": int(len(close)),
        "python_version": sys.version.split()[0],
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
    }
    (OUT / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return raw, close, metadata


def make_dataset(close: pd.Series) -> pd.DataFrame:
    """
    Each row at date t uses data known at the close of t.
    The target is whether the next observed trading day's return is positive.
    """
    frame = pd.DataFrame({"close": close})
    returns = frame["close"].pct_change()
    frame["return_today"] = returns
    frame["return_previous_day"] = returns.shift(1)
    frame["mean_return_5d"] = returns.rolling(5).mean()
    frame["volatility_20d"] = returns.rolling(20).std()
    frame["price_gap_sma5"] = frame["close"] / frame["close"].rolling(5).mean() - 1
    frame["price_gap_sma20"] = frame["close"] / frame["close"].rolling(20).mean() - 1

    frame["next_day_return"] = frame["close"].shift(-1) / frame["close"] - 1
    frame["target_up"] = (frame["next_day_return"] > 0).astype("int64")
    # The final row has no next-day return; remove it, as well as rows without
    # enough history to calculate the rolling features.
    frame.loc[frame["next_day_return"].isna(), "target_up"] = np.nan
    frame = frame.dropna(subset=FEATURES + ["next_day_return", "target_up"]).copy()
    frame["target_up"] = frame["target_up"].astype(int)
    return frame


def save_figures(frame: pd.DataFrame, test: pd.DataFrame, metrics: pd.DataFrame) -> None:
    """Create plots from the downloaded data and the actual model evaluation."""
    # Price history
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(frame.index, frame["close"], linewidth=1)
    ax.set_title("NIFTY 50 daily closing level (downloaded sample)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Index level")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "price_history.png", dpi=180)
    plt.close(fig)

    # Daily returns
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(frame.index, frame["return_today"] * 100, linewidth=0.6)
    ax.axhline(0, linewidth=0.8)
    ax.set_title("Daily percentage returns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Return (%)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "daily_returns.png", dpi=180)
    plt.close(fig)

    # Model vs baseline metric comparison
    plot_metrics = metrics.set_index("model")[["accuracy", "balanced_accuracy", "f1"]]
    ax = plot_metrics.plot(kind="bar", figsize=(8, 5), ylim=(0, 1))
    ax.set_title("Out-of-sample classification metrics")
    ax.set_ylabel("Score (0–1)")
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIG / "model_comparison.png", dpi=180)
    plt.close()

    # Confusion matrix for logistic regression
    cm = confusion_matrix(test["actual"], test["prediction"], labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    image = ax.imshow(cm)
    ax.set_title("Logistic regression: test-set confusion matrix")
    ax.set_xlabel("Predicted direction (0=down/flat, 1=up)")
    ax.set_ylabel("Actual direction (0=down/flat, 1=up)")
    ax.set_xticks([0, 1], labels=["Down/flat", "Up"])
    ax.set_yticks([0, 1], labels=["Down/flat", "Up"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIG / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    # Actual next-day return and model probability on the held-out period
    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.plot(test.index, test["next_day_return"] * 100, linewidth=0.7,
             label="Actual next-day return (%)")
    ax1.axhline(0, linewidth=0.8)
    ax1.set_ylabel("Actual next-day return (%)")
    ax1.set_xlabel("Prediction date")
    ax1.set_title("Held-out period: returns and predicted probability")
    ax2 = ax1.twinx()
    ax2.plot(test.index, test["probability_up"], linewidth=0.8, alpha=0.75,
             label="Predicted probability of up day")
    ax2.set_ylabel("Predicted probability of up day")
    ax2.set_ylim(0, 1)
    ax1.grid(True, alpha=0.25)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "test_predictions.png", dpi=180)
    plt.close(fig)


def build_results_report(metrics: pd.DataFrame, metadata: dict, frame: pd.DataFrame,
                         split_date: str, positive_rate_train: float,
                         positive_rate_test: float) -> None:
    """Write a report containing numbers measured in this run."""
    model_metrics = metrics.set_index("model")
    logit = model_metrics.loc["Logistic regression"]
    baseline = model_metrics.loc["Majority-class baseline"]
    if logit["accuracy"] > baseline["accuracy"]:
        accuracy_statement = (
            "Logistic regression had higher accuracy than the majority-class baseline "
            "on this held-out sample."
        )
    elif logit["accuracy"] < baseline["accuracy"]:
        accuracy_statement = (
            "Logistic regression had lower accuracy than the majority-class baseline "
            "on this held-out sample."
        )
    else:
        accuracy_statement = (
            "Logistic regression and the majority-class baseline had equal accuracy "
            "on this held-out sample."
        )

    report = f"""# Results Report: NIFTY 50 Next-Day Direction Experiment

## Run information

- Data source: {metadata['provider']} (retrieved through `yfinance`)
- Ticker: `{metadata['ticker']}`
- Download timestamp (UTC): {metadata['downloaded_at_utc']}
- First price date: {metadata['first_observation']}
- Last price date: {metadata['last_observation']}
- Number of price rows: {metadata['number_of_price_rows']}
- Number of usable feature/target rows: {len(frame)}
- Chronological training/test split date: {split_date}
- Training fraction: {TRAIN_FRACTION:.0%}

## Research question

Can a logistic regression model using simple historical price features predict whether
the next observed NIFTY 50 trading-day return is positive, and does it outperform a
majority-class baseline on a later held-out period?

## Measured results

{metrics.to_markdown(index=False, floatfmt='.4f')}

- Share of positive target days in training data: {positive_rate_train:.4f}
- Share of positive target days in test data: {positive_rate_test:.4f}

## Interpretation

{accuracy_statement}

This statement compares accuracy only. Inspect balanced accuracy, F1, the confusion
matrix, class balance, and the prediction plot before drawing broader conclusions.
A higher score on one held-out period does not establish a stable market anomaly,
future performance, or profitability.

## Method

At each date t, features use closing-price information available by the end of date t.
The target is 1 when the next observed close-to-close return is positive, and 0 otherwise.
Rows are ordered chronologically. The earliest {TRAIN_FRACTION:.0%} of usable observations
are used for fitting; the later observations are held out for evaluation. Standardization
is fit on the training set only using a scikit-learn pipeline. The logistic regression
model is compared with a dummy classifier that predicts the most frequent class in the
training set.

## Features

- `return_today`: close-to-close percentage return through date t.
- `return_previous_day`: previous observed daily return.
- `mean_return_5d`: mean of the last five observed daily returns, including date t.
- `volatility_20d`: standard deviation of the last 20 observed daily returns.
- `price_gap_sma5`: close divided by its 5-day moving average, minus one.
- `price_gap_sma20`: close divided by its 20-day moving average, minus one.

## Limitations

1. Historical relationships may not persist; financial markets are noisy and adaptive.
2. A single chronological split can be sensitive to the chosen time period.
3. Index direction prediction is not the same as a profitable trading strategy.
4. Transaction costs, slippage, position sizing, risk, and market impact are not modeled.
5. The data is retrieved from a third-party provider and may be revised or unavailable.
6. This project uses a small set of features and models; model selection can create
   overfitting if many alternatives are tried and only the best is reported.
7. The target treats a zero return as not-up; exact zero returns are rare but this rule
   should be remembered when interpreting the target.

## Reproducibility

Run `python quant_finance_experiment.py` after installing `requirements.txt`.
The script saves the downloaded price series, run metadata, row-level predictions,
summary metrics, and plots in `outputs/`. Keep these files with the report so the
exact sample and results can be audited. A later download may differ.

## Conclusion (write after examining all outputs)

Use the measured metrics and figures to answer the research question. State clearly
whether the model did or did not outperform the baseline on the held-out period.
Discuss uncertainty and limitations; do not claim predictive power or profitability
that the experiment has not established.

## References

- Yahoo Finance data access through `yfinance`: https://github.com/ranaroussi/yfinance
- scikit-learn `TimeSeriesSplit` documentation (background on time-aware evaluation):
  https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
- scikit-learn `LogisticRegression` documentation:
  https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- scikit-learn model evaluation documentation:
  https://scikit-learn.org/stable/modules/model_evaluation.html
- Fama, E. F. (1970). Efficient Capital Markets: A Review of Theory and Empirical Work.
  *The Journal of Finance*, 25(2), 383–417. https://doi.org/10.2307/2325486
"""
    (OUT / "results_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    print("Downloading daily NIFTY 50 data...")
    _, close, metadata = download_data()
    frame = make_dataset(close)
    if len(frame) < 250:
        raise RuntimeError("Too few usable observations after feature construction.")

    split_idx = int(len(frame) * TRAIN_FRACTION)
    train = frame.iloc[:split_idx].copy()
    test = frame.iloc[split_idx:].copy()
    if train["target_up"].nunique() < 2:
        raise RuntimeError("Training target contains only one class; cannot fit logistic regression.")
    if test["target_up"].nunique() < 2:
        print("Warning: the test period contains only one target class; some metrics may be limited.")

    X_train, y_train = train[FEATURES], train["target_up"]
    X_test, y_test = test[FEATURES], test["target_up"]

    model = Pipeline([
        ("scale", StandardScaler()),
        ("logistic", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ])
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    baseline_model = DummyClassifier(strategy="most_frequent")
    baseline_model.fit(X_train, y_train)
    baseline_predictions = baseline_model.predict(X_test)

    test_output = test[["close", "next_day_return", "target_up"]].copy()
    test_output = test_output.rename(columns={"target_up": "actual"})
    test_output["prediction"] = predictions
    test_output["probability_up"] = probabilities
    test_output["baseline_prediction"] = baseline_predictions
    test_output.to_csv(OUT / "test_predictions.csv", index_label="prediction_date")

    def measure(name: str, y_true, y_pred, probs=None) -> dict:
        result = {
            "model": name,
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "precision_up": precision_score(y_true, y_pred, zero_division=0),
            "recall_up": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }
        if probs is not None and len(np.unique(y_true)) == 2:
            result["roc_auc"] = roc_auc_score(y_true, probs)
        else:
            result["roc_auc"] = np.nan
        return result

    metrics = pd.DataFrame([
        measure("Majority-class baseline", y_test, baseline_predictions),
        measure("Logistic regression", y_test, predictions, probabilities),
    ])
    metrics.to_csv(OUT / "model_metrics.csv", index=False)
    frame.to_csv(OUT / "modeling_dataset.csv", index_label="feature_date")

    test_for_plot = test_output.copy()
    test_for_plot.index = pd.to_datetime(test_for_plot.index)
    save_figures(frame, test_for_plot, metrics)

    split_date = str(test.index.min().date())
    build_results_report(
        metrics=metrics,
        metadata=metadata,
        frame=frame,
        split_date=split_date,
        positive_rate_train=float(y_train.mean()),
        positive_rate_test=float(y_test.mean()),
    )

    print("\n=== ACTUAL OUT-OF-SAMPLE RESULTS ===")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSaved outputs to:", OUT.resolve())
    print("Read outputs/results_report.md for the automatically generated report.")
    print("Graphs are in outputs/figures/. No investment or profitability claim is made.")


if __name__ == "__main__":
    main()
