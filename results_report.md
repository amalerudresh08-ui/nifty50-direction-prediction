# Results Report: NIFTY 50 Next-Day Direction Experiment

## Run information

- Data source: Yahoo Finance via yfinance (retrieved through `yfinance`)
- Ticker: `^NSEI`
- Download timestamp (UTC): 2026-10-10T19:55:21.589411+00:00
- First price date: 2016-10-10
- Last price date: 2026-10-09
- Number of price rows: 2466
- Number of usable feature/target rows: 2445
- Chronological training/test split date: 2024-10-17
- Training fraction: 80%

## Research question

Can a logistic regression model using simple historical price features predict whether
the next observed NIFTY 50 trading-day return is positive, and does it outperform a
majority-class baseline on a later held-out period?

## Measured results

| model                   |   accuracy |   balanced_accuracy |   precision_up |   recall_up |     f1 |   roc_auc |
|:------------------------|-----------:|--------------------:|---------------:|------------:|-------:|----------:|
| Majority-class baseline |     0.4847 |              0.5000 |         0.4847 |      1.0000 | 0.6529 |  nan      |
| Logistic regression     |     0.4765 |              0.4913 |         0.4802 |      0.9747 | 0.6435 |    0.4904 |

- Share of positive target days in training data: 0.5521
- Share of positive target days in test data: 0.4847

## Interpretation

Logistic regression had lower accuracy than the majority-class baseline on this held-out sample.

This statement compares accuracy only. Inspect balanced accuracy, F1, the confusion
matrix, class balance, and the prediction plot before drawing broader conclusions.
A higher score on one held-out period does not establish a stable market anomaly,
future performance, or profitability.

## Method

At each date t, features use closing-price information available by the end of date t.
The target is 1 when the next observed close-to-close return is positive, and 0 otherwise.
Rows are ordered chronologically. The earliest 80% of usable observations
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
