
# Pairs Trading Strategy Using Statistical Arbitrage

This repository implements a **Pairs Trading** strategy based on statistical arbitrage. The goal is to identify pairs of stocks that move together historically and exploit temporary deviations in their price relationship to generate profits.

---

## Dataset

- **Source**: Historical stock price data fetched using the `yfinance` API.
- **Symbols**: A list of user-specified stock symbols.
- **Timeframe**: Customizable start and end dates for backtesting.

---

## Approach

The project combines data preprocessing, pair selection, strategy backtesting, and performance evaluation. It includes the following stages:

### 1. Data Collection & Preprocessing
- **Download**: Automated fetching of historical price data using `yfinance`.
- **Cleaning**: Removal of missing values and normalization.
- **Log Returns**: Computation for stability and mean-reversion testing.

### 2. Pair Selection
- **Correlation Analysis**: Identify pairs with high Pearson correlation.
- **Cointegration Test (Optional)**: Further statistical filtering for mean-reverting pairs.
- **Selection**: Top N pairs are selected based on correlation or test statistics.

### 3. Strategy Backtesting
- **Z-Score Computation**: Measure the spread between the pair.
- **Trading Rules**:
  - Long one asset, short the other when z-score exceeds threshold.
  - Exit trades when z-score returns to mean.
- **Capital Allocation**: Equal capital allocation or based on volatility.
- **Transaction Logging**: Track entries, exits, and P&L.

### 4. Portfolio Evaluation
- **Equity Curve**: Portfolio value over time.
- **Metrics**:
  - Total Return
  - Sharpe Ratio
  - Maximum Drawdown
  - Win/Loss Ratio

---

## Evaluation Metrics

The performance of the strategy is evaluated using:

| Metric              | Description                                  |
|---------------------|----------------------------------------------|
| **Total Return**    | Overall profit/loss from trades              |
| **Sharpe Ratio**    | Risk-adjusted return                         |
| **Drawdown**        | Largest peak-to-trough decline               |
| **Trade Accuracy**  | Percentage of profitable trades              |
| **Annualized Return** | Compounded yearly return                   |

---

## Results

The strategy demonstrated profitable behavior under historical data. Below are some key findings:

- Successfully detected mean-reverting behavior in selected pairs.
- Generated consistent returns with relatively low drawdown.
- Performance varied across pairs; diversification improved results.
- Trade timing based on z-score thresholds proved effective.

---

## Challenges

- **Lookahead Bias**: Care was taken to prevent use of future data.
- **Transaction Costs**: Not included; could impact real-world performance.
- **Data Quality**: Stock splits or dividend adjustments may affect returns.
- **Stationarity**: Market relationships change over time — regular recalibration needed.

---

## Technologies Used

- `Python 3.x`
- `yfinance`
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scipy.stats` for correlation and z-score
- `statsmodels` for cointegration testing (optional)

---
