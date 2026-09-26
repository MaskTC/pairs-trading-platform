# Statistical Arbitrage (Pairs Trading) Research Platform

A Python research platform that identifies pairs of historically related stocks,
validates that their relationship is *statistically* mean-reverting (not just
correlated), generates trading signals from deviations in their spread, and
evaluates the strategy through an event-driven backtester with realistic
transaction costs, parameter optimization, and out-of-sample validation.

A companion **Kalshi mispricing module** (`kalshi/`) provides a read-only
client for Kalshi's public prediction-market API plus locked-in-edge detection
(complementary yes/no prices, cross-market event arbitrage).

## Strategy in brief

1. **Pair identification** (`src/pairs_trading/pairs.py`) — for every candidate
   pair: Pearson correlation, OLS hedge ratio β (statsmodels), the
   **Engle-Granger cointegration test** (`statsmodels.tsa.stattools.coint`),
   and the half-life of mean reversion from the regression
   Δspread = λ·spread₋₁ (half-life = −ln 2 / λ). A pair is traded only if it
   passes *all three* filters: high correlation **and** p < 0.05 **and**
   0 < half-life < 252 days. Pairs that are merely correlated are rejected.
2. **Signal generation** (`signals.py`) — spread `S = A − β·B`, rolling
   z-score `(S − rolling_mean) / rolling_std`, then a state machine:
   enter long/short when |z| > entry threshold, exit when |z| < exit
   threshold or z crosses 0. Signals at bar *t* use only data through bar *t*
   and are executed at that close — **no look-ahead bias**.
3. **Backtesting** (`backtest.py`) — explicit bar-by-bar event loop tracking
   cash and share positions (long 1×A / short β×B per spread contract), so
   equity accounting is exact: with no trade, Δequity = position × Δspread.
   Costs (5 bps commission + 2 bps slippage per side by default) are charged
   once per position change; a flip costs twice (close + open).
4. **Optimization** (`optimize.py`) — grid search over
   window × entry_z × exit_z ranked by **in-sample Sharpe**, then evaluated
   once on a held-out test slice. A large train/test Sharpe divergence raises
   an overfitting flag.
5. **Metrics** (`metrics.py`) — Sharpe (√252 annualized, 0.0 on zero
   volatility), max drawdown from the running peak, CAGR, win rate on
   completed round trips, cumulative returns.

## Results (real run, 2020-01-01 → 2025-01-01, $10k, 5 bps + 2 bps per side)

Pair screening on KO, PEP, XOM, CVX (1,258 trading days):

| pair | corr | β | coint p | half-life (d) | selected |
|------|------|---|---------|---------------|----------|
| KO/PEP | 0.897 | 0.34 | 0.577 | 97.3 | ❌ (correlated but not cointegrated) |
| KO/XOM | 0.894 | 0.23 | 0.015 | 41.2 | ✅ |
| KO/CVX | 0.860 | 0.19 | 0.079 | 52.7 | ❌ |
| PEP/XOM | 0.908 | 0.61 | 0.019 | 29.1 | ✅ |
| PEP/CVX | 0.911 | 0.54 | 0.003 | 25.3 | ✅ |
| XOM/CVX | 0.958 | 0.85 | 0.386 | 106.1 | ❌ (highest correlation — still rejected) |

Out-of-sample performance of the winning parameters (train 60% / test 40%):

| pair | window | entry_z | exit_z | train Sharpe | **test Sharpe** | test ann. return | test max DD | test win rate | test trades | overfit? |
|------|--------|---------|--------|--------------|-----------------|------------------|-------------|---------------|-------------|----------|
| KO/XOM | 20 | 2.5 | 0.0 | 0.595 | **−1.276** | −9.5% | −23.5% | 0.444 | 9 | ⚠️ yes |
| PEP/XOM | 20 | 2.5 | 0.5 | 1.137 | **0.153** | +0.5% | −5.4% | 0.750 | 4 | no |
| PEP/CVX | 60 | 2.5 | 0.0 | 0.836 | **0.258** | +1.6% | −9.2% | 0.600 | 5 | no |

Note what the numbers say: KO/PEP and XOM/CVX look like "obvious" pairs on
correlation alone, yet fail cointegration — the filter earns its keep. And
KO/XOM's in-sample Sharpe of 0.60 collapses to −1.28 out of sample, which is
exactly why the pipeline validates out of sample instead of reporting the
grid-search winner's training metrics.

![PEP/CVX spread and z-score bands](outputs/spread_PEP_CVX.png)
![PEP/CVX out-of-sample equity vs SPY buy-and-hold](outputs/equity_PEP_CVX.png)

## Setup

```bash
cd pairs-trading-platform
python3 -m venv .venv          # project-local venv (system pip is externally managed)
.venv/bin/pip install -r requirements.txt
```

(A virtualenv is used because the system Python is PEP-668 externally
managed; user-level installs would also work.)

## Usage

```bash
# Run the test suite (49 tests, all synthetic/seeded — no network needed)
.venv/bin/python -m pytest tests/ -v

# Run the full pipeline on real market data (downloads via yfinance)
.venv/bin/python scripts/run_pipeline.py
# -> outputs/summary.csv, outputs/spread_*.png, outputs/equity_*.png
```

## Project layout

```
pairs-trading-platform/
├── README.md
├── requirements.txt
├── scripts/
│   └── run_pipeline.py      # end-to-end: fetch -> pairs -> grid search -> backtest -> plots
├── src/pairs_trading/
│   ├── data.py              # yfinance fetch, calendar alignment, ffill, returns
│   ├── pairs.py             # correlation, OLS hedge ratio, Engle-Granger, half-life
│   ├── signals.py           # spread, rolling z-score, entry/exit state machine
│   ├── backtest.py          # event-driven bar-by-bar backtester (cash + shares)
│   ├── costs.py             # bps commission + slippage model
│   ├── metrics.py           # Sharpe, drawdown, CAGR, win rate
│   ├── optimize.py          # grid search with train/test split + overfit flag
│   └── visualize.py         # spread/z-band and equity-vs-benchmark plots
├── kalshi/
│   ├── client.py            # read-only Kalshi trade-api v2 client (public endpoints)
│   └── mispricing.py        # yes/no complement arb + cross-market event arb detection
├── tests/                   # pytest suite (see below)
└── outputs/                 # pipeline artifacts (CSVs, PNGs)
```

## Testing

49 tests, all passing (`python -m pytest tests/ -v`). Highlights:

- **No look-ahead**: a signal at bar *t* leaves equity at *t* untouched; the
  position earns only from *t+1*.
- **Exact accounting**: whenever the position is unchanged, equity change
  equals position × spread change, to the cent; costs are deducted exactly
  once per position change (a flip = close + open = 2 charges).
- **β recovery**: OLS recovers the true β=2.5 on a seeded cointegrated pair.
- **Cointegration**: accepts a truly cointegrated pair (p < 0.05), rejects two
  independent random walks.
- **Optimizer**: the reported winner is verified to be the genuine argmax of
  in-sample Sharpe, and test metrics are recomputed independently on the
  held-out slice.
- **Kalshi**: all tests use mocked responses (no network); the client
  endpoints and response shapes were verified live against the real API.

## Kalshi module

`kalshi/client.py` wraps the public endpoints (base
`https://api.elections.kalshi.com/trade-api/v2`, verified live 2026-09-26):
`GET /markets`, `GET /markets/{ticker}`, `GET /markets/{ticker}/orderbook`
(live shape: `{"orderbook_fp": {"yes_dollars": [...], "no_dollars": [...]}}`),
`GET /events`. `kalshi/mispricing.py` flags (a) markets where
yes_ask + no_ask < 100¢ (buy both) or yes_bid + no_bid > 100¢ (sell both),
and (b) events whose mutually-exclusive markets sum away from 100¢.
**Read-only analysis only — no order placement, no real-money trading.**

## Limitations (honest)

- **Fill-at-close assumption**: trades execute at the closing price with no
  market impact; real fills differ, especially at larger size.
- **Survivorship/selection bias**: pairs are picked from today's large-cap
  tickers; delisted or distressed names never enter the screen.
- **Regime change**: cointegration is estimated on history — the KO/XOM
  train/test collapse is a live example of a relationship breaking down.
- **Daily bars only**: no intraday data, so intraday mean reversion and
  overnight gaps are invisible to the backtest.
- **No borrow costs**: shorting β shares of B is assumed frictionless beyond
  the bps cost model; hard-to-borrow names would cost more.
- **Look-ahead in pair selection**: the screen uses the full sample; a
  production system would re-screen on rolling windows.
