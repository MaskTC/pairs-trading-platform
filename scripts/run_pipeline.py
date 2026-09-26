"""End-to-end pipeline: fetch -> find pairs -> grid search -> backtest -> plots.

Stages
------
1. Daily stage: pair screening, 160-combination grid search with train/test
   split, baseline-vs-optimized Sharpe comparison, plots, summary.csv.
2. Signal-quality stage: naive (correlation-only) screen vs the full
   cointegration filter -- measures false-signal suppression.
3. Intraday stage: 1-minute bars for a 65-ticker universe (500k+
   observations), correlation pre-screen, cointegration check on the top
   candidates.

Data sources
------------
By default prices come from yfinance. Pass ``--data csv --data-path
prices.csv --data-format long`` to run the daily stages on your own
dataset instead (formats: long / wide / dir -- see
``pairs_trading.datasets``). The same flags exist for the intraday
stage (``--intraday-data csv --intraday-path ...``).

Usage:
    .venv/bin/python scripts/run_pipeline.py
    .venv/bin/python scripts/run_pipeline.py --data csv --data-path my_prices.csv --data-format long
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from pairs_trading import backtest, costs, data, datasets, intraday, metrics, optimize, pairs, quality, signals, visualize

TICKERS = ["KO", "PEP", "XOM", "CVX"]
BENCHMARK = "SPY"
START = "2020-01-01"
END = "2025-01-01"
INITIAL_CAPITAL = 10_000.0

# 8 x 5 x 4 = 160 parameter combinations.
WINDOWS = [10, 15, 20, 30, 45, 60, 90, 120]
ENTRY_ZS = [1.0, 1.5, 2.0, 2.5, 3.0]
EXIT_ZS = [0.0, 0.25, 0.5, 0.75]

# 80 liquid large-caps: ~7.8k minute bars each over the 30-day yfinance
# window -> 500k+ observations with headroom for failed downloads.
INTRADAY_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "TSLA",
    "AVGO", "JPM", "V", "MA", "XOM", "JNJ", "WMT", "PG",
    "ORCL", "NFLX", "AMD", "CRM", "INTC", "DIS", "BAC", "ABBV",
    "MRK", "PFE", "LLY", "COST", "HD", "KO", "PEP", "CVX",
    "ADBE", "QCOM", "TXN", "AMAT", "MU", "GS", "MS", "C",
    "CAT", "BA", "GE", "IBM", "UBER", "SHOP", "XYZ", "PLTR",
    "COIN", "MSTR", "NEE", "DUK", "SO", "T", "VZ", "CMCSA",
    "HON", "UNP", "LOW", "SBUX", "NKE", "MCD", "ABT", "DHR", "LIN",
    "ADP", "ICE", "CME", "AON", "APP", "AEP", "AMT", "PLD",
    "EQIX", "DLR", "SPG", "O", "KMI", "WMB", "EPD",
]


def _load_custom(path: str, fmt: str) -> pd.DataFrame:
    loaders = {
        "long": datasets.read_long_csv,
        "wide": datasets.read_wide_csv,
        "dir": datasets.read_ticker_dir,
    }
    try:
        loader = loaders[fmt]
    except KeyError:
        raise SystemExit(f"--data-format must be one of {sorted(loaders)}")
    return loader(path)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Pairs-trading research pipeline")
    p.add_argument("--data", choices=["yfinance", "csv"], default="yfinance",
                   help="price source for the daily stages")
    p.add_argument("--data-path", default=None,
                   help="CSV file (or directory, for --data-format dir)")
    p.add_argument("--data-format", choices=["long", "wide", "dir"],
                   default="long", help="custom CSV layout (see datasets)")
    p.add_argument("--intraday-data", choices=["yfinance", "csv"],
                   default="yfinance", help="price source for the intraday stage")
    p.add_argument("--intraday-path", default=None,
                   help="CSV file (or directory) of intraday bars")
    p.add_argument("--intraday-format", choices=["long", "wide", "dir"],
                   default="long", help="custom intraday CSV layout")
    args = p.parse_args(argv)
    if args.data == "csv" and not args.data_path:
        p.error("--data csv requires --data-path")
    if args.intraday_data == "csv" and not args.intraday_path:
        p.error("--intraday-data csv requires --intraday-path")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    n_combos = len(WINDOWS) * len(ENTRY_ZS) * len(EXIT_ZS)

    if args.data == "csv":
        print(f"Loading custom dataset: {args.data_path} (format={args.data_format}) ...")
        prices = _load_custom(args.data_path, args.data_format)
        tickers = [c for c in prices.columns if c != BENCHMARK]
    else:
        print(f"Fetching {TICKERS + [BENCHMARK]} from {START} to {END} ...")
        prices = data.fetch_prices(TICKERS + [BENCHMARK], START, END)
        tickers = TICKERS
    print(f"Aligned panel: {prices.shape[0]} bars, "
          f"{prices.index[0]} -> {prices.index[-1]}")

    if BENCHMARK in prices.columns:
        bench, bench_label = prices[BENCHMARK], "SPY buy-and-hold"
    else:
        bench, bench_label = prices.iloc[:, 0], f"{prices.columns[0]} buy-and-hold"
        print(f"  (no {BENCHMARK} in custom data; benchmarking vs {prices.columns[0]})")
    px = prices[[c for c in prices.columns if c != BENCHMARK]]

    print("\n=== Pair screening (corr >= 0.7, Engle-Granger p < 0.05, 0 < HL < 252) ===")
    found = pairs.find_pairs(px)
    print(found[["asset_a", "asset_b", "correlation", "beta",
                 "coint_p_value", "half_life", "selected"]].to_string(index=False))

    cost_model = costs.TransactionCostModel(cost_bps=5.0, slippage_bps=2.0)
    selected = found[found["selected"]]
    if selected.empty:
        print("\nNo pairs passed the screen.")
        return

    summary_rows = []
    sharpe_improvements = []
    for _, row in selected.iterrows():
        a_name, b_name = row["asset_a"], row["asset_b"]
        pair = f"{a_name}/{b_name}"
        print(f"\n=== {pair}: grid search ({n_combos} combos, train 60% / test 40%) ===")
        a, b = px[a_name], px[b_name]
        opt = optimize.grid_search(
            a, b,
            windows=WINDOWS, entry_zs=ENTRY_ZS, exit_zs=EXIT_ZS,
            train_frac=0.6, cost_model=cost_model,
            initial_capital=INITIAL_CAPITAL,
        )
        bp = opt["best_params"]
        tm = opt["test_metrics"]
        print(f"  best params: window={bp['window']}, entry_z={bp['entry_z']}, "
              f"exit_z={bp['exit_z']}  (train beta={opt['train_beta']:.4f})")
        print(f"  train Sharpe: {opt['best_train_sharpe']:.3f} | "
              f"test Sharpe: {tm['sharpe']:.3f} | "
              f"overfit flag: {opt['overfit_flag']}")

        imp = optimize.baseline_vs_optimized(
            a, b, bp, opt["train_beta"],
            train_frac=0.6, cost_model=cost_model,
            initial_capital=INITIAL_CAPITAL,
        )
        sharpe_improvements.append(imp["improvement"])
        print(f"  baseline test Sharpe: {imp['baseline_sharpe']:.3f} -> "
              f"optimized: {imp['optimized_sharpe']:.3f} "
              f"(improvement {imp['improvement']:+.3f})")

        # Full-sample signals with the winning params for the spread plot.
        spread = signals.compute_spread(a, b, opt["train_beta"])
        z = signals.compute_zscore(spread, window=bp["window"])
        sig = signals.generate_signals(z, entry_z=bp["entry_z"], exit_z=bp["exit_z"])
        spread_path = os.path.join(out_dir, f"spread_{a_name}_{b_name}.png")
        visualize.plot_spread(
            spread, z, bp["entry_z"], bp["exit_z"], spread_path,
            title=f"{pair} spread",
        )

        # Out-of-sample equity vs benchmark buy-and-hold on the test slice.
        split = int(len(a) * 0.6)
        a_te, b_te = a.iloc[split:], b.iloc[split:]
        z_te = signals.compute_zscore(
            signals.compute_spread(a_te, b_te, opt["train_beta"]),
            window=bp["window"],
        )
        sig_te = signals.generate_signals(
            z_te, entry_z=bp["entry_z"], exit_z=bp["exit_z"]
        )
        res_te = backtest.run_backtest(
            a_te, b_te, opt["train_beta"], sig_te["target"],
            cost_model=cost_model, initial_capital=INITIAL_CAPITAL,
        )
        bench_te = bench.iloc[split:]
        bench_curve = bench_te / bench_te.iloc[0] * INITIAL_CAPITAL
        equity_path = os.path.join(out_dir, f"equity_{a_name}_{b_name}.png")
        visualize.plot_equity(
            res_te["equity"], bench_curve, equity_path,
            title=f"{pair} out-of-sample equity vs {bench_label}",
        )
        print(f"  plots: {spread_path}, {equity_path}")

        summary_rows.append(
            {
                "pair": pair,
                "window": bp["window"],
                "entry_z": bp["entry_z"],
                "exit_z": bp["exit_z"],
                "beta": round(opt["train_beta"], 4),
                "train_sharpe": round(opt["best_train_sharpe"], 3),
                "test_sharpe": round(tm["sharpe"], 3),
                "test_ann_return": round(tm["annualized_return"], 4),
                "test_ann_vol": round(tm["annualized_vol"], 4),
                "test_max_dd": round(tm["max_drawdown"], 4),
                "test_win_rate": (
                    round(tm["win_rate"], 3) if tm["win_rate"] == tm["win_rate"] else None
                ),
                "test_n_trades": tm["n_trades"],
                "overfit_flag": opt["overfit_flag"],
                "sharpe_improvement": round(imp["improvement"], 3),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out_dir, "summary.csv")
    summary.to_csv(summary_path, index=False)
    print("\n=== Out-of-sample summary ===")
    print(summary.to_string(index=False))
    print(f"\nSaved: {summary_path}")

    # ---- Signal-quality stage -------------------------------------------
    print("\n=== Signal quality: naive (corr-only) vs filtered (corr + cointegration) ===")
    q = quality.screen_comparison(px, corr_threshold=0.7, cost_model=cost_model,
                                  initial_capital=INITIAL_CAPITAL)
    print(f"  naive screen:    {q['naive']['n_pairs']} pairs, "
          f"{q['naive']['total_signals']} entry signals, "
          f"{q['naive']['losing_trades']} losing round trips")
    print(f"  filtered screen: {q['filtered']['n_pairs']} pairs, "
          f"{q['filtered']['total_signals']} entry signals, "
          f"{q['filtered']['losing_trades']} losing round trips")
    print(f"  false-signal suppression: {q['signal_suppression']:.1%}")
    print(f"  losing-trade reduction:   {q['losing_trade_reduction']:.1%}")

    # ---- Intraday stage ---------------------------------------------------
    print(f"\n=== Intraday stage: 1-minute bars, "
          f"{len(INTRADAY_TICKERS)} tickers, last 30 days ===")
    try:
        if args.intraday_data == "csv":
            print(f"  loading custom intraday data: {args.intraday_path}")
            minute_prices = _load_custom(args.intraday_path, args.intraday_format)
        else:
            minute_prices = intraday.fetch_intraday(
                INTRADAY_TICKERS, days=30, interval="1m",
                batch_size=16, pause=0.3,
            )
        n_obs = intraday.count_observations(minute_prices)
        print(f"  {minute_prices.shape[0]:,} minute bars x {minute_prices.shape[1]} "
              f"tickers = {n_obs:,} observations")
        cands = intraday.screen_universe(minute_prices, corr_threshold=0.7, top_n=15)
        if cands.empty:
            print("  no intraday pairs cleared the correlation pre-screen")
            intra_selected = 0
        else:
            print("  top correlation candidates:")
            print(cands.to_string(index=False))
            # Full cointegration check on the top candidates only.
            intra_selected = 0
            for _, crow in cands.iterrows():
                a_name, b_name = crow["asset_a"], crow["asset_b"]
                ma, mb = minute_prices[a_name], minute_prices[b_name]
                beta, _ = pairs.hedge_ratio(ma, mb)
                eg = pairs.engle_granger(ma, mb)
                hl = pairs.half_life(signals.compute_spread(ma, mb, beta))
                ok = bool(eg["p_value"] < 0.05 and 0.0 < hl < 252.0)
                intra_selected += ok
                print(f"    {a_name}/{b_name}: corr={crow['correlation']:.3f}, "
                      f"coint p={eg['p_value']:.4f}, HL={hl:.1f}d "
                      f"-> {'SELECTED' if ok else 'rejected'}")
            print(f"  intraday pairs selected: {intra_selected}/{len(cands)}")
    except RuntimeError as exc:
        print(f"  intraday fetch failed ({exc}); continuing without it")
        n_obs = 0
        intra_selected = 0

    # ---- Resume metrics -----------------------------------------------------
    resume_metrics = {
        "intraday_observations": n_obs,
        "n_grid_combinations": n_combos,
        "false_signal_suppression": round(q["signal_suppression"], 4),
        "losing_trade_reduction": round(q["losing_trade_reduction"], 4),
        "max_sharpe_improvement": (
            round(max(sharpe_improvements), 4) if sharpe_improvements else 0.0
        ),
    }
    metrics_path = os.path.join(out_dir, "resume_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(resume_metrics, f, indent=2)
    print("\n=== Resume metrics ===")
    print(f"  intraday observations processed: {resume_metrics['intraday_observations']:,}")
    print(f"  parameter combinations searched: {resume_metrics['n_grid_combinations']}")
    print(f"  false-signal suppression:       {resume_metrics['false_signal_suppression']:.1%}")
    print(f"  losing-trade reduction:         {resume_metrics['losing_trade_reduction']:.1%}")
    print(f"  max Sharpe improvement (tuned vs baseline): "
          f"{resume_metrics['max_sharpe_improvement']:+.3f}")
    print(f"\nSaved: {metrics_path}")


if __name__ == "__main__":
    main()
