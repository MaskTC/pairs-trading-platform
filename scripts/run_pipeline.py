"""End-to-end pipeline: fetch -> find pairs -> grid search -> backtest -> plots.

Usage:
    .venv/bin/python scripts/run_pipeline.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from pairs_trading import backtest, costs, data, metrics, optimize, pairs, signals, visualize

TICKERS = ["KO", "PEP", "XOM", "CVX"]
BENCHMARK = "SPY"
START = "2020-01-01"
END = "2025-01-01"
INITIAL_CAPITAL = 10_000.0

WINDOWS = [20, 30, 60]
ENTRY_ZS = [1.5, 2.0, 2.5]
EXIT_ZS = [0.0, 0.5]


def main() -> None:
    out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Fetching {TICKERS + [BENCHMARK]} from {START} to {END} ...")
    prices = data.fetch_prices(TICKERS + [BENCHMARK], START, END)
    print(f"Aligned panel: {prices.shape[0]} trading days, "
          f"{prices.index[0].date()} -> {prices.index[-1].date()}")

    bench = prices[BENCHMARK]
    px = prices[TICKERS]

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
    for _, row in selected.iterrows():
        a_name, b_name = row["asset_a"], row["asset_b"]
        pair = f"{a_name}/{b_name}"
        print(f"\n=== {pair}: grid search (train 60% / test 40%) ===")
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

        # Full-sample signals with the winning params for the spread plot.
        spread = signals.compute_spread(a, b, opt["train_beta"])
        z = signals.compute_zscore(spread, window=bp["window"])
        sig = signals.generate_signals(z, entry_z=bp["entry_z"], exit_z=bp["exit_z"])
        full = backtest.run_backtest(
            a, b, opt["train_beta"], sig["target"],
            cost_model=cost_model, initial_capital=INITIAL_CAPITAL,
        )
        spread_path = os.path.join(out_dir, f"spread_{a_name}_{b_name}.png")
        visualize.plot_spread(
            spread, z, bp["entry_z"], bp["exit_z"], spread_path,
            title=f"{pair} spread",
        )

        # Out-of-sample equity vs SPY buy-and-hold on the test slice.
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
            title=f"{pair} out-of-sample equity vs SPY buy-and-hold",
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
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out_dir, "summary.csv")
    summary.to_csv(summary_path, index=False)
    print("\n=== Out-of-sample summary ===")
    print(summary.to_string(index=False))
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
