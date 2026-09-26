"""Parameter grid search with a train/test split.

For every (window, entry_z, exit_z) combination the strategy is backtested
on the *train* slice only (hedge ratio estimated on train as well, so no
information leaks across the split). The winner is picked by in-sample
Sharpe, then evaluated once on the held-out *test* slice. A large gap
between in-sample and out-of-sample Sharpe raises the overfit flag.
"""
import itertools

import pandas as pd

from . import metrics
from .backtest import run_backtest
from .pairs import hedge_ratio
from .signals import compute_spread, compute_zscore, generate_signals


def detect_overfit(
    train_sharpe: float, test_sharpe: float, train_floor: float = 0.5
) -> bool:
    """Flag likely overfitting: strong in-sample, negative out-of-sample."""
    return bool(train_sharpe > train_floor and test_sharpe < 0.0)


def _summarize(res: dict) -> dict:
    dd = metrics.max_drawdown(res["equity"])
    return {
        "sharpe": metrics.sharpe_ratio(res["daily_returns"]),
        "total_return": metrics.total_return(res["equity"]),
        "annualized_return": metrics.annualized_return(res["equity"]),
        "annualized_vol": metrics.annualized_volatility(res["daily_returns"]),
        "max_drawdown": dd["max_drawdown"],
        "win_rate": metrics.win_rate(res["trades"]),
        "n_trades": res["n_trades"],
        "final_equity": res["final_equity"],
    }


def grid_search(
    price_a,
    price_b,
    windows=(20, 30, 60),
    entry_zs=(1.5, 2.0, 2.5),
    exit_zs=(0.0, 0.5),
    train_frac: float = 0.6,
    cost_model=None,
    initial_capital: float = 10_000.0,
) -> dict:
    """Grid-search signal parameters with out-of-sample validation.

    Returns ``{"results", "best_params", "train_beta", "best_train_sharpe",
    "train_metrics", "test_metrics", "overfit_flag"}``.
    """
    a = pd.Series(price_a, dtype=float)
    b = pd.Series(price_b, dtype=float)
    n = len(a)
    if n < 10:
        raise ValueError("Need at least 10 bars for a train/test split")
    split = int(n * train_frac)
    a_tr, b_tr = a.iloc[:split], b.iloc[:split]
    a_te, b_te = a.iloc[split:], b.iloc[split:]

    beta, _alpha = hedge_ratio(a_tr, b_tr)  # train-only estimation

    rows = []
    for window, entry_z, exit_z in itertools.product(windows, entry_zs, exit_zs):
        z_tr = compute_zscore(
            compute_spread(a_tr, b_tr, beta), window=int(window)
        )
        sig_tr = generate_signals(z_tr, entry_z=float(entry_z), exit_z=float(exit_z))
        res_tr = run_backtest(
            a_tr,
            b_tr,
            beta,
            sig_tr["target"],
            cost_model=cost_model,
            initial_capital=initial_capital,
        )
        summary = _summarize(res_tr)
        rows.append(
            {
                "window": int(window),
                "entry_z": float(entry_z),
                "exit_z": float(exit_z),
                "train_sharpe": summary["sharpe"],
                "train_total_return": summary["total_return"],
                "train_max_drawdown": summary["max_drawdown"],
                "train_win_rate": summary["win_rate"],
                "train_n_trades": summary["n_trades"],
            }
        )
    results = pd.DataFrame(rows)
    best_idx = results["train_sharpe"].idxmax()
    best_params = {
        "window": int(results.loc[best_idx, "window"]),
        "entry_z": float(results.loc[best_idx, "entry_z"]),
        "exit_z": float(results.loc[best_idx, "exit_z"]),
    }
    best_train_sharpe = float(results.loc[best_idx, "train_sharpe"])

    # Winner evaluated once on the held-out slice (train beta frozen).
    z_te = compute_zscore(
        compute_spread(a_te, b_te, beta), window=best_params["window"]
    )
    sig_te = generate_signals(
        z_te, entry_z=best_params["entry_z"], exit_z=best_params["exit_z"]
    )
    res_te = run_backtest(
        a_te,
        b_te,
        beta,
        sig_te["target"],
        cost_model=cost_model,
        initial_capital=initial_capital,
    )
    test_metrics = _summarize(res_te)

    # In-sample metrics of the winner, for the overfit comparison.
    z_tr = compute_zscore(
        compute_spread(a_tr, b_tr, beta), window=best_params["window"]
    )
    sig_tr = generate_signals(
        z_tr, entry_z=best_params["entry_z"], exit_z=best_params["exit_z"]
    )
    res_tr = run_backtest(
        a_tr,
        b_tr,
        beta,
        sig_tr["target"],
        cost_model=cost_model,
        initial_capital=initial_capital,
    )
    train_metrics = _summarize(res_tr)

    return {
        "results": results,
        "best_params": best_params,
        "train_beta": float(beta),
        "best_train_sharpe": best_train_sharpe,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "overfit_flag": detect_overfit(best_train_sharpe, test_metrics["sharpe"]),
    }


BASELINE_PARAMS = {"window": 30, "entry_z": 2.0, "exit_z": 0.5}
"""Textbook-default signal parameters, used as the no-tuning baseline."""


def baseline_vs_optimized(
    price_a,
    price_b,
    best_params: dict,
    beta: float,
    train_frac: float = 0.6,
    cost_model=None,
    initial_capital: float = 10_000.0,
    baseline_params: dict | None = None,
) -> dict:
    """Sharpe improvement of tuned params over the untuned baseline.

    Both parameter sets are evaluated once on the held-out test slice with
    the train-frozen ``beta`` and identical costs, so the difference
    isolates the value of the grid search. Returns
    ``{"baseline_sharpe", "optimized_sharpe", "improvement"}`` where
    ``improvement = optimized_sharpe - baseline_sharpe``.
    """
    if baseline_params is None:
        baseline_params = BASELINE_PARAMS
    a = pd.Series(price_a, dtype=float)
    b = pd.Series(price_b, dtype=float)
    split = int(len(a) * train_frac)
    a_te, b_te = a.iloc[split:], b.iloc[split:]

    def _test_sharpe(params: dict) -> float:
        z = compute_zscore(
            compute_spread(a_te, b_te, float(beta)),
            window=int(params["window"]),
        )
        sig = generate_signals(
            z,
            entry_z=float(params["entry_z"]),
            exit_z=float(params["exit_z"]),
        )
        res = run_backtest(
            a_te,
            b_te,
            float(beta),
            sig["target"],
            cost_model=cost_model,
            initial_capital=float(initial_capital),
        )
        return float(metrics.sharpe_ratio(res["daily_returns"]))

    base_sharpe = _test_sharpe(baseline_params)
    opt_sharpe = _test_sharpe(best_params)
    return {
        "baseline_sharpe": base_sharpe,
        "optimized_sharpe": opt_sharpe,
        "improvement": opt_sharpe - base_sharpe,
    }
