"""Tests for pairs_trading.optimize: grid search, selection, train/test split."""
import numpy as np
import pandas as pd
import pytest

from pairs_trading import backtest, optimize, signals


def make_mean_reverting_prices(n=600, seed=123):
    # Cointegrated pair with a genuinely mean-reverting spread:
    # B is a random walk, A = B + AR(1) noise, so spread ~= AR(1).
    rng = np.random.default_rng(seed)
    b_walk = np.cumsum(rng.normal(0, 0.3, n))
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = 0.85 * spread[t - 1] + rng.normal(0, 1)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    a = pd.Series(100.0 + b_walk + spread, index=idx)
    b = pd.Series(100.0 + b_walk, index=idx)
    return a, b


def test_grid_search_covers_full_grid_and_picks_argmax():
    a, b = make_mean_reverting_prices()
    out = optimize.grid_search(
        a, b,
        windows=[20, 30],
        entry_zs=[1.5, 2.0],
        exit_zs=[0.0, 0.5],
        train_frac=0.6,
    )
    results = out["results"]
    assert len(results) == 2 * 2 * 2
    # The reported best params are genuinely the argmax of in-sample Sharpe.
    best_idx = results["train_sharpe"].idxmax()
    for col in ("window", "entry_z", "exit_z"):
        assert out["best_params"][col] == results.loc[best_idx, col]


def test_test_metrics_use_only_held_out_data():
    a, b = make_mean_reverting_prices()
    out = optimize.grid_search(
        a, b, windows=[20, 30], entry_zs=[2.0], exit_zs=[0.5], train_frac=0.6
    )
    bp = out["best_params"]
    n = len(a)
    split = int(n * 0.6)
    a_test, b_test = a.iloc[split:], b.iloc[split:]
    beta = out["train_beta"]
    z = signals.compute_zscore(
        signals.compute_spread(a_test, b_test, beta), window=bp["window"]
    )
    sig = signals.generate_signals(z, entry_z=bp["entry_z"], exit_z=bp["exit_z"])
    res = backtest.run_backtest(a_test, b_test, beta, sig["target"])
    from pairs_trading import metrics

    assert out["test_metrics"]["sharpe"] == pytest.approx(
        metrics.sharpe_ratio(res["daily_returns"])
    )


def test_detect_overfit_flags_divergence():
    assert optimize.detect_overfit(2.0, -0.5) is True
    assert optimize.detect_overfit(1.5, 1.2) is False
    assert optimize.detect_overfit(0.2, -0.1) is False  # weak in-sample: not "overfit"


def test_profitable_on_mean_reverting_synthetic():
    a, b = make_mean_reverting_prices()
    out = optimize.grid_search(
        a, b,
        windows=[20, 30, 60],
        entry_zs=[1.5, 2.0, 2.5],
        exit_zs=[0.0, 0.5],
        train_frac=0.6,
    )
    assert out["best_train_sharpe"] > 0
    assert out["test_metrics"]["n_trades"] >= 0  # runs cleanly out of sample


def test_grid_search_handles_160_combinations():
    # The resume claims 150+ parameter combinations: 8 x 5 x 4 = 160.
    a, b = make_mean_reverting_prices(n=250, seed=7)
    out = optimize.grid_search(
        a, b,
        windows=[10, 15, 20, 30, 45, 60, 90, 120],
        entry_zs=[1.0, 1.5, 2.0, 2.5, 3.0],
        exit_zs=[0.0, 0.25, 0.5, 0.75],
        train_frac=0.6,
    )
    assert len(out["results"]) == 160
    best_idx = out["results"]["train_sharpe"].idxmax()
    for col in ("window", "entry_z", "exit_z"):
        assert out["best_params"][col] == out["results"].loc[best_idx, col]


def test_baseline_vs_optimized_reports_improvement():
    a, b = make_mean_reverting_prices(n=300, seed=11)
    gs = optimize.grid_search(
        a, b, windows=[20, 30], entry_zs=[1.5, 2.0], exit_zs=[0.0, 0.5]
    )
    out = optimize.baseline_vs_optimized(
        a, b, gs["best_params"], gs["train_beta"], train_frac=0.6
    )
    assert set(out) == {"baseline_sharpe", "optimized_sharpe", "improvement"}
    assert out["improvement"] == pytest.approx(
        out["optimized_sharpe"] - out["baseline_sharpe"]
    )
    for v in out.values():
        assert isinstance(v, float)


def test_baseline_params_constant_is_documented_default():
    assert optimize.BASELINE_PARAMS == {"window": 30, "entry_z": 2.0, "exit_z": 0.5}
