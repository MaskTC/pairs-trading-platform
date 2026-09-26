"""Tests for pairs_trading.costs and pairs_trading.metrics."""
import numpy as np
import pandas as pd
import pytest

from pairs_trading import costs, metrics


# ---------- costs ----------


def test_cost_arithmetic_exact():
    model = costs.TransactionCostModel(cost_bps=5.0, slippage_bps=2.0)
    # 7 bps of 10_000 = 7.0, exactly.
    assert model.cost(10_000.0) == pytest.approx(7.0)
    assert model.cost(0.0) == 0.0


def test_cost_default_is_five_bps_per_side():
    model = costs.TransactionCostModel()
    assert model.cost_bps == pytest.approx(5.0)


# ---------- metrics ----------


def test_max_drawdown_on_handcrafted_curve():
    equity = pd.Series([100.0, 120.0, 110.0, 90.0, 100.0])
    dd = metrics.max_drawdown(equity)
    assert dd["max_drawdown"] == pytest.approx(-0.25)  # (90-120)/120
    assert dd["peak_index"] == 1
    assert dd["trough_index"] == 3


def test_max_drawdown_zero_when_monotonic_up():
    equity = pd.Series([100.0, 101.0, 102.0])
    assert metrics.max_drawdown(equity)["max_drawdown"] == pytest.approx(0.0)


def test_sharpe_zero_volatility_returns_zero():
    r = pd.Series([0.01] * 100)
    assert metrics.sharpe_ratio(r) == 0.0


def test_sharpe_matches_formula():
    r = pd.Series([0.01, -0.005, 0.02, 0.003, -0.01, 0.015])
    expected = r.mean() / r.std(ddof=1) * np.sqrt(252)
    assert metrics.sharpe_ratio(r) == pytest.approx(expected)


def test_total_and_annualized_return():
    n = 252
    equity = pd.Series(100.0 * (1.10 ** (np.arange(n + 1) / n)))
    assert metrics.total_return(equity) == pytest.approx(0.10)
    assert metrics.annualized_return(equity) == pytest.approx(0.10, abs=1e-6)


def test_win_rate_on_round_trips():
    trades = pd.DataFrame({"net_pnl": [10.0, -5.0, 3.0, -1.0]})
    assert metrics.win_rate(trades) == pytest.approx(0.5)


def test_win_rate_empty_is_nan():
    trades = pd.DataFrame({"net_pnl": []})
    assert np.isnan(metrics.win_rate(trades))


def test_cumulative_returns():
    r = pd.Series([0.10, -0.10])
    cum = metrics.cumulative_returns(r)
    assert cum.iloc[-1] == pytest.approx(1.1 * 0.9 - 1)
