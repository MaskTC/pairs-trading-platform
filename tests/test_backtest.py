"""Tests for pairs_trading.backtest: exact equity accounting, no look-ahead, costs."""
import numpy as np
import pandas as pd
import pytest

from pairs_trading import backtest, costs


def make_prices():
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    a = pd.Series([100.0, 101, 102, 103, 104, 105, 106, 107, 108, 109], index=idx)
    b = pd.Series([50.0] * 10, index=idx)
    return a, b


class CountingCost(costs.TransactionCostModel):
    def __init__(self):
        super().__init__(cost_bps=5.0, slippage_bps=2.0)
        self.calls = 0
        self.notionals = []

    def cost(self, notional):
        self.calls += 1
        self.notionals.append(notional)
        return super().cost(notional)


def test_no_lookahead_signal_trades_next_bar():
    a, b = make_prices()
    # Enter long at close of bar 2; equity at bar 2 must be untouched.
    target = pd.Series([0, 0, 1, 1, 1, 1, 1, 1, 1, 1], index=a.index)
    res = backtest.run_backtest(
        a, b, beta=1.0, target=target,
        cost_model=costs.TransactionCostModel(cost_bps=0.0, slippage_bps=0.0),
    )
    assert res["equity"].iloc[2] == pytest.approx(10_000.0)
    # The first bar that earns is bar 3: equity[3]-equity[2] == contracts*(S3-S2).
    s = (a - b).to_numpy()
    pos = res["positions"].to_numpy()
    assert res["equity"].iloc[3] - res["equity"].iloc[2] == pytest.approx(
        pos[3] * (s[3] - s[2])
    )


def test_equity_accounting_exact_when_no_trade():
    a, b = make_prices()
    s = (a - b).to_numpy()
    target = pd.Series([0, 0, 1, 1, 1, 0, 0, 0, 0, 0], index=a.index)
    res = backtest.run_backtest(
        a, b, beta=1.0, target=target,
        cost_model=costs.TransactionCostModel(cost_bps=0.0, slippage_bps=0.0),
    )
    eq = res["equity"].to_numpy()
    pos = res["positions"].to_numpy()
    # Whenever the position did not change at the prior close, the equity
    # change must equal position * spread change, exactly.
    for t in range(1, len(eq) - 1):
        if pos[t] == pos[t - 1]:
            assert eq[t] - eq[t - 1] == pytest.approx(pos[t] * (s[t] - s[t - 1]))
    # Round-trip P&L: entered at close 2 (S=52), exited at close 5 (S=55).
    assert len(res["trades"]) == 1
    trade = res["trades"].iloc[0]
    expected_contracts = 10_000.0 / (102.0 + 50.0)
    assert trade["contracts"] == pytest.approx(expected_contracts)
    assert trade["net_pnl"] == pytest.approx(expected_contracts * (55.0 - 52.0))


def test_costs_applied_once_per_position_change():
    a, b = make_prices()
    target = pd.Series([0, 0, 1, 1, 1, -1, -1, 0, 0, 0], index=a.index)
    model = CountingCost()
    res = backtest.run_backtest(a, b, beta=1.0, target=target, cost_model=model)
    # enter (1) + flip close/open (2) + exit (1) = 4 cost events.
    assert model.calls == 4
    assert all(n > 0 for n in model.notionals)
    # Final equity == initial + gross spread P&L - total costs, exactly.
    gross = res["trades"]["gross_pnl"].sum()
    total_cost = sum(model.notionals) * 0.0007
    assert res["equity"].iloc[-1] == pytest.approx(10_000.0 + gross - total_cost)


def test_flip_records_two_round_trips():
    a, b = make_prices()
    target = pd.Series([0, 1, 1, -1, -1, 0, 0, 0, 0, 0], index=a.index)
    res = backtest.run_backtest(
        a, b, beta=1.0, target=target,
        cost_model=costs.TransactionCostModel(cost_bps=0.0, slippage_bps=0.0),
    )
    assert len(res["trades"]) == 2
    assert res["trades"]["side"].tolist() == [1, -1]


def test_flat_target_never_trades():
    a, b = make_prices()
    target = pd.Series([0] * 10, index=a.index)
    res = backtest.run_backtest(a, b, beta=1.0, target=target)
    assert res["n_trades"] == 0
    assert (res["equity"] == 10_000.0).all()
    assert (res["daily_returns"] == 0.0).all()
