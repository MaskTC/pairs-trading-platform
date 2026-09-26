"""Tests for pairs_trading.quality: naive vs filtered signal comparison."""
import numpy as np
import pandas as pd
import pytest

from pairs_trading import quality
from pairs_trading.costs import TransactionCostModel


def _prices(seed=42, n=400):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    # A/B: cointegrated-ish (shared walk + mean-reverting spread).
    walk = np.cumsum(rng.normal(0, 0.5, n))
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = 0.8 * spread[t - 1] + rng.normal(0, 1.0)
    # C: independent walk (uncorrelated with A/B).
    other = np.cumsum(rng.normal(0, 0.5, n))
    return pd.DataFrame(
        {
            "A": 100.0 + walk + spread,
            "B": 100.0 + walk,
            "C": 50.0 + other,
        },
        index=idx,
    )


def _fake_screen(selected_pairs):
    """Build a fake find_pairs result over A/B/C with controlled selection."""

    def fake(prices, corr_threshold=0.7, p_threshold=0.05, max_half_life=252.0):
        rows = []
        for a, b in [("A", "B"), ("A", "C"), ("B", "C")]:
            rows.append(
                {
                    "asset_a": a,
                    "asset_b": b,
                    "correlation": 0.85,
                    "beta": 1.0,
                    "alpha": 0.0,
                    "coint_t_stat": -4.0,
                    "coint_p_value": 0.001,
                    "half_life": 20.0,
                    "selected": (a, b) in selected_pairs,
                }
            )
        return pd.DataFrame(rows)

    return fake


def test_screen_comparison_suppression_is_consistent(monkeypatch):
    prices = _prices()
    monkeypatch.setattr(quality, "find_pairs", _fake_screen({("A", "B")}))
    out = quality.screen_comparison(
        prices, cost_model=TransactionCostModel(), initial_capital=10_000.0
    )
    assert out["naive"]["n_pairs"] == 3
    assert out["filtered"]["n_pairs"] == 1
    assert out["filtered"]["total_signals"] <= out["naive"]["total_signals"]
    n, f = out["naive"]["total_signals"], out["filtered"]["total_signals"]
    assert out["signal_suppression"] == pytest.approx(1 - f / n)
    assert 0.0 <= out["signal_suppression"] <= 1.0
    assert 0.0 <= out["losing_trade_reduction"] <= 1.0


def test_screen_comparison_full_suppression_when_nothing_selected(monkeypatch):
    prices = _prices()
    monkeypatch.setattr(quality, "find_pairs", _fake_screen(set()))
    out = quality.screen_comparison(prices)
    assert out["filtered"]["n_pairs"] == 0
    assert out["filtered"]["total_signals"] == 0
    if out["naive"]["total_signals"] > 0:
        assert out["signal_suppression"] == pytest.approx(1.0)


def test_screen_comparison_handles_empty_screen(monkeypatch):
    prices = _prices()

    def fake_empty(prices, corr_threshold=0.7, p_threshold=0.05, max_half_life=252.0):
        return pd.DataFrame(
            columns=[
                "asset_a", "asset_b", "correlation", "beta", "alpha",
                "coint_t_stat", "coint_p_value", "half_life", "selected",
            ]
        )

    monkeypatch.setattr(quality, "find_pairs", fake_empty)
    out = quality.screen_comparison(prices)
    assert out["naive"]["total_signals"] == 0
    assert out["signal_suppression"] == 0.0
    assert out["losing_trade_reduction"] == 0.0


def test_screen_comparison_real_screen_runs():
    # Smoke test with the real pair screen on synthetic data.
    prices = _prices()
    out = quality.screen_comparison(prices)
    assert set(out) == {
        "naive", "filtered", "signal_suppression", "losing_trade_reduction"
    }
