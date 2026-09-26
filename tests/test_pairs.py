"""Tests for pairs_trading.pairs: hedge ratio, cointegration, half-life, selection."""
import numpy as np
import pandas as pd
import pytest

from pairs_trading import pairs


def make_cointegrated_pair(n=500, beta=2.5, rho=0.5, seed=42):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(0, 1, n))
    eps = np.zeros(n)
    for t in range(1, n):
        eps[t] = rho * eps[t - 1] + rng.normal(0, 1)
    y = beta * x + eps
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return (
        pd.Series(y, index=idx, name="Y"),
        pd.Series(x, index=idx, name="X"),
    )


def test_hedge_ratio_recovers_true_beta():
    y, x = make_cointegrated_pair()
    beta, alpha = pairs.hedge_ratio(y, x)
    assert abs(beta - 2.5) < 0.10
    assert abs(alpha) < 1.0  # noise has zero mean; intercept should be small


def test_cointegration_accepts_truly_cointegrated_pair():
    y, x = make_cointegrated_pair()
    result = pairs.engle_granger(y, x)
    assert result["p_value"] < 0.05


def test_cointegration_rejects_independent_random_walks():
    rng = np.random.default_rng(7)
    n = 500
    x = np.cumsum(rng.normal(0, 1, n))
    y = np.cumsum(rng.normal(0, 1, n))
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    result = pairs.engle_granger(
        pd.Series(y, index=idx), pd.Series(x, index=idx)
    )
    assert result["p_value"] >= 0.05


def test_half_life_matches_known_ar1():
    # AR(1) with rho=0.95 -> half-life = -ln(2)/ln(0.95) ~= 13.51
    rng = np.random.default_rng(11)
    n = 3000
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.95 * x[t - 1] + rng.normal(0, 1)
    hl = pairs.half_life(pd.Series(x))
    assert abs(hl - 13.51) < 2.0


def test_half_life_random_walk_much_larger_than_mean_reverting():
    # A random walk has no real mean reversion: its estimated half-life
    # should dwarf that of a strongly mean-reverting AR(1) series.
    rng = np.random.default_rng(0)
    n = 2000
    ar = np.zeros(n)
    for t in range(1, n):
        ar[t] = 0.9 * ar[t - 1] + rng.normal(0, 1)
    hl_ar = pairs.half_life(pd.Series(ar))

    rng = np.random.default_rng(3)
    rw = np.cumsum(rng.normal(0, 1, n))
    hl_rw = pairs.half_life(pd.Series(rw))
    assert hl_rw > 5 * hl_ar


def test_half_life_inf_for_explosive_series():
    # Positive feedback (lambda >= 0) -> no mean reversion -> inf.
    explosive = pd.Series([1.01**t for t in range(200)])
    assert pairs.half_life(explosive) == np.inf


def test_find_pairs_selects_only_cointegrated():
    n = 400
    rng = np.random.default_rng(21)
    a = np.cumsum(rng.normal(0, 1, n))
    eps = np.zeros(n)
    for t in range(1, n):
        eps[t] = 0.4 * eps[t - 1] + rng.normal(0, 0.5)
    b = 1.8 * a + eps  # cointegrated with A
    c = np.cumsum(rng.normal(0, 1, n))  # independent random walk
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    prices = pd.DataFrame({"A": a, "B": b, "C": c}, index=idx)

    found = pairs.find_pairs(prices, corr_threshold=0.5)

    ab = found[(found.asset_a == "A") & (found.asset_b == "B")].iloc[0]
    assert bool(ab["selected"])
    assert ab["coint_p_value"] < 0.05
    assert ab["half_life"] > 0

    # Pairs involving the independent walk must be rejected (not merely correlated).
    for _, row in found.iterrows():
        if "C" in (row["asset_a"], row["asset_b"]):
            assert not bool(row["selected"])
