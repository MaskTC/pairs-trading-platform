"""Tests for pairs_trading.data: fetching, calendar alignment, ffill, returns."""
import numpy as np
import pandas as pd
import pytest

from pairs_trading import data


def test_align_prices_ffills_mismatched_calendars():
    # Asset A trades every day; asset B is missing two days in the middle.
    idx_a = pd.date_range("2020-01-01", periods=10, freq="D")
    idx_b = idx_a.delete([4, 5])  # missing 2020-01-05 and 2020-01-06
    prices = pd.DataFrame({"A": np.arange(10, dtype=float)}, index=idx_a)
    prices["B"] = pd.Series(np.arange(8, dtype=float) * 2, index=idx_b)

    aligned = data.align_prices(prices)

    # Both columns share the exact same (union) index afterwards.
    assert aligned.index.equals(idx_a)
    assert not aligned.isna().any().any()
    # Forward-filled: B on 01-05 and 01-06 equals B's 01-04 close.
    assert aligned.loc["2020-01-05", "B"] == aligned.loc["2020-01-04", "B"]
    assert aligned.loc["2020-01-06", "B"] == aligned.loc["2020-01-04", "B"]
    # Untouched values are identical to the input.
    assert aligned.loc["2020-01-04", "B"] == 3 * 2.0


def test_align_prices_drops_leading_gaps():
    # Asset C lists two days after A/B: leading NaNs cannot be ffilled -> rows dropped.
    idx = pd.date_range("2020-01-01", periods=6, freq="D")
    prices = pd.DataFrame(
        {
            "A": np.arange(6, dtype=float),
            "B": np.arange(6, dtype=float),
            "C": pd.Series([10.0, 11.0, 12.0, 13.0], index=idx[2:]),
        },
        index=idx,
    )
    aligned = data.align_prices(prices)
    assert aligned.index[0] == idx[2]
    assert not aligned.isna().any().any()
    assert len(aligned) == 4


def test_compute_returns_matches_pct_change():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    prices = pd.DataFrame({"A": [100.0, 101.0, 102.0, 101.0, 103.0]}, index=idx)
    rets = data.compute_returns(prices)
    expected = prices.pct_change()
    pd.testing.assert_frame_equal(rets, expected)
    assert np.isnan(rets.iloc[0, 0])


def test_fetch_prices_uses_yfinance_and_aligns(monkeypatch):
    # Mock yfinance: multi-ticker download with mismatched calendars.
    idx_a = pd.date_range("2020-01-01", periods=6, freq="D")
    idx_b = idx_a.delete([2])
    cols = pd.MultiIndex.from_tuples([("Close", "AAA"), ("Close", "BBB")])
    raw = pd.DataFrame(
        [
            [10.0, 20.0],
            [11.0, 21.0],
            [12.0, np.nan],
            [13.0, 23.0],
            [14.0, 24.0],
            [15.0, 25.0],
        ],
        index=idx_a,
        columns=cols,
    )
    # Simulate the missing trading day for BBB by dropping that row's BBB value
    # (already NaN above); fetch_prices must still align/ffill.

    def fake_download(tickers, start, end, auto_adjust=True, progress=False):
        assert set(tickers) == {"AAA", "BBB"}
        return raw

    import yfinance as yf

    monkeypatch.setattr(yf, "download", fake_download)
    out = data.fetch_prices(["AAA", "BBB"], "2020-01-01", "2020-01-07")

    assert list(out.columns) == ["AAA", "BBB"]
    assert out.index.equals(idx_a)
    assert not out.isna().any().any()
    # BBB's missing day is forward-filled from the prior close.
    assert out.loc["2020-01-03", "BBB"] == 21.0
