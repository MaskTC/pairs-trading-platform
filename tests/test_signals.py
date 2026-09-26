"""Tests for pairs_trading.signals: z-scores and entry/exit triggers."""
import numpy as np
import pandas as pd
import pytest

from pairs_trading import signals


def test_zscore_matches_manual_computation():
    # spread = [1,2,3,4,5], window=3, ddof=1:
    # idx2: mean=2, std=1 -> z=1; idx3: z=1; idx4: z=1; first two NaN.
    spread = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    z = signals.compute_zscore(spread, window=3)
    assert len(z) == len(spread)  # never silently dropped
    assert np.isnan(z.iloc[0]) and np.isnan(z.iloc[1])
    assert z.iloc[2] == pytest.approx(1.0)
    assert z.iloc[3] == pytest.approx(1.0)
    assert z.iloc[4] == pytest.approx(1.0)


def test_zscore_zero_std_gives_nan_not_inf():
    spread = pd.Series([5.0, 5.0, 5.0, 5.0, 5.0])
    z = signals.compute_zscore(spread, window=3)
    assert not np.isinf(z.fillna(0)).any()
    assert z.iloc[2:].isna().all()


def test_zscore_explicit_min_periods():
    spread = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    z = signals.compute_zscore(spread, window=3, min_periods=3)
    # min_periods == window -> first two bars are NaN by construction.
    assert z.iloc[:2].isna().all()


def test_compute_spread():
    a = pd.Series([10.0, 11.0, 12.0])
    b = pd.Series([5.0, 5.0, 5.0])
    s = signals.compute_spread(a, b, beta=2.0)
    pd.testing.assert_series_equal(s, pd.Series([0.0, 1.0, 2.0]))


def test_generate_signals_entry_exit_logic():
    # z: flat, short entry, hold, exit on |z|<exit_z, long entry, exit, flat.
    z = pd.Series([np.nan, 0.0, 2.5, 2.5, 0.1, -2.5, -0.1, 0.0])
    sig = signals.generate_signals(z, entry_z=2.0, exit_z=0.5)

    assert sig["entry_short"].tolist() == [False, False, True, False, False, False, False, False]
    assert sig["entry_long"].tolist() == [False, False, False, False, False, True, False, False]
    assert sig["target"].tolist() == [0, 0, -1, -1, 0, 1, 0, 0]


def test_generate_signals_exits_on_zero_cross_when_exit_z_zero():
    # With exit_z=0, |z|<0 never fires; the zero-cross rule must exit the trade.
    z = pd.Series([1.5, 0.5, -0.5])
    sig = signals.generate_signals(z, entry_z=1.0, exit_z=0.0)
    assert sig["target"].tolist() == [-1, -1, 0]


def test_generate_signals_no_lookahead_shape():
    # Output aligns 1:1 with input; NaN z-scores never trigger entries.
    z = pd.Series([np.nan] * 5 + [3.0, -3.0])
    sig = signals.generate_signals(z, entry_z=2.0, exit_z=0.5)
    assert len(sig) == len(z)
    assert sig["target"].iloc[:5].tolist() == [0] * 5
    assert sig["target"].iloc[5] == -1
    assert sig["target"].iloc[6] == 1  # flip: short -> long on opposite entry
