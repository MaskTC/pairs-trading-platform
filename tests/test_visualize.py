"""Tests for pairs_trading.visualize: plots are written to disk."""
import numpy as np
import pandas as pd

from pairs_trading import visualize


def test_plot_spread_writes_png(tmp_path):
    idx = pd.date_range("2020-01-01", periods=100, freq="D")
    spread = pd.Series(np.sin(np.linspace(0, 6, 100)), index=idx)
    z = pd.Series(np.sin(np.linspace(0, 6, 100)) * 1.5, index=idx)
    path = tmp_path / "spread.png"
    visualize.plot_spread(spread, z, entry_z=2.0, exit_z=0.5, path=str(path))
    assert path.exists() and path.stat().st_size > 0


def test_plot_equity_writes_png(tmp_path):
    idx = pd.date_range("2020-01-01", periods=100, freq="D")
    equity = pd.Series(10000 * (1 + np.linspace(0, 0.1, 100)), index=idx)
    benchmark = pd.Series(10000 * (1 + np.linspace(0, 0.05, 100)), index=idx)
    path = tmp_path / "equity.png"
    visualize.plot_equity(equity, benchmark, path=str(path))
    assert path.exists() and path.stat().st_size > 0
