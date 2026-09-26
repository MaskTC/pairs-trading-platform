"""Tests for pairs_trading.datasets: custom CSV layouts, validation, synthetic data."""
import numpy as np
import pandas as pd
import pytest

from pairs_trading import datasets


def _long_csv_text():
    return (
        "datetime,symbol,close,volume\n"
        "2026-01-02 09:30:00,AAA,100.0,1000\n"
        "2026-01-02 09:31:00,AAA,100.5,1100\n"
        "2026-01-02 09:30:00,BBB,50.0,2000\n"
        "2026-01-02 09:31:00,BBB,50.2,2100\n"
    )


def test_read_long_csv(tmp_path):
    p = tmp_path / "prices.csv"
    p.write_text(_long_csv_text())
    out = datasets.read_long_csv(p)
    assert list(out.columns) == ["AAA", "BBB"]
    assert len(out) == 2
    assert out.loc["2026-01-02 09:30:00", "AAA"] == pytest.approx(100.0)
    assert out.loc["2026-01-02 09:31:00", "BBB"] == pytest.approx(50.2)


def test_read_long_csv_missing_column(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("datetime,symbol\n2026-01-02,AAA\n")
    with pytest.raises(ValueError, match="missing required columns"):
        datasets.read_long_csv(p)


def test_read_long_csv_rejects_duplicate_bars(tmp_path):
    p = tmp_path / "dup.csv"
    p.write_text(
        "datetime,symbol,close\n"
        "2026-01-02 09:30:00,AAA,100.0\n"
        "2026-01-02 09:30:00,AAA,100.1\n"
    )
    with pytest.raises(ValueError, match="duplicate"):
        datasets.read_long_csv(p)


def test_read_long_csv_rejects_bad_datetimes(tmp_path):
    p = tmp_path / "badt.csv"
    p.write_text("datetime,symbol,close\nnot-a-date,AAA,100.0\n")
    with pytest.raises(ValueError, match="Could not parse datetime"):
        datasets.read_long_csv(p)


def test_read_wide_csv(tmp_path):
    p = tmp_path / "wide.csv"
    p.write_text(
        "datetime,AAA,BBB\n"
        "2026-01-02,100.0,50.0\n"
        "2026-01-03,101.0,49.5\n"
    )
    out = datasets.read_wide_csv(p)
    assert list(out.columns) == ["AAA", "BBB"]
    assert out.loc["2026-01-03", "AAA"] == pytest.approx(101.0)


def test_read_wide_csv_forward_fills_gaps(tmp_path):
    p = tmp_path / "wide_gap.csv"
    p.write_text(
        "datetime,AAA,BBB\n"
        "2026-01-02,100.0,50.0\n"
        "2026-01-03,101.0,\n"
    )
    out = datasets.read_wide_csv(p)
    assert not out.isna().any().any()
    assert out.loc["2026-01-03", "BBB"] == pytest.approx(50.0)


def test_read_ticker_dir(tmp_path):
    (tmp_path / "AAA.csv").write_text(
        "datetime,close\n2026-01-02,100.0\n2026-01-03,101.0\n"
    )
    (tmp_path / "BBB.csv").write_text(
        "datetime,close\n2026-01-02,50.0\n2026-01-03,49.5\n"
    )
    out = datasets.read_ticker_dir(tmp_path)
    assert set(out.columns) == {"AAA", "BBB"}
    assert len(out) == 2


def test_read_ticker_dir_empty_directory(tmp_path):
    with pytest.raises(ValueError, match="No files matching"):
        datasets.read_ticker_dir(tmp_path)


def test_validate_panel_rejects_non_datetime_index():
    with pytest.raises(ValueError, match="DatetimeIndex"):
        datasets.validate_panel(pd.DataFrame({"A": [1.0, 2.0]}))


def test_validate_panel_rejects_duplicate_timestamps():
    idx = pd.DatetimeIndex(["2026-01-02", "2026-01-02"])
    with pytest.raises(ValueError, match="Duplicate timestamps"):
        datasets.validate_panel(pd.DataFrame({"A": [1.0, 2.0]}, index=idx))


def test_validate_panel_rejects_non_numeric():
    idx = pd.date_range("2026-01-02", periods=2)
    with pytest.raises(ValueError, match="Non-numeric"):
        datasets.validate_panel(pd.DataFrame({"A": ["x", "y"]}, index=idx))


def test_validate_panel_sorts_index():
    idx = pd.DatetimeIndex(["2026-01-03", "2026-01-02"])
    out = datasets.validate_panel(pd.DataFrame({"A": [2.0, 1.0]}, index=idx))
    assert out.index.is_monotonic_increasing


def test_write_example_long_csv_round_trip(tmp_path):
    p = tmp_path / "example.csv"
    datasets.write_example_long_csv(p)
    out = datasets.read_long_csv(p)
    assert set(out.columns) == {"AAA", "BBB"}
    assert len(out) == 390 * 3


def test_generate_synthetic_minutes_shape_and_seed():
    a = datasets.generate_synthetic_minutes(["AAA", "BBB"], trading_days=5, seed=7)
    b = datasets.generate_synthetic_minutes(["AAA", "BBB"], trading_days=5, seed=7)
    assert a.shape == (5 * 390, 2)
    pd.testing.assert_frame_equal(a, b)
    assert not a.isna().any().any()
    assert (a > 0).all().all()
    # Different seed -> different data.
    c = datasets.generate_synthetic_minutes(["AAA", "BBB"], trading_days=5, seed=8)
    assert not a.equals(c)


def test_generate_synthetic_minutes_intraday_index():
    a = datasets.generate_synthetic_minutes(["AAA"], trading_days=2, seed=1)
    assert isinstance(a.index, pd.DatetimeIndex)
    # Bars fall inside regular trading hours.
    assert ((a.index.time >= pd.Timestamp("09:30").time())
            & (a.index.time < pd.Timestamp("16:00").time())).all()
