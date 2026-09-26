"""Tests for pairs_trading.intraday: 1-minute ingestion, counting, screening."""
import numpy as np
import pandas as pd
import pytest

from pairs_trading import intraday


def _fake_minute_frame():
    # Mimics yfinance 1m output: MultiIndex (field, ticker) columns,
    # minute DatetimeIndex, one missing minute for BBB.
    idx = pd.date_range("2026-09-01 09:30", periods=8, freq="min")
    cols = pd.MultiIndex.from_tuples([("Close", "AAA"), ("Close", "BBB")])
    vals = np.array(
        [
            [10.0, 20.0],
            [10.1, 20.2],
            [10.2, np.nan],  # BBB missing this minute
            [10.1, 20.1],
            [10.3, 20.4],
            [10.2, 20.3],
            [10.4, 20.5],
            [10.5, 20.6],
        ]
    )
    return pd.DataFrame(vals, index=idx, columns=cols)


def test_fetch_intraday_uses_yfinance_and_aligns(monkeypatch):
    raw = _fake_minute_frame()

    def fake_download(tickers, start, end, interval, auto_adjust=True, progress=False):
        assert interval == "1m"
        return raw

    import yfinance as yf

    monkeypatch.setattr(yf, "download", fake_download)
    out = intraday.fetch_intraday(["AAA", "BBB"], days=16, interval="1m", pause=0)

    assert list(out.columns) == ["AAA", "BBB"]
    assert len(out) == 8
    assert not out.isna().any().any()
    # BBB's missing minute is forward-filled from the prior bar.
    assert out["BBB"].iloc[2] == out["BBB"].iloc[1] == 20.2


def test_fetch_intraday_raises_when_everything_empty(monkeypatch):
    import yfinance as yf

    monkeypatch.setattr(
        yf, "download", lambda *a, **k: pd.DataFrame()
    )
    try:
        intraday.fetch_intraday(["AAA", "BBB"])
    except RuntimeError as exc:
        assert "no intraday data" in str(exc)
    else:  # pragma: no cover - must raise
        raise AssertionError("expected RuntimeError")


def test_fetch_intraday_batches_requests(monkeypatch):
    calls = []

    def fake_download(tickers, start, end, interval, auto_adjust=True, progress=False):
        calls.append((list(tickers), start, end))
        idx = pd.date_range("2026-09-01 09:30", periods=4, freq="min")
        cols = pd.MultiIndex.from_tuples([("Close", t) for t in tickers])
        return pd.DataFrame(
            np.ones((4, len(tickers))), index=idx, columns=cols
        )

    import yfinance as yf

    monkeypatch.setattr(yf, "download", fake_download)
    # 16 days -> 2 chunks of 8 days; 5 tickers, batch_size=2 -> 3 batches.
    out = intraday.fetch_intraday(
        ["A", "B", "C", "D", "E"], days=16, batch_size=2, pause=0
    )

    assert len(calls) == 2 * 3
    assert all(len(tickers) <= 2 for tickers, _, _ in calls)
    assert {t for tickers, _, _ in calls for t in tickers} == {"A", "B", "C", "D", "E"}
    for _, s, e in calls:
        window = (
            pd.Timestamp(e) - pd.Timestamp(s)
        ).days
        assert window <= intraday.MAX_CHUNK_DAYS_1M
    assert list(out.columns) == ["A", "B", "C", "D", "E"]


def test_fetch_intraday_clamps_1m_lookback(monkeypatch):
    seen = []

    def fake_download(tickers, start, end, interval, auto_adjust=True, progress=False):
        seen.append((start, end))
        return pd.DataFrame()

    import yfinance as yf

    monkeypatch.setattr(yf, "download", fake_download)
    try:
        intraday.fetch_intraday(["AAA"], days=90, pause=0)
    except RuntimeError:
        pass
    # 90d requested, but 1m lookback clamps to 30d -> earliest start is 30d back.
    earliest = min(pd.Timestamp(s) for s, _ in seen)
    latest = max(pd.Timestamp(e) for _, e in seen)
    assert (latest - earliest).days <= intraday.MAX_LOOKBACK_DAYS_1M


def test_count_observations_counts_non_null_cells():
    idx = pd.date_range("2026-09-01", periods=5, freq="D")
    prices = pd.DataFrame(
        {"A": [1.0, 2.0, np.nan, 4.0, 5.0], "B": [1.0, 2.0, 3.0, 4.0, 5.0]},
        index=idx,
    )
    assert intraday.count_observations(prices) == 9


def test_screen_universe_sorts_by_correlation_and_caps():
    rng = np.random.default_rng(0)
    n = 500
    idx = pd.date_range("2026-01-01", periods=n, freq="min")
    base = np.cumsum(rng.normal(0, 1, n))
    prices = pd.DataFrame(
        {
            "A": 100 + base,
            "B": 100 + base + rng.normal(0, 0.5, n),  # tight pair
            "C": 100 + np.cumsum(rng.normal(0, 1, n)),  # unrelated
            "D": 50 + base * 0.8 + rng.normal(0, 0.8, n),  # looser pair
        },
        index=idx,
    )
    out = intraday.screen_universe(prices, corr_threshold=0.5, top_n=2)
    assert len(out) == 2
    # Tight pair ranks first.
    assert set(out.iloc[0][["asset_a", "asset_b"]]) == {"A", "B"}
    assert out["correlation"].is_monotonic_decreasing
    assert (out["correlation"] >= 0.5).all()


def test_screen_universe_empty_when_nothing_passes():
    rng = np.random.default_rng(1)
    n = 200
    idx = pd.date_range("2026-01-01", periods=n, freq="min")
    prices = pd.DataFrame(
        {t: 100 + np.cumsum(rng.normal(0, 1, n)) for t in "ABCD"}, index=idx
    )
    out = intraday.screen_universe(prices, corr_threshold=0.999, top_n=10)
    assert out.empty


def test_fetch_intraday_stacks_chunks_vertically(monkeypatch):
    """Regression test: batches join along columns, chunks stack along rows.

    Two earlier implementations were silently wrong:
    (1) concatenating everything along axis=1 duplicated the ticker columns,
    so the column-dedup kept only the first chunk's data and forward-filled
    the rest; (2) concatenating everything along axis=0 unioned the disjoint
    ticker sets, so after the index-dedup most tickers were entirely NaN and
    align_prices dropped every row.
    """
    import yfinance as yf

    def fake_download(tickers, start, end, interval, auto_adjust=True, progress=False):
        # Each chunk returns a distinct 2-bar window; the price level encodes
        # the chunk start date and the ticker, so misplaced data is visible.
        base = pd.Timestamp(start)
        idx = pd.DatetimeIndex([base, base + pd.Timedelta(minutes=1)])
        level = float((base - pd.Timestamp("2026-01-01")).days)
        cols = pd.MultiIndex.from_tuples([("Close", t) for t in tickers])
        data = np.column_stack([
            np.full(2, 100.0 + level + (0.5 if t == "BBB" else 0.0))
            for t in tickers
        ])
        return pd.DataFrame(data, index=idx, columns=cols)

    monkeypatch.setattr(yf, "download", fake_download)
    # 2 chunks x 2 batches (batch_size=1) -> 4 bars x 2 tickers.
    out = intraday.fetch_intraday(
        ["AAA", "BBB"], days=16, chunk_days=8, batch_size=1, pause=0
    )

    assert out.shape == (4, 2)
    assert list(out.columns) == ["AAA", "BBB"]
    assert not out.index.has_duplicates
    assert out.index.is_monotonic_increasing
    # Bars within a chunk share the chunk's level; the later chunk's level
    # is higher -- i.e. chunk 2's data survived for BOTH tickers.
    assert out["AAA"].iloc[0] == out["AAA"].iloc[1]
    assert out["AAA"].iloc[2] == out["AAA"].iloc[3]
    assert out["AAA"].iloc[2] > out["AAA"].iloc[1]
    assert out["BBB"].iloc[0] == out["AAA"].iloc[0] + 0.5
    assert out["BBB"].iloc[2] == out["AAA"].iloc[2] + 0.5


def test_fetch_intraday_drops_tickers_missing_from_a_chunk(monkeypatch):
    """Tickers absent from any chunk are excluded (their leading gaps would
    otherwise force align_prices to drop every row)."""
    import yfinance as yf

    calls = {"n": 0}

    def fake_download(tickers, start, end, interval, auto_adjust=True, progress=False):
        calls["n"] += 1
        # BBB is missing from the first chunk's response.
        keep = ["AAA"] if calls["n"] == 1 else ["AAA", "BBB"]
        base = pd.Timestamp(start)
        idx = pd.DatetimeIndex([base, base + pd.Timedelta(minutes=1)])
        cols = pd.MultiIndex.from_tuples([("Close", t) for t in keep])
        return pd.DataFrame(
            np.full((2, len(keep)), 100.0), index=idx, columns=cols
        )

    monkeypatch.setattr(yf, "download", fake_download)
    out = intraday.fetch_intraday(
        ["AAA", "BBB"], days=16, chunk_days=8, batch_size=2, pause=0
    )
    assert list(out.columns) == ["AAA"]
    assert out.shape == (4, 1)


def test_fetch_intraday_retries_empty_responses(monkeypatch):
    import yfinance as yf

    calls = {"n": 0}

    def fake_download(tickers, start, end, interval, auto_adjust=True, progress=False):
        calls["n"] += 1
        if calls["n"] == 1:
            return pd.DataFrame()  # first attempt empty
        idx = pd.DatetimeIndex([pd.Timestamp(start)])
        cols = pd.MultiIndex.from_tuples([("Close", t) for t in tickers])
        return pd.DataFrame(np.ones((1, len(tickers))), index=idx, columns=cols)

    monkeypatch.setattr(yf, "download", fake_download)
    out = intraday.fetch_intraday(["AAA"], days=8, chunk_days=8, pause=0)
    assert calls["n"] == 2
    assert out.shape == (1, 1)


def test_fetch_intraday_raises_when_no_ticker_covers_all_chunks(monkeypatch):
    import yfinance as yf

    calls = {"n": 0}

    def fake_download(tickers, start, end, interval, auto_adjust=True, progress=False):
        # Each chunk returns a different ticker -> empty intersection.
        calls["n"] += 1
        t = "AAA" if calls["n"] == 1 else "BBB"
        idx = pd.DatetimeIndex([pd.Timestamp(start)])
        cols = pd.MultiIndex.from_tuples([("Close", t)])
        return pd.DataFrame(np.ones((1, 1)), index=idx, columns=cols)

    monkeypatch.setattr(yf, "download", fake_download)
    with pytest.raises(RuntimeError, match="every intraday chunk"):
        intraday.fetch_intraday(
            ["AAA", "BBB"], days=16, chunk_days=8, batch_size=2, pause=0
        )


def test_fetch_intraday_drops_all_nan_ticker_columns(monkeypatch):
    """A ticker whose column is all-NaN in a chunk (yfinance failed/delisted
    symbol) must not poison the panel: without the drop, align_prices would
    discard every row."""
    import yfinance as yf

    def fake_download(tickers, start, end, interval, auto_adjust=True, progress=False):
        base = pd.Timestamp(start)
        idx = pd.DatetimeIndex([base, base + pd.Timedelta(minutes=1)])
        cols = pd.MultiIndex.from_tuples([("Close", t) for t in tickers])
        data = np.full((2, len(tickers)), 100.0)
        if "DEAD" in tickers:
            data[:, tickers.index("DEAD")] = np.nan
        return pd.DataFrame(data, index=idx, columns=cols)

    monkeypatch.setattr(yf, "download", fake_download)
    out = intraday.fetch_intraday(
        ["AAA", "DEAD"], days=16, chunk_days=8, batch_size=2, pause=0
    )
    assert list(out.columns) == ["AAA"]
    assert out.shape == (4, 1)
    assert out.notna().all().all()
