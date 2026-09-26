"""High-frequency (intraday) data ingestion and screening.

Fetches 1-minute bars via yfinance so the platform can process hundreds of
thousands of observations (7,800 bars x 80 tickers = 624,000 (bar, ticker)
observations in the reference run). Minute bars go through the same
cleaning contract as daily data
(aligned to a common calendar, forward-filled, no silent interpolation).

Note on yfinance limits (verified 2026-09-26): 1-minute bars are served
only for roughly the last 30 days, and at most ~8 days per request, so
:func:`fetch_intraday` walks the window in ``chunk_days`` slices.
"""
import time

import pandas as pd
import yfinance as yf

from .data import _extract_close, align_prices

#: yfinance lookback cap for 1-minute bars (verified 2026-09-26).
MAX_LOOKBACK_DAYS_1M = 30
#: yfinance per-request cap for 1-minute bars (verified 2026-09-26).
MAX_CHUNK_DAYS_1M = 8


def fetch_intraday(
    tickers,
    days: int = 30,
    interval: str = "1m",
    batch_size: int = 8,
    chunk_days: int = MAX_CHUNK_DAYS_1M,
    pause: float = 0.2,
    max_retries: int = 1,
) -> pd.DataFrame:
    """Download intraday bars for ``tickers`` over the last ``days`` days.

    The window is walked in ``chunk_days`` slices (yfinance caps 1-minute
    requests at ~8 days) and requests are batched (``batch_size`` tickers
    at a time, ``pause`` seconds between calls) to stay polite to the API.
    Empty responses are retried ``max_retries`` times. Returns an aligned
    close-price panel (see :func:`align_prices`).

    Only tickers present in *every* chunk are kept: a ticker missing from
    one chunk would leave leading gaps that :func:`align_prices` must drop,
    taking the whole panel down with it.

    Raises
    ------
    RuntimeError
        If every request comes back empty, or no ticker survived all chunks.
    """
    tickers = [str(t) for t in tickers]
    if interval == "1m":
        days = min(days, MAX_LOOKBACK_DAYS_1M)
        chunk_days = min(chunk_days, MAX_CHUNK_DAYS_1M)
    end = pd.Timestamp.now(tz="America/New_York").normalize() + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=days)
    chunk_panels = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + pd.Timedelta(days=chunk_days), end)
        s = chunk_start.strftime("%Y-%m-%d")
        e = chunk_end.strftime("%Y-%m-%d")
        batch_frames = []
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i : i + batch_size]
            raw = None
            for _ in range(max_retries + 1):
                raw = yf.download(
                    batch, start=s, end=e, interval=interval,
                    auto_adjust=True, progress=False,
                )
                if raw is not None and not raw.empty:
                    break
                time.sleep(pause * 2)
            if raw is None or raw.empty:
                continue
            batch_frames.append(_extract_close(raw, batch))
            time.sleep(pause)
        if batch_frames:
            # Batches within a chunk share the time index but cover
            # disjoint tickers -> join along columns.
            panel = pd.concat(batch_frames, axis=1)
            panel = panel.loc[:, ~panel.columns.duplicated()]
            # yfinance sometimes returns a ticker's column full of NaN
            # (failed/delisted symbol); drop those before the cross-chunk
            # intersection, otherwise their leading gaps would force
            # align_prices to discard every row.
            panel = panel.dropna(axis=1, how="all")
            if panel.shape[1]:
                chunk_panels.append(panel)
        chunk_start = chunk_end
    if not chunk_panels:
        raise RuntimeError(
            f"yfinance returned no intraday data for {tickers} "
            f"(last {days}d, interval={interval})."
        )
    # Keep the tickers present in every chunk; the chunks cover disjoint
    # time windows for those tickers -> stack along rows.
    common = [t for t in tickers if all(t in p.columns for p in chunk_panels)]
    if not common:
        raise RuntimeError(
            "No ticker returned data for every intraday chunk; "
            f"{len(tickers)} requested."
        )
    combined = pd.concat([p[common] for p in chunk_panels], axis=0)
    combined = combined[~combined.index.duplicated(keep="first")]
    return align_prices(combined)


def count_observations(prices: pd.DataFrame) -> int:
    """Total number of (bar, ticker) observations in an aligned panel."""
    return int(prices.count().sum())


def screen_universe(
    prices: pd.DataFrame,
    corr_threshold: float = 0.7,
    top_n: int = 25,
) -> pd.DataFrame:
    """Fast correlation pre-screen on (intraday) returns.

    Computes the full correlation matrix on bar-to-bar returns and returns
    candidate pairs with correlation >= ``corr_threshold``, sorted by
    correlation descending and capped at ``top_n`` rows. This is the cheap
    first pass; the expensive cointegration tests run only on these
    candidates downstream.
    """
    rets = prices.pct_change().dropna(how="all")
    corr = rets.corr()
    tickers = [str(c) for c in prices.columns]
    rows = []
    for i, a in enumerate(tickers):
        for b in tickers[i + 1 :]:
            c = float(corr.loc[a, b])
            if c >= corr_threshold:
                rows.append({"asset_a": a, "asset_b": b, "correlation": c})
    out = pd.DataFrame(rows, columns=["asset_a", "asset_b", "correlation"])
    if out.empty:
        return out
    return (
        out.sort_values("correlation", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
