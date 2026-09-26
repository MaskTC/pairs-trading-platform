"""Custom dataset support: bring your own price data.

Three layouts are accepted; all are normalized to the same aligned
close-price panel the rest of the platform uses:

* **long CSV** -- columns ``[datetime, symbol, close]``
  (one row per bar per ticker; extra columns are ignored)
* **wide CSV** -- columns ``[datetime, AAA, BBB, ...]``
  (one column per ticker)
* **directory** -- one CSV per ticker, each with ``[datetime, close]``;
  the file stem (``AAPL.csv`` -> ``AAPL``) becomes the ticker.

Timestamps are parsed with :func:`pandas.to_datetime` (a ``time_format``
may be passed through for speed) and normalized to tz-naive values so
mixed-tz files cannot silently misalign. Every loader runs
:func:`validate_panel` and then the standard alignment
(forward-fill, drop leading gaps).

:func:`generate_synthetic_minutes` builds a seeded, clearly-labeled
*synthetic* 1-minute panel -- useful for exercising the high-frequency
path (and the 500k+ observation scale) with no network access. Synthetic
data must never be presented as real market data.
"""
import os

import numpy as np
import pandas as pd

from .data import align_prices

DATETIME_COL = "datetime"
TICKER_COL = "symbol"
PRICE_COL = "close"


def _parse_datetimes(values, time_format=None) -> pd.DatetimeIndex:
    """Parse to tz-naive datetimes, raising a clear error on failure."""
    try:
        parsed = pd.to_datetime(values, format=time_format, utc=True)
    except Exception as exc:
        raise ValueError(
            f"Could not parse datetime values (first few: "
            f"{list(values[:3])!r}): {exc}"
        ) from exc
    if parsed.isna().any():
        bad = values[parsed.isna()][:3]
        raise ValueError(
            f"Unparseable datetime values (first few: {list(bad)!r}). "
            f"Pass time_format= explicitly if the layout is unusual."
        )
    return pd.DatetimeIndex(parsed.dt.tz_localize(None))


def validate_panel(prices: pd.DataFrame) -> pd.DataFrame:
    """Check a price panel and return it with a clean, sorted DatetimeIndex.

    Raises
    ------
    ValueError
        If the index is not datetime-like, contains duplicates, any column
        is non-numeric, or the frame is empty.
    """
    prices = pd.DataFrame(prices)
    if prices.empty:
        raise ValueError("Price panel is empty.")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError(
            f"Index must be a DatetimeIndex, got {type(prices.index).__name__}. "
            f"Parse your timestamp column first (see datasets._parse_datetimes)."
        )
    if prices.index.has_duplicates:
        dupes = prices.index[prices.index.duplicated()][:3]
        raise ValueError(
            f"Duplicate timestamps in index (e.g. {list(dupes)!r}). "
            f"Aggregate or de-duplicate before loading."
        )
    non_numeric = [
        str(c) for c in prices.columns
        if not pd.api.types.is_numeric_dtype(prices[c])
    ]
    if non_numeric:
        raise ValueError(
            f"Non-numeric price columns: {non_numeric}. "
            f"Point price_col= at the right column."
        )
    prices = prices.copy()
    prices.columns = [str(c) for c in prices.columns]
    return prices.sort_index()


def read_long_csv(
    path,
    datetime_col: str = DATETIME_COL,
    ticker_col: str = TICKER_COL,
    price_col: str = PRICE_COL,
    time_format=None,
) -> pd.DataFrame:
    """Load a long-format CSV (``datetime, symbol, close``) into a panel."""
    df = pd.read_csv(path)
    missing = {datetime_col, ticker_col, price_col} - set(df.columns)
    if missing:
        raise ValueError(
            f"{path}: missing required columns {sorted(missing)} "
            f"(found {list(df.columns)})."
        )
    df = df[[datetime_col, ticker_col, price_col]].dropna()
    if df.empty:
        raise ValueError(f"{path}: no usable rows after dropping NaNs.")
    if df.duplicated(subset=[datetime_col, ticker_col]).any():
        raise ValueError(
            f"{path}: duplicate (datetime, symbol) rows found; "
            f"aggregate to one price per bar first."
        )
    df[datetime_col] = _parse_datetimes(df[datetime_col], time_format)
    panel = df.pivot_table(
        index=datetime_col, columns=ticker_col, values=price_col, aggfunc="last"
    )
    panel.columns = [str(c) for c in panel.columns]
    return align_prices(validate_panel(panel))


def read_wide_csv(path, datetime_col: str = DATETIME_COL, time_format=None) -> pd.DataFrame:
    """Load a wide-format CSV (``datetime, AAA, BBB, ...``) into a panel."""
    df = pd.read_csv(path)
    if datetime_col not in df.columns:
        raise ValueError(
            f"{path}: missing datetime column {datetime_col!r} "
            f"(found {list(df.columns)})."
        )
    tickers = [c for c in df.columns if c != datetime_col]
    if not tickers:
        raise ValueError(f"{path}: no ticker columns found.")
    df[datetime_col] = _parse_datetimes(df[datetime_col], time_format)
    panel = df.set_index(datetime_col)[tickers]
    panel.columns = [str(c) for c in panel.columns]
    return align_prices(validate_panel(panel))


def read_ticker_dir(
    directory,
    datetime_col: str = DATETIME_COL,
    price_col: str = PRICE_COL,
    pattern: str = "*.csv",
    time_format=None,
) -> pd.DataFrame:
    """Load one CSV per ticker from ``directory`` (file stem = ticker)."""
    import glob

    files = sorted(glob.glob(os.path.join(str(directory), pattern)))
    if not files:
        raise ValueError(f"No files matching {pattern!r} in {directory}.")
    series = {}
    for path in files:
        ticker = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_csv(path)
        missing = {datetime_col, price_col} - set(df.columns)
        if missing:
            raise ValueError(
                f"{path}: missing required columns {sorted(missing)} "
                f"(found {list(df.columns)})."
            )
        df = df[[datetime_col, price_col]].dropna()
        df[datetime_col] = _parse_datetimes(df[datetime_col], time_format)
        series[ticker] = df.set_index(datetime_col)[price_col]
    panel = pd.DataFrame(series)
    return align_prices(validate_panel(panel))


def write_example_long_csv(path, seed: int = 0) -> str:
    """Write a small example long-format CSV demonstrating the layout."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-02 09:30", periods=390 * 3, freq="min")
    rows = []
    for ticker, base in (("AAA", 100.0), ("BBB", 50.0)):
        walk = base + np.cumsum(rng.normal(0, 0.05, len(idx)))
        for ts, px in zip(idx, walk):
            rows.append({"datetime": ts, "symbol": ticker, "close": round(float(px), 4)})
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def generate_synthetic_minutes(
    tickers,
    trading_days: int = 22,
    seed: int = 0,
    start: str = "2026-08-01",
) -> pd.DataFrame:
    """Generate a seeded *synthetic* 1-minute panel (for offline testing).

    Each ticker follows a geometric-style random walk sharing a common
    market factor (so the universe has realistic cross-correlation).
    Bars run 09:30-16:00 Eastern. The output is synthetic -- it must
    never be presented as real market data.
    """
    tickers = [str(t) for t in tickers]
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start=start, periods=trading_days)
    bars_per_day = 390
    n = trading_days * bars_per_day
    market = rng.normal(0, 0.0008, n)
    data = {}
    for i, ticker in enumerate(tickers):
        beta_mkt = 0.6 + 0.8 * rng.random()
        idio = rng.normal(0, 0.0012, n)
        rets = beta_mkt * market + idio
        base = 20.0 + 180.0 * rng.random()
        data[ticker] = base * np.exp(np.cumsum(rets))
    idx = pd.DatetimeIndex(
        [d + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(minutes=m)
         for d in days for m in range(bars_per_day)]
    )
    panel = pd.DataFrame(data, index=idx)
    return align_prices(validate_panel(panel))
