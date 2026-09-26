"""Market data acquisition and cleaning.

All price handling lives here so that every downstream module (pairs,
signals, backtest) works on an identical, fully-aligned panel.
"""
import pandas as pd
import yfinance as yf


def _extract_close(raw: pd.DataFrame, tickers) -> pd.DataFrame:
    """Pull the close-price panel out of a yfinance download.

    Handles both the multi-ticker MultiIndex layout and the single-ticker
    flat layout, tolerating either (field, ticker) or (ticker, field)
    level ordering.
    """
    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = raw.columns.get_level_values(0)
        lvl1 = raw.columns.get_level_values(1)
        if "Adj Close" in lvl0 or "Close" in lvl0:
            field = "Adj Close" if "Adj Close" in lvl0 else "Close"
            close = raw.xs(field, level=0, axis=1)
        elif "Adj Close" in lvl1 or "Close" in lvl1:
            field = "Adj Close" if "Adj Close" in lvl1 else "Close"
            close = raw.xs(field, level=1, axis=1)
        else:  # pragma: no cover - defensive
            raise ValueError(f"Could not find a Close column in {raw.columns!r}")
    else:
        field = "Adj Close" if "Adj Close" in raw.columns else "Close"
        name = tickers[0] if isinstance(tickers, (list, tuple)) else tickers
        close = raw[[field]].rename(columns={field: name})
    close = close.copy()
    close.columns = [str(c) for c in close.columns]
    return close


def fetch_prices(tickers, start, end) -> pd.DataFrame:
    """Download adjusted close prices and return an aligned panel.

    Parameters
    ----------
    tickers : list[str]
    start, end : str
        Date bounds understood by yfinance (``end`` is exclusive).

    Returns
    -------
    pd.DataFrame
        Columns are tickers, index is trading dates, no missing values
        (see :func:`align_prices`).
    """
    raw = yf.download(
        tickers, start=start, end=end, auto_adjust=True, progress=False
    )
    if raw.empty:
        raise RuntimeError(
            f"yfinance returned no data for {tickers} in [{start}, {end})."
        )
    return align_prices(_extract_close(raw, tickers))


def align_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Align assets to a common calendar.

    Reindexes to the union of trading dates, forward-fills gaps where one
    asset traded and another did not, and drops leading rows that cannot be
    filled (e.g. an asset that lists after the sample starts). Interior
    missing values are therefore always filled; leading ones are removed,
    never silently interpolated.
    """
    aligned = prices.sort_index().ffill().dropna()
    aligned.columns = [str(c) for c in aligned.columns]
    return aligned


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily percentage returns (first row is NaN)."""
    return prices.pct_change()
