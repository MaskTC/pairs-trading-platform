"""Performance and risk metrics for backtested equity curves."""
import numpy as np
import pandas as pd


def total_return(equity: pd.Series) -> float:
    """(final / initial) - 1."""
    eq = pd.Series(equity).dropna()
    return float(eq.iloc[-1] / eq.iloc[0] - 1.0)


def annualized_return(equity: pd.Series, periods_per_year: int = 252) -> float:
    """CAGR over the equity curve."""
    eq = pd.Series(equity).dropna()
    n = len(eq) - 1
    if n <= 0:
        return 0.0
    return float((eq.iloc[-1] / eq.iloc[0]) ** (periods_per_year / n) - 1.0)


def annualized_volatility(
    daily_returns: pd.Series, periods_per_year: int = 252
) -> float:
    """Annualized standard deviation of daily returns (ddof=1)."""
    r = pd.Series(daily_returns).dropna()
    if len(r) < 2:
        return 0.0
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    daily_returns: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sharpe ratio.

    Returns 0.0 when volatility is zero (e.g. a flat equity curve from
    no trades) instead of NaN/inf, so optimizers can rank it.
    """
    r = pd.Series(daily_returns).dropna()
    if len(r) < 2:
        return 0.0
    vol = r.std(ddof=1)
    if vol < 1e-12:
        # Zero (up to floating-point noise) volatility: a flat equity curve
        # has no meaningful Sharpe; return 0.0 instead of NaN/inf.
        return 0.0
    return float((r.mean() - risk_free) / vol * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> dict:
    """Largest peak-to-trough decline.

    Returns ``{"max_drawdown": negative fraction, "peak_index": ...,
    "trough_index": ...}``. A monotonically rising curve has 0.0 drawdown.
    """
    eq = pd.Series(equity).dropna().to_numpy(dtype=float)
    running_peak = np.maximum.accumulate(eq)
    drawdown = (eq - running_peak) / running_peak
    trough = int(np.argmin(drawdown))
    peak = int(np.argmax(eq[: trough + 1]))
    return {
        "max_drawdown": float(drawdown[trough]),
        "peak_index": peak,
        "trough_index": trough,
    }


def win_rate(trades: pd.DataFrame) -> float:
    """Fraction of completed round-trip trades with positive net P&L.

    Returns NaN when there are no completed trades.
    """
    pnl = pd.Series(trades["net_pnl"]) if len(trades) else pd.Series([], dtype=float)
    if len(pnl) == 0:
        return float("nan")
    return float((pnl > 0).mean())


def cumulative_returns(daily_returns: pd.Series) -> pd.Series:
    """Cumulative return series: ``(1 + r).cumprod() - 1``."""
    r = pd.Series(daily_returns)
    return (1.0 + r).cumprod() - 1.0
