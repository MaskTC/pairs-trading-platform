"""Signal-quality analysis: how much does screening cut false signals?

A "naive" screen trades every pair whose correlation clears the threshold.
The "filtered" screen additionally requires the Engle-Granger cointegration
test (p < 0.05) and a finite positive half-life of mean reversion.

Entry signals fired on pairs that fail the cointegration filter are counted
as false signals -- the strategy would have traded a relationship with no
statistical mean reversion. The suppression rate is the fraction of naive
signals the filter removes, and the losing-trade reduction is the matching
fraction for round-trip trades that lost money (net of costs).
"""
import pandas as pd

from .backtest import run_backtest
from .pairs import find_pairs
from .signals import compute_spread, compute_zscore, generate_signals


def _screen_stats(
    screen: pd.DataFrame,
    prices: pd.DataFrame,
    window: int,
    entry_z: float,
    exit_z: float,
    cost_model,
    initial_capital: float,
) -> dict:
    """Total entry signals and losing round trips for every screened pair."""
    total_signals = 0
    losing_trades = 0
    total_trades = 0
    for _, row in screen.iterrows():
        a = pd.Series(prices[row["asset_a"]], dtype=float)
        b = pd.Series(prices[row["asset_b"]], dtype=float)
        beta = float(row["beta"])
        z = compute_zscore(compute_spread(a, b, beta), window=int(window))
        sig = generate_signals(z, entry_z=float(entry_z), exit_z=float(exit_z))
        total_signals += int(sig["entry_long"].sum() + sig["entry_short"].sum())
        res = run_backtest(
            a,
            b,
            beta,
            sig["target"],
            cost_model=cost_model,
            initial_capital=float(initial_capital),
        )
        trades = res["trades"]
        total_trades += len(trades)
        if len(trades):
            losing_trades += int((trades["net_pnl"] < 0).sum())
    return {
        "n_pairs": int(len(screen)),
        "total_signals": total_signals,
        "total_trades": total_trades,
        "losing_trades": losing_trades,
    }


def screen_comparison(
    prices: pd.DataFrame,
    corr_threshold: float = 0.7,
    window: int = 30,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    cost_model=None,
    initial_capital: float = 10_000.0,
) -> dict:
    """Compare the naive (correlation-only) screen against the full filter.

    Both screens are evaluated with identical signal parameters and costs,
    so any difference is attributable to the cointegration / half-life
    filter alone.

    Returns a dict with per-screen stats plus ``signal_suppression``
    (fraction of naive entry signals removed by the filter) and
    ``losing_trade_reduction`` (fraction of naive losing round trips
    removed). Both are 0.0 when the naive screen fires nothing.
    """
    prices = pd.DataFrame(prices)
    found = find_pairs(prices, corr_threshold=corr_threshold)
    naive = found[found["correlation"] >= corr_threshold].reset_index(drop=True)
    filtered = found[found["selected"]].reset_index(drop=True)

    kwargs = dict(
        prices=prices,
        window=window,
        entry_z=entry_z,
        exit_z=exit_z,
        cost_model=cost_model,
        initial_capital=initial_capital,
    )
    naive_stats = _screen_stats(naive, **kwargs)
    filtered_stats = _screen_stats(filtered, **kwargs)

    n_sig = naive_stats["total_signals"]
    n_lose = naive_stats["losing_trades"]
    suppression = 1.0 - filtered_stats["total_signals"] / n_sig if n_sig else 0.0
    loser_reduction = (
        1.0 - filtered_stats["losing_trades"] / n_lose if n_lose else 0.0
    )
    return {
        "naive": naive_stats,
        "filtered": filtered_stats,
        "signal_suppression": float(suppression),
        "losing_trade_reduction": float(loser_reduction),
    }
