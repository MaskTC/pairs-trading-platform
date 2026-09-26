"""Pair identification: correlation, hedge ratio, cointegration, half-life.

A pair is only tradeable when it passes *all* filters: high correlation
**and** a significant Engle-Granger cointegration test **and** a positive,
finite half-life of mean reversion. Correlation alone is not enough --
two independent random walks can look correlated over short samples.
"""
import itertools

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

from .signals import compute_spread


def hedge_ratio(y: pd.Series, x: pd.Series):
    """OLS hedge ratio (beta) from ``y = alpha + beta * x``.

    Returns
    -------
    (beta, alpha) : tuple[float, float]
    """
    df = pd.DataFrame({"y": y, "x": x}).dropna()
    if df["x"].std() == 0:
        raise ValueError("hedge_ratio is undefined: x has zero variance")
    X = sm.add_constant(df["x"].to_numpy())
    res = sm.OLS(df["y"].to_numpy(), X).fit()
    return float(res.params[1]), float(res.params[0])


def engle_granger(y: pd.Series, x: pd.Series) -> dict:
    """Engle-Granger two-step cointegration test.

    Returns a dict with the test statistic, p-value and critical values.
    A p-value below 0.05 rejects the null of *no* cointegration.
    """
    df = pd.DataFrame({"y": y, "x": x}).dropna()
    t_stat, p_value, crit_values = coint(df["y"], df["x"])
    # statsmodels returns critical values as an array for 1%, 5%, 10%.
    crit = {"1%": float(crit_values[0]), "5%": float(crit_values[1]),
            "10%": float(crit_values[2])}
    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "crit_values": crit,
    }


def half_life(spread: pd.Series) -> float:
    """Half-life of mean reversion of a spread series.

    Regress ``Δspread`` on lagged ``spread``: ``Δs_t = λ s_{t-1} + ε``.
    Half-life is ``-ln(2) / λ``. Returns ``inf`` when ``λ >= 0``
    (no mean reversion).
    """
    s = pd.Series(spread).dropna()
    df = pd.DataFrame({"lag": s.shift(1), "delta": s.diff()}).dropna()
    X = sm.add_constant(df["lag"].to_numpy())
    res = sm.OLS(df["delta"].to_numpy(), X).fit()
    lam = float(res.params[1])
    if lam >= 0:
        return np.inf
    return float(-np.log(2.0) / lam)


def find_pairs(
    prices: pd.DataFrame,
    corr_threshold: float = 0.7,
    p_threshold: float = 0.05,
    max_half_life: float = 252.0,
) -> pd.DataFrame:
    """Screen every pair of columns in ``prices``.

    A pair is ``selected`` only if correlation >= ``corr_threshold``,
    the Engle-Granger p-value < ``p_threshold`` and
    ``0 < half_life < max_half_life``.
    """
    tickers = [str(c) for c in prices.columns]
    corr = prices.corr()
    rows = []
    for a, b in itertools.combinations(tickers, 2):
        correlation = float(corr.loc[a, b])
        beta, alpha = hedge_ratio(prices[a], prices[b])
        eg = engle_granger(prices[a], prices[b])
        hl = half_life(compute_spread(prices[a], prices[b], beta))
        selected = bool(
            correlation >= corr_threshold
            and eg["p_value"] < p_threshold
            and 0.0 < hl < max_half_life
        )
        rows.append(
            {
                "asset_a": a,
                "asset_b": b,
                "correlation": correlation,
                "beta": beta,
                "alpha": alpha,
                "coint_t_stat": eg["t_stat"],
                "coint_p_value": eg["p_value"],
                "half_life": hl,
                "selected": selected,
            }
        )
    return pd.DataFrame(rows)
