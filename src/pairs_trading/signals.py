"""Spread construction, z-scores and entry/exit signal generation.

Convention: ``target = +1`` means long the spread (long asset A, short
``beta`` units of asset B); ``target = -1`` means short the spread;
``target = 0`` is flat. Signals at bar ``t`` are computed from data
available at the close of bar ``t`` and are acted on at that close, so the
position earns returns starting from bar ``t + 1`` -- no look-ahead bias.
"""
import numpy as np
import pandas as pd


def compute_spread(price_a: pd.Series, price_b: pd.Series, beta: float) -> pd.Series:
    """Spread ``S = A - beta * B`` (index-aligned by pandas)."""
    return pd.Series(price_a) - beta * pd.Series(price_b)


def compute_zscore(
    spread: pd.Series, window: int, min_periods: int | None = None
) -> pd.Series:
    """Rolling z-score ``(spread - rolling_mean) / rolling_std``.

    ``min_periods`` defaults to ``window`` so the warm-up period is
    explicitly NaN. A zero rolling standard deviation yields NaN
    (never ``inf``), and the output keeps the input's length and index --
    NaNs are never silently dropped mid-series.
    """
    if min_periods is None:
        min_periods = window
    s = pd.Series(spread)
    mean = s.rolling(window, min_periods=min_periods).mean()
    std = s.rolling(window, min_periods=min_periods).std(ddof=1)
    std = std.mask(std == 0)  # constant window -> undefined z, not inf
    return (s - mean) / std


def generate_signals(
    z: pd.Series, entry_z: float, exit_z: float
) -> pd.DataFrame:
    """State-machine signal generator.

    * Enter long (+1) when flat and ``z < -entry_z``;
      enter short (-1) when flat and ``z > entry_z``.
    * Exit to flat when ``|z| < exit_z`` **or** ``z`` crosses 0
      (the cross rule is what exits trades when ``exit_z == 0``).
    * An exit and an opposite entry may occur on the same bar
      (a flip); an entry bar never immediately exits.
    * NaN z-scores hold the current position.
    """
    z = pd.Series(z)
    n = len(z)
    entry_long = np.zeros(n, dtype=bool)
    entry_short = np.zeros(n, dtype=bool)
    exit_signal = np.zeros(n, dtype=bool)
    target = np.zeros(n, dtype=int)

    pos = 0
    prev_z = np.nan
    for t in range(n):
        zt = z.iloc[t]
        if np.isnan(zt):
            target[t] = pos
            prev_z = np.nan
            continue
        if pos == 1:
            crossed_up = (not np.isnan(prev_z)) and (prev_z < 0 <= zt)
            if abs(zt) < exit_z or crossed_up:
                pos = 0
                exit_signal[t] = True
        elif pos == -1:
            crossed_down = (not np.isnan(prev_z)) and (prev_z > 0 >= zt)
            if abs(zt) < exit_z or crossed_down:
                pos = 0
                exit_signal[t] = True
        if pos == 0:
            if zt < -entry_z:
                pos = 1
                entry_long[t] = True
            elif zt > entry_z:
                pos = -1
                entry_short[t] = True
        target[t] = pos
        prev_z = zt

    return pd.DataFrame(
        {
            "entry_long": entry_long,
            "entry_short": entry_short,
            "exit_signal": exit_signal,
            "target": target,
        },
        index=z.index,
    )
