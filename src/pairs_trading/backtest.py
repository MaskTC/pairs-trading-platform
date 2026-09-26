"""Event-driven pairs backtester (bar-by-bar, no look-ahead bias).

Accounting model
----------------
One "spread contract" = long 1 share of A and short ``beta`` shares of B.
With ``S_t = A_t - beta * B_t`` the spread value:

* equity at bar ``t`` = ``cash + contracts * S_t``,
* opening a long contract at the close: ``cash -= S_t``
  (buy A for ``A_t``, receive ``beta * B_t`` from shorting B),
* closing: ``cash += contracts * S_t``,
* so a trade itself never changes equity except through costs.

A ``target`` signal decided at the close of bar ``t`` changes the position
at that close; the new position therefore earns the return from bar ``t``
to bar ``t + 1`` -- signals never profit from the bar they are computed on.

Position sizing: on entry, contracts are sized so gross notional
``|contracts| * (A_t + beta * B_t)`` equals ``equity * max_gross_leverage``
(1x by default). Costs are charged once per position change: entries and
exits each cost once; a flip (long -> short) costs twice (close + open).
Any position still open at the final bar is force-closed at that close.
"""
import numpy as np
import pandas as pd

from .costs import TransactionCostModel

_TRADE_COLUMNS = [
    "entry_time",
    "exit_time",
    "side",
    "contracts",
    "entry_spread",
    "exit_spread",
    "gross_pnl",
    "costs",
    "net_pnl",
]


def run_backtest(
    price_a,
    price_b,
    beta: float,
    target,
    cost_model: TransactionCostModel | None = None,
    initial_capital: float = 10_000.0,
    max_gross_leverage: float = 1.0,
) -> dict:
    """Run the bar-by-bar backtest.

    Returns a dict with ``equity`` (pd.Series), ``positions`` (signed
    contracts held during each bar), ``trades`` (one row per completed
    round trip), ``daily_returns``, ``n_trades`` and ``final_equity``.
    """
    if cost_model is None:
        cost_model = TransactionCostModel()

    a = pd.Series(price_a, dtype=float)
    b = pd.Series(price_b, dtype=float)
    if not a.index.equals(b.index):
        raise ValueError("price_a and price_b must share the same index")
    idx = a.index
    A = a.to_numpy()
    B = b.to_numpy()
    S = A - float(beta) * B
    n = len(A)

    # NaN target -> hold the previous target (never trade on missing data).
    tgt = (
        pd.Series(target, index=idx).ffill().fillna(0).astype(int).to_numpy()
    )

    cash = float(initial_capital)
    contracts = 0.0
    side = 0
    open_trade = None

    equity = np.zeros(n)
    positions = np.zeros(n)
    trades: list[dict] = []

    def _record_close(t, exit_contracts, entry, c_close):
        trades.append(
            {
                "entry_time": entry["entry_time"],
                "exit_time": idx[t],
                "side": entry["side"],
                "contracts": abs(entry["contracts"]),
                "entry_spread": entry["entry_spread"],
                "exit_spread": float(S[t]),
                "gross_pnl": float(
                    exit_contracts * (S[t] - entry["entry_spread"])
                ),
                "costs": float(entry["costs"] + c_close),
                "net_pnl": float(
                    exit_contracts * (S[t] - entry["entry_spread"])
                    - (entry["costs"] + c_close)
                ),
            }
        )

    for t in range(n):
        # Mark to market with the position held since the previous close.
        equity[t] = cash + contracts * S[t]
        positions[t] = contracts

        desired = int(tgt[t])
        if desired != side:
            if contracts != 0.0:
                gross_close = abs(contracts) * (A[t] + beta * B[t])
                c_close = cost_model.cost(gross_close)
                cash += contracts * S[t]  # settle the share positions
                cash -= c_close
                _record_close(t, contracts, open_trade, c_close)
                open_trade = None
                contracts = 0.0
            if desired != 0:
                per_contract_gross = A[t] + beta * B[t]
                if per_contract_gross <= 0:  # pragma: no cover - defensive
                    raise ValueError("Non-positive gross notional per contract")
                new_contracts = (
                    desired * (cash * max_gross_leverage) / per_contract_gross
                )
                gross_open = abs(new_contracts) * per_contract_gross
                c_open = cost_model.cost(gross_open)
                cash -= new_contracts * S[t]
                cash -= c_open
                contracts = new_contracts
                open_trade = {
                    "entry_time": idx[t],
                    "side": desired,
                    "contracts": new_contracts,
                    "entry_spread": float(S[t]),
                    "costs": float(c_open),
                }
            side = desired

    # Force-close any open position at the final close.
    if contracts != 0.0:
        t = n - 1
        gross_close = abs(contracts) * (A[t] + beta * B[t])
        c_close = cost_model.cost(gross_close)
        cash += contracts * S[t]
        cash -= c_close
        _record_close(t, contracts, open_trade, c_close)
        contracts = 0.0
        equity[t] = cash

    equity_s = pd.Series(equity, index=idx, name="equity")
    positions_s = pd.Series(positions, index=idx, name="contracts")
    trades_df = pd.DataFrame(trades, columns=_TRADE_COLUMNS)
    daily_returns = equity_s.pct_change().fillna(0.0)
    return {
        "equity": equity_s,
        "positions": positions_s,
        "trades": trades_df,
        "daily_returns": daily_returns,
        "n_trades": len(trades_df),
        "final_equity": float(equity_s.iloc[-1]),
    }
