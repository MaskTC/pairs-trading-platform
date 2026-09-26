"""Mispricing detection on Kalshi market snapshots (read-only analysis).

Two checks, both in "does a locked-in edge exist" form:

1. **Complementary yes/no** -- within one market, buying the yes *and* the
   no (or selling both) pays exactly 100c at settlement no matter what
   happens. If ``yes_ask + no_ask < 100c`` you can buy both for a locked-in
   profit; if ``yes_bid + no_bid > 100c`` you can sell both.
2. **Event arbitrage** -- for an event whose markets are mutually exclusive
   and exhaustive outcomes (e.g. "which team wins"), buying every yes at a
   combined ask below 100c (or selling every yes at a combined bid above
   100c) locks in a profit.

Prices on the wire are dollar-denominated decimal strings
(``yes_bid_dollars`` etc.); everything here is normalized to cents.
"""
import pandas as pd

_CONTRACT_CENTS = 100.0


def _price_cents(market: dict, field: str) -> float:
    """Extract a yes/no bid/ask price in cents.

    Prefers the current ``<field>_dollars`` string shape; falls back to a
    legacy integer-cents ``<field>`` shape. Returns NaN when absent.
    """
    dollar_key = f"{field}_dollars"
    if dollar_key in market and market[dollar_key] not in (None, ""):
        return float(market[dollar_key]) * 100.0
    if field in market and market[field] not in (None, ""):
        return float(market[field])
    return float("nan")


def _valid(*prices) -> bool:
    import math

    return all(p is not None and not math.isnan(p) for p in prices)


def complementary_arb(market: dict, threshold_cents: float = 0.0) -> dict | None:
    """Check one market for a yes+no / no+yes locked-in edge.

    Returns a dict with ``direction`` (``"buy_both"`` or ``"sell_both"``),
    ``edge_cents`` and the trade's cost/proceeds, or ``None``.
    """
    yes_bid = _price_cents(market, "yes_bid")
    yes_ask = _price_cents(market, "yes_ask")
    no_bid = _price_cents(market, "no_bid")
    no_ask = _price_cents(market, "no_ask")
    if not _valid(yes_bid, yes_ask, no_bid, no_ask):
        return None

    buy_cost = yes_ask + no_ask
    if buy_cost < _CONTRACT_CENTS - threshold_cents:
        return {
            "ticker": market.get("ticker"),
            "event_ticker": market.get("event_ticker"),
            "direction": "buy_both",
            "cost_cents": buy_cost,
            "edge_cents": _CONTRACT_CENTS - buy_cost,
        }
    sell_proceeds = yes_bid + no_bid
    if sell_proceeds > _CONTRACT_CENTS + threshold_cents:
        return {
            "ticker": market.get("ticker"),
            "event_ticker": market.get("event_ticker"),
            "direction": "sell_both",
            "proceeds_cents": sell_proceeds,
            "edge_cents": sell_proceeds - _CONTRACT_CENTS,
        }
    return None


def event_arbitrage(markets: list, threshold_cents: float = 0.0) -> list:
    """Check each event's markets for a buy-all / sell-all yes edge.

    Assumes the event's markets are mutually exclusive and exhaustive --
    exactly one of them settles yes. Events with fewer than two priced
    markets are skipped.
    """
    by_event: dict[str, list] = {}
    for m in markets:
        by_event.setdefault(m.get("event_ticker"), []).append(m)

    arbs = []
    for event_ticker, group in by_event.items():
        yes_asks = [_price_cents(m, "yes_ask") for m in group]
        yes_bids = [_price_cents(m, "yes_bid") for m in group]
        if len(group) < 2 or not _valid(*yes_asks, *yes_bids):
            continue
        total_ask = sum(yes_asks)
        total_bid = sum(yes_bids)
        if total_ask < _CONTRACT_CENTS - threshold_cents:
            arbs.append(
                {
                    "event_ticker": event_ticker,
                    "direction": "buy_all_yes",
                    "tickers": [m.get("ticker") for m in group],
                    "cost_cents": total_ask,
                    "edge_cents": _CONTRACT_CENTS - total_ask,
                }
            )
        elif total_bid > _CONTRACT_CENTS + threshold_cents:
            arbs.append(
                {
                    "event_ticker": event_ticker,
                    "direction": "sell_all_yes",
                    "tickers": [m.get("ticker") for m in group],
                    "proceeds_cents": total_bid,
                    "edge_cents": total_bid - _CONTRACT_CENTS,
                }
            )
    return arbs


def find_mispricings(markets: list, threshold_cents: float = 0.0) -> pd.DataFrame:
    """Run the complementary yes/no check over many markets.

    Returns a DataFrame with one row per mispriced market.
    """
    rows = []
    for m in markets:
        arb = complementary_arb(m, threshold_cents=threshold_cents)
        if arb is not None:
            rows.append(arb)
    return pd.DataFrame(rows)
