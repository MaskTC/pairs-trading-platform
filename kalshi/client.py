"""Read-only client for the Kalshi Predictions REST API (trade-api v2).

Market-data endpoints are public -- no authentication is needed for reads.
Base URL verified live against https://api.elections.kalshi.com/trade-api/v2
(see README for the verification date). This client deliberately exposes
no order-placement or account endpoints: analysis only, no real-money
trading.
"""
import requests


class KalshiClient:
    """Thin wrapper around the public Kalshi market-data endpoints."""

    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

    def __init__(self, session=None, timeout: int = 15):
        # ``session`` is injectable so tests can run without the network.
        self.session = session or requests.Session()
        self.timeout = timeout

    def _get(self, path: str, params: dict | None = None) -> dict:
        resp = self.session.get(
            self.BASE_URL + path, params=params or {}, timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json()

    def list_markets(self, limit: int = 100, cursor: str | None = None, **filters) -> list:
        """GET /markets -- paginated list of market snapshots.

        Extra keyword arguments are passed as query params
        (e.g. ``event_ticker=...``, ``status="active"``).
        """
        params = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        params.update(filters)
        return self._get("/markets", params).get("markets", [])

    def get_market(self, ticker: str) -> dict:
        """GET /markets/{ticker} -- single market snapshot."""
        return self._get(f"/markets/{ticker}").get("market", {})

    def get_orderbook(self, ticker: str, depth: int | None = None) -> dict:
        """GET /markets/{ticker}/orderbook -- resting yes/no bids and asks.

        Live response shape (verified 2026-09-26)::

            {"orderbook_fp": {"yes_dollars": [[price, size], ...],
                              "no_dollars": [[price, size], ...]}}

        (older docs show an ``orderbook`` key with ``yes``/``no`` lists;
        both are accepted).
        """
        params = {"depth": depth} if depth else {}
        data = self._get(f"/markets/{ticker}/orderbook", params)
        return data.get("orderbook_fp", data.get("orderbook", {}))

    def list_events(self, limit: int = 100, cursor: str | None = None) -> list:
        """GET /events -- paginated list of events."""
        params = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        return self._get("/events", params).get("events", [])

    def get_event_markets(self, event_ticker: str, limit: int = 100) -> list:
        """All markets belonging to one event (for cross-market checks)."""
        return self.list_markets(limit=limit, event_ticker=event_ticker)
