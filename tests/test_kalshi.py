"""Tests for kalshi.client and kalshi.mispricing — all mocked, no network."""
import pytest

from kalshi import client as kalshi_client
from kalshi import mispricing


# ---------- fixtures in the documented API shape (dollar-denominated strings) ----------

def make_market(ticker, event, yes_bid, yes_ask, no_bid, no_ask):
    return {
        "ticker": ticker,
        "event_ticker": event,
        "market_type": "binary",
        "status": "active",
        "yes_bid_dollars": f"{yes_bid:.2f}",
        "yes_ask_dollars": f"{yes_ask:.2f}",
        "no_bid_dollars": f"{no_bid:.2f}",
        "no_ask_dollars": f"{no_ask:.2f}",
        "last_price_dollars": f"{(yes_bid + yes_ask) / 2:.2f}",
    }


# ---------- mispricing: complementary yes/no ----------


def test_buy_both_arb_when_asks_sum_below_dollar():
    m = make_market("X-1", "E-1", 0.38, 0.40, 0.48, 0.50)
    arb = mispricing.complementary_arb(m)
    assert arb is not None
    assert arb["direction"] == "buy_both"
    # Pay 40c + 50c = 90c; exactly one side settles at 100c -> 10c edge.
    assert arb["edge_cents"] == pytest.approx(10.0)
    assert arb["cost_cents"] == pytest.approx(90.0)


def test_sell_both_arb_when_bids_sum_above_dollar():
    m = make_market("X-2", "E-1", 0.60, 0.62, 0.55, 0.57)
    arb = mispricing.complementary_arb(m)
    assert arb is not None
    assert arb["direction"] == "sell_both"
    # Collect 60c + 55c = 115c; exactly one side settles at 100c -> 15c edge.
    assert arb["edge_cents"] == pytest.approx(15.0)


def test_no_arb_when_sums_bracket_dollar():
    m = make_market("X-3", "E-1", 0.47, 0.52, 0.48, 0.53)
    assert mispricing.complementary_arb(m) is None


def test_threshold_filters_dust():
    m = make_market("X-4", "E-1", 0.40, 0.49, 0.48, 0.50)  # 1c edge
    assert mispricing.complementary_arb(m, threshold_cents=2.0) is None
    assert mispricing.complementary_arb(m, threshold_cents=0.5) is not None


# ---------- mispricing: cross-market event arbitrage ----------


def test_event_arb_buy_all_when_yes_asks_sum_below_dollar():
    markets = [
        make_market("E2-A", "E-2", 0.28, 0.30, 0.68, 0.70),
        make_market("E2-B", "E-2", 0.28, 0.30, 0.68, 0.70),
        make_market("E2-C", "E-2", 0.28, 0.30, 0.68, 0.70),
        make_market("E3-A", "E-3", 0.28, 0.30, 0.68, 0.70),  # different event: ignored
    ]
    arbs = mispricing.event_arbitrage(markets)
    assert len(arbs) == 1
    arb = arbs[0]
    assert arb["event_ticker"] == "E-2"
    assert arb["direction"] == "buy_all_yes"
    assert arb["edge_cents"] == pytest.approx(10.0)


def test_event_arb_sell_all_when_yes_bids_sum_above_dollar():
    markets = [
        make_market("E4-A", "E-4", 0.40, 0.42, 0.58, 0.60),
        make_market("E4-B", "E-4", 0.40, 0.42, 0.58, 0.60),
        make_market("E4-C", "E-4", 0.40, 0.42, 0.58, 0.60),
    ]
    arbs = mispricing.event_arbitrage(markets)
    assert len(arbs) == 1
    assert arbs[0]["direction"] == "sell_all_yes"
    assert arbs[0]["edge_cents"] == pytest.approx(20.0)


def test_event_arb_ignores_single_market_events():
    markets = [make_market("E5-A", "E-5", 0.10, 0.12, 0.86, 0.88)]
    assert mispricing.event_arbitrage(markets) == []


def test_find_mispricings_wrapper():
    markets = [
        make_market("X-1", "E-1", 0.38, 0.40, 0.48, 0.50),  # buy-both arb
        make_market("X-3", "E-9", 0.47, 0.52, 0.48, 0.53),  # no arb
    ]
    df = mispricing.find_mispricings(markets)
    assert len(df) == 1
    assert df.iloc[0]["ticker"] == "X-1"


# ---------- client (mocked transport) ----------


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, payload):
        self._payload = payload
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": params})
        return FakeResponse(self._payload)


def test_client_list_markets_hits_documented_endpoint():
    session = FakeSession({"markets": [make_market("X-1", "E-1", 0.4, 0.42, 0.56, 0.58)], "cursor": ""})
    c = kalshi_client.KalshiClient(session=session)
    markets = c.list_markets(limit=2)
    assert session.calls[0]["url"] == kalshi_client.KalshiClient.BASE_URL + "/markets"
    assert session.calls[0]["params"]["limit"] == 2
    assert markets[0]["ticker"] == "X-1"


def test_client_get_market_and_orderbook_urls():
    session = FakeSession({"market": make_market("X-1", "E-1", 0.4, 0.42, 0.56, 0.58)})
    c = kalshi_client.KalshiClient(session=session)
    c.get_market("X-1")
    assert session.calls[0]["url"].endswith("/markets/X-1")

    session2 = FakeSession(
        {"orderbook_fp": {"yes_dollars": [["0.40", 100]],
                          "no_dollars": [["0.60", 50]]}}
    )
    c2 = kalshi_client.KalshiClient(session=session2)
    ob = c2.get_orderbook("X-1")
    assert session2.calls[0]["url"].endswith("/markets/X-1/orderbook")
    assert ob["yes_dollars"] == [["0.40", 100]]
