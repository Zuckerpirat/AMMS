from __future__ import annotations

import httpx
import respx

from amms.broker.alpaca import AlpacaClient

PAPER_URL = "https://paper-api.alpaca.markets"


@respx.mock
def test_get_clock_parses_alpaca_response() -> None:
    respx.get(f"{PAPER_URL}/v2/clock").mock(
        return_value=httpx.Response(
            200,
            json={
                "timestamp": "2026-05-13T14:00:00-04:00",
                "is_open": True,
                "next_open": "2026-05-14T09:30:00-04:00",
                "next_close": "2026-05-13T16:00:00-04:00",
            },
        )
    )
    client = AlpacaClient("k", "s", PAPER_URL)
    clock = client.get_clock()
    assert clock.is_open is True
    assert clock.timestamp.year == 2026
    assert clock.next_close.hour == 16


@respx.mock
def test_list_orders_passes_query_params() -> None:
    route = respx.get(f"{PAPER_URL}/v2/orders").mock(
        return_value=httpx.Response(200, json=[])
    )
    client = AlpacaClient("k", "s", PAPER_URL)
    out = client.list_orders(status="open", symbols=["aapl", "msft"])
    assert out == []
    assert route.called
    qs = route.calls.last.request.url.params
    assert qs["status"] == "open"
    assert qs["symbols"] == "AAPL,MSFT"
    assert qs["limit"] == "100"


@respx.mock
def test_list_orders_returns_parsed_orders() -> None:
    respx.get(f"{PAPER_URL}/v2/orders").mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "id": "o-1",
                    "client_order_id": "cid-1",
                    "symbol": "AAPL",
                    "side": "buy",
                    "qty": "1",
                    "type": "market",
                    "status": "new",
                    "submitted_at": "2026-05-13T14:00:00Z",
                }
            ],
        )
    )
    client = AlpacaClient("k", "s", PAPER_URL)
    orders = client.list_orders()
    assert len(orders) == 1
    assert orders[0].symbol == "AAPL"
    assert orders[0].side == "buy"
