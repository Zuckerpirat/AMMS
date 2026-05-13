from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Literal

import httpx

from amms.clock import ClockStatus, parse_alpaca_dt
from amms.config import PAPER_HOST_MARKER

Side = Literal["buy", "sell"]
OrderType = Literal["market", "limit"]
TimeInForce = Literal["day", "gtc"]

ALLOWED_SIDES: frozenset[str] = frozenset({"buy", "sell"})


class AlpacaError(RuntimeError):
    """Raised when Alpaca returns a non-2xx response."""


@dataclass(frozen=True)
class Account:
    equity: float
    cash: float
    buying_power: float
    status: str
    raw: dict[str, Any]


@dataclass(frozen=True)
class Position:
    symbol: str
    qty: float
    avg_entry_price: float
    market_value: float
    unrealized_pl: float
    raw: dict[str, Any]


@dataclass(frozen=True)
class Order:
    id: str
    client_order_id: str
    symbol: str
    side: Side
    qty: float
    type: OrderType
    status: str
    submitted_at: str
    filled_at: str | None
    filled_avg_price: float | None
    raw: dict[str, Any]


class AlpacaClient:
    """Thin synchronous Alpaca paper-trading client.

    The constructor refuses any base URL that is not a paper endpoint. This is
    the second of two guards (the first lives in `amms.config`) that physically
    prevents the bot from talking to live trading.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str,
        *,
        timeout: float = 10.0,
        client: httpx.Client | None = None,
    ) -> None:
        if PAPER_HOST_MARKER not in base_url:
            raise RuntimeError(
                f"AlpacaClient refuses to start: base_url must contain "
                f"{PAPER_HOST_MARKER!r}. Got: {base_url!r}"
            )
        self._base_url = base_url.rstrip("/")
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=timeout,
            headers={
                "APCA-API-KEY-ID": api_key,
                "APCA-API-SECRET-KEY": api_secret,
                "Content-Type": "application/json",
                "User-Agent": "amms/0.1",
            },
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> AlpacaClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        url = f"{self._base_url}{path}"
        resp = self._client.request(method, url, json=json_body, params=params)
        if resp.status_code >= 400:
            raise AlpacaError(
                f"Alpaca {method} {path} -> {resp.status_code}: {resp.text}"
            )
        if resp.status_code == 204 or not resp.content:
            return None
        return resp.json()

    def get_account(self) -> Account:
        data = self._request("GET", "/v2/account")
        return Account(
            equity=float(data["equity"]),
            cash=float(data["cash"]),
            buying_power=float(data["buying_power"]),
            status=data["status"],
            raw=data,
        )

    def get_positions(self) -> list[Position]:
        data = self._request("GET", "/v2/positions") or []
        return [
            Position(
                symbol=p["symbol"],
                qty=float(p["qty"]),
                avg_entry_price=float(p["avg_entry_price"]),
                market_value=float(p["market_value"]),
                unrealized_pl=float(p["unrealized_pl"]),
                raw=p,
            )
            for p in data
        ]

    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: Side,
        *,
        order_type: OrderType = "market",
        time_in_force: TimeInForce = "day",
        client_order_id: str | None = None,
    ) -> Order:
        if side not in ALLOWED_SIDES:
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
        if qty <= 0:
            raise ValueError(f"qty must be > 0, got {qty}")
        body: dict[str, Any] = {
            "symbol": symbol.upper(),
            "qty": str(qty),
            "side": side,
            "type": order_type,
            "time_in_force": time_in_force,
            "client_order_id": client_order_id or f"amms-{uuid.uuid4()}",
        }
        data = self._request("POST", "/v2/orders", json_body=body)
        return self._order_from_payload(data)

    def get_order(self, order_id: str) -> Order:
        data = self._request("GET", f"/v2/orders/{order_id}")
        return self._order_from_payload(data)

    def cancel_order(self, order_id: str) -> None:
        self._request("DELETE", f"/v2/orders/{order_id}")

    def list_orders(
        self,
        *,
        status: str = "open",
        symbols: list[str] | None = None,
        limit: int = 100,
    ) -> list[Order]:
        params: dict[str, Any] = {"status": status, "limit": limit}
        if symbols:
            params["symbols"] = ",".join(s.upper() for s in symbols)
        data = self._request("GET", "/v2/orders", params=params) or []
        return [self._order_from_payload(d) for d in data]

    def get_clock(self) -> ClockStatus:
        data = self._request("GET", "/v2/clock")
        return ClockStatus(
            timestamp=parse_alpaca_dt(data["timestamp"]),
            is_open=bool(data["is_open"]),
            next_open=parse_alpaca_dt(data["next_open"]),
            next_close=parse_alpaca_dt(data["next_close"]),
        )

    @staticmethod
    def _order_from_payload(data: dict[str, Any]) -> Order:
        filled_avg = data.get("filled_avg_price")
        return Order(
            id=data["id"],
            client_order_id=data["client_order_id"],
            symbol=data["symbol"],
            side=data["side"],
            qty=float(data["qty"]),
            type=data["type"],
            status=data["status"],
            submitted_at=data["submitted_at"],
            filled_at=data.get("filled_at"),
            filled_avg_price=float(filled_avg) if filled_avg is not None else None,
            raw=data,
        )
