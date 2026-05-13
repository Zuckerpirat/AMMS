from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

from amms import db
from amms.broker import AlpacaClient
from amms.config import AppConfig, SchedulerConfig, StrategyConfig
from amms.data import MarketDataClient
from amms.executor import build_daily_summary, run_tick
from amms.risk import RiskConfig
from amms.strategy import SmaCross

PAPER_URL = "https://paper-api.alpaca.markets"
DATA_URL = "https://data.alpaca.markets"


def _config(watchlist=("AAPL",)) -> AppConfig:
    return AppConfig(
        watchlist=watchlist,
        strategy=StrategyConfig(name="sma_cross", params={"fast": 3, "slow": 5}),
        risk=RiskConfig(
            max_open_positions=5,
            max_position_pct=0.02,
            daily_loss_pct=-0.03,
        ),
        scheduler=SchedulerConfig(tick_seconds=60),
    )


def _bars_payload(symbol: str, closes: list[float]) -> dict:
    return {
        "bars": {
            symbol.upper(): [
                {
                    "t": f"2025-01-{i + 1:02d}T05:00:00Z",
                    "o": c, "h": c, "l": c, "c": c, "v": 100,
                }
                for i, c in enumerate(closes)
            ]
        },
        "next_page_token": None,
    }


def _account_payload(equity: float = 100_000, cash: float = 100_000) -> dict:
    return {
        "equity": str(equity),
        "cash": str(cash),
        "buying_power": str(cash),
        "status": "ACTIVE",
    }


@respx.mock
def test_run_tick_dry_run_does_not_post_orders(tmp_path: Path) -> None:
    respx.get(f"{PAPER_URL}/v2/account").mock(
        return_value=httpx.Response(200, json=_account_payload())
    )
    respx.get(f"{PAPER_URL}/v2/positions").mock(return_value=httpx.Response(200, json=[]))
    respx.get(f"{PAPER_URL}/v2/orders").mock(return_value=httpx.Response(200, json=[]))
    respx.get(f"{DATA_URL}/v2/stocks/bars").mock(
        return_value=httpx.Response(200, json=_bars_payload("AAPL", [1] * 6 + [10]))
    )
    orders_post = respx.post(f"{PAPER_URL}/v2/orders")

    conn = db.connect(tmp_path / "x.sqlite")
    db.migrate(conn)
    with (
        AlpacaClient("k", "s", PAPER_URL) as broker,
        MarketDataClient("k", "s", DATA_URL) as data,
    ):
        result = run_tick(
            broker=broker,
            data=data,
            conn=conn,
            config=_config(),
            strategy=SmaCross(fast=3, slow=5),
            execute=False,
        )

    assert not orders_post.called
    assert len(result.signals) == 1
    assert result.signals[0].kind == "buy"
    assert "would buy" in result.blocked[0][1]
    snap = conn.execute("SELECT equity FROM equity_snapshots").fetchone()
    assert snap[0] == pytest.approx(100_000.0)
    conn.close()


@respx.mock
def test_run_tick_execute_places_order(tmp_path: Path) -> None:
    respx.get(f"{PAPER_URL}/v2/account").mock(
        return_value=httpx.Response(200, json=_account_payload())
    )
    respx.get(f"{PAPER_URL}/v2/positions").mock(return_value=httpx.Response(200, json=[]))
    respx.get(f"{PAPER_URL}/v2/orders").mock(return_value=httpx.Response(200, json=[]))
    respx.get(f"{DATA_URL}/v2/stocks/bars").mock(
        return_value=httpx.Response(200, json=_bars_payload("AAPL", [1] * 6 + [10]))
    )
    orders_post = respx.post(f"{PAPER_URL}/v2/orders").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "ord-1",
                "client_order_id": "cid-1",
                "symbol": "AAPL",
                "side": "buy",
                "qty": "200",
                "type": "market",
                "status": "accepted",
                "submitted_at": "2026-05-13T14:00:00Z",
            },
        )
    )

    conn = db.connect(tmp_path / "x.sqlite")
    db.migrate(conn)
    with (
        AlpacaClient("k", "s", PAPER_URL) as broker,
        MarketDataClient("k", "s", DATA_URL) as data,
    ):
        result = run_tick(
            broker=broker,
            data=data,
            conn=conn,
            config=_config(),
            strategy=SmaCross(fast=3, slow=5),
            execute=True,
        )

    assert orders_post.called
    assert result.placed_order_ids == ["ord-1"]
    row = conn.execute("SELECT symbol, side, status FROM orders").fetchone()
    assert tuple(row) == ("AAPL", "buy", "accepted")
    conn.close()


@respx.mock
def test_run_tick_skips_when_pending_order_exists(tmp_path: Path) -> None:
    respx.get(f"{PAPER_URL}/v2/account").mock(
        return_value=httpx.Response(200, json=_account_payload())
    )
    respx.get(f"{PAPER_URL}/v2/positions").mock(return_value=httpx.Response(200, json=[]))
    respx.get(f"{PAPER_URL}/v2/orders").mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "id": "pending-1",
                    "client_order_id": "cid-x",
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
    respx.get(f"{DATA_URL}/v2/stocks/bars").mock(
        return_value=httpx.Response(200, json=_bars_payload("AAPL", [1] * 6 + [10]))
    )
    orders_post = respx.post(f"{PAPER_URL}/v2/orders")

    conn = db.connect(tmp_path / "x.sqlite")
    db.migrate(conn)
    with (
        AlpacaClient("k", "s", PAPER_URL) as broker,
        MarketDataClient("k", "s", DATA_URL) as data,
    ):
        result = run_tick(
            broker=broker,
            data=data,
            conn=conn,
            config=_config(),
            strategy=SmaCross(fast=3, slow=5),
            execute=True,
        )

    assert not orders_post.called
    assert result.placed_order_ids == []
    assert result.blocked == [("AAPL", "open buy order already pending")]
    conn.close()


@respx.mock
def test_build_daily_summary_includes_equity_and_counts(tmp_path: Path) -> None:
    respx.get(f"{PAPER_URL}/v2/account").mock(
        return_value=httpx.Response(200, json=_account_payload(equity=123_456))
    )
    respx.get(f"{PAPER_URL}/v2/positions").mock(return_value=httpx.Response(200, json=[]))

    conn = db.connect(tmp_path / "x.sqlite")
    db.migrate(conn)
    with AlpacaClient("k", "s", PAPER_URL) as broker:
        text = build_daily_summary(broker, conn)
    assert "amms daily summary" in text
    assert "$123,456" in text
    assert "Open positions: 0" in text
    assert "Orders today:   0" in text
    conn.close()
