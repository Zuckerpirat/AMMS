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
def test_run_tick_persists_features(tmp_path: Path) -> None:
    respx.get(f"{PAPER_URL}/v2/account").mock(
        return_value=httpx.Response(200, json=_account_payload())
    )
    respx.get(f"{PAPER_URL}/v2/positions").mock(return_value=httpx.Response(200, json=[]))
    respx.get(f"{PAPER_URL}/v2/orders").mock(return_value=httpx.Response(200, json=[]))
    # 25 bars so all standard features can be computed.
    closes = [100.0 + i for i in range(25)]
    respx.get(f"{DATA_URL}/v2/stocks/bars").mock(
        return_value=httpx.Response(200, json=_bars_payload("AAPL", closes))
    )

    conn = db.connect(tmp_path / "x.sqlite")
    db.migrate(conn)
    with (
        AlpacaClient("k", "s", PAPER_URL) as broker,
        MarketDataClient("k", "s", DATA_URL) as data,
    ):
        run_tick(
            broker=broker,
            data=data,
            conn=conn,
            config=_config(),
            strategy=SmaCross(fast=3, slow=5),
            execute=False,
        )
    rows = conn.execute(
        "SELECT name FROM features WHERE symbol=?", ("AAPL",)
    ).fetchall()
    names = {r["name"] for r in rows}
    assert {"momentum_20d", "rsi_14", "atr_14", "realized_vol_20d", "rvol_20"} <= names
    conn.close()


class _ScoredFakeStrategy:
    """Inline strategy whose evaluate() returns a controlled per-symbol buy score."""

    name = "fake_scored"

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores

    @property
    def lookback(self) -> int:
        return 1

    def evaluate(self, symbol, bars):
        from amms.strategy.base import Signal

        return Signal(
            symbol=symbol,
            kind="buy",
            reason="fake",
            price=bars[-1].close,
            score=self._scores.get(symbol, 0.0),
        )


@respx.mock
def test_run_tick_top_n_keeps_highest_scoring_buys(tmp_path: Path) -> None:
    respx.get(f"{PAPER_URL}/v2/account").mock(
        return_value=httpx.Response(200, json=_account_payload())
    )
    respx.get(f"{PAPER_URL}/v2/positions").mock(return_value=httpx.Response(200, json=[]))
    respx.get(f"{PAPER_URL}/v2/orders").mock(return_value=httpx.Response(200, json=[]))

    def _bars_route(request):
        symbol = request.url.params["symbols"]
        payload = _bars_payload(symbol, [10.0])
        return httpx.Response(200, json=payload)

    respx.get(f"{DATA_URL}/v2/stocks/bars").mock(side_effect=_bars_route)
    orders_post = respx.post(f"{PAPER_URL}/v2/orders")
    orders_post.mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "ord-x",
                "client_order_id": "cid-x",
                "symbol": "MSFT",
                "side": "buy",
                "qty": "1",
                "type": "market",
                "status": "accepted",
                "submitted_at": "2026-05-13T14:00:00Z",
            },
        )
    )

    cfg = AppConfig(
        watchlist=("AAPL", "MSFT", "NVDA"),
        strategy=StrategyConfig(name="sma_cross", params={"fast": 3, "slow": 5}),
        risk=RiskConfig(
            max_open_positions=5,
            max_position_pct=0.5,  # generous so risk doesn't block sizing
            daily_loss_pct=-0.99,
            max_buys_per_tick=1,
        ),
        scheduler=SchedulerConfig(tick_seconds=60),
    )

    conn = db.connect(tmp_path / "x.sqlite")
    db.migrate(conn)
    fake = _ScoredFakeStrategy({"AAPL": 1.0, "MSFT": 5.0, "NVDA": 3.0})
    with (
        AlpacaClient("k", "s", PAPER_URL) as broker,
        MarketDataClient("k", "s", DATA_URL) as data,
    ):
        result = run_tick(
            broker=broker,
            data=data,
            conn=conn,
            config=cfg,
            strategy=fake,
            execute=True,
        )
    conn.close()

    # Only the highest-scoring symbol (MSFT, score=5.0) was submitted.
    assert orders_post.call_count == 1
    assert result.placed_order_ids == ["ord-x"]
    blocked_symbols = {s for s, _ in result.blocked}
    assert blocked_symbols == {"AAPL", "NVDA"}


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
