import logging
from fastapi import FastAPI
from bot.broker import alpaca as broker
from bot.db import repository

logger = logging.getLogger(__name__)

app = FastAPI(title="AMMS Paper Trading Bot", version="0.1.0")

_paused: bool = False


def is_paused() -> bool:
    return _paused


@app.get("/status")
def status():
    clock = broker.get_clock()
    account = broker.get_account()
    return {
        "paused": _paused,
        "market_open": clock.is_open,
        "next_open": str(clock.next_open),
        "next_close": str(clock.next_close),
        "equity": float(account.equity),
        "cash": float(account.cash),
        "buying_power": float(account.buying_power),
    }


@app.get("/positions")
def positions():
    return [
        {
            "symbol": p.symbol,
            "qty": float(p.qty),
            "avg_entry_price": float(p.avg_entry_price),
            "current_price": float(p.current_price),
            "unrealized_pl": float(p.unrealized_pl),
            "unrealized_plpc": float(p.unrealized_plpc),
        }
        for p in broker.get_positions()
    ]


@app.get("/trades")
def trades(limit: int = 50):
    return repository.get_recent_trades(limit)


@app.get("/snapshots")
def snapshots(limit: int = 30):
    return repository.get_snapshots(limit)


@app.post("/pause")
def pause():
    global _paused
    _paused = True
    logger.info("Bot paused via API")
    return {"paused": True}


@app.post("/resume")
def resume():
    global _paused
    _paused = False
    logger.info("Bot resumed via API")
    return {"paused": False}
