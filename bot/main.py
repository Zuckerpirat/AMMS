import logging
import uvicorn
from bot.db.models import init_db
from bot.scheduler import build_scheduler
from bot.api.server import app, is_paused
from bot.broker import alpaca as broker
from bot.data import market
from bot.strategy.momentum import MomentumStrategy
from bot.risk import manager as risk
from bot.db import repository
from bot.alerts import telegram
from config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_strategy = MomentumStrategy()


def morning_scan() -> None:
    if is_paused():
        logger.info("Bot paused — skipping morning scan")
        return

    account = broker.get_account()
    equity = float(account.equity)
    open_positions = broker.get_positions()
    held = {p.symbol for p in open_positions}

    candidates = [s for s in settings.WATCHLIST if s not in held]
    signals = _strategy.generate_signals(candidates)

    for signal in signals:
        if signal.side != "buy":
            continue

        snap = market.get_snapshots([signal.symbol]).get(signal.symbol, {})
        price = snap.get("price")
        if not price:
            continue

        open_positions = broker.get_positions()
        if not risk.can_buy(signal.symbol, price, equity, len(open_positions)):
            continue

        qty = risk.calculate_qty(price, equity)
        if qty < 1:
            continue

        order = broker.place_market_order(signal.symbol, qty, "buy")
        trade_id = repository.insert_trade(
            signal.symbol, "buy", qty, price, str(order.id), signal.reason
        )
        repository.update_trade_status(trade_id, "submitted")
        telegram.send_trade_alert(signal.symbol, "buy", qty, price, signal.reason)


def close_scan() -> None:
    if is_paused():
        logger.info("Bot paused — skipping close scan")
        return

    for pos in broker.get_positions():
        entry = float(pos.avg_entry_price)
        current = float(pos.current_price)
        qty = int(float(pos.qty))

        reason: str | None = None
        if risk.should_stop_loss(entry, current):
            reason = f"stop loss triggered (entry={entry:.2f}, now={current:.2f})"
        elif risk.should_take_profit(entry, current):
            reason = f"take profit triggered (entry={entry:.2f}, now={current:.2f})"

        if reason:
            order = broker.place_market_order(pos.symbol, qty, "sell")
            trade_id = repository.insert_trade(
                pos.symbol, "sell", qty, current, str(order.id), reason
            )
            repository.update_trade_status(trade_id, "submitted")
            telegram.send_trade_alert(pos.symbol, "sell", qty, current, reason)


def daily_summary() -> None:
    account = broker.get_account()
    positions = broker.get_positions()
    equity = float(account.equity)
    cash = float(account.cash)

    prior = repository.get_snapshots(limit=2)
    prior_equity = prior[1]["equity"] if len(prior) >= 2 else equity
    daily_pnl = equity - prior_equity

    repository.insert_snapshot(equity, cash, len(positions))
    telegram.send_daily_summary(equity, cash, positions, daily_pnl)
    logger.info("Daily summary: equity=%.2f  pnl=%.2f  positions=%d", equity, daily_pnl, len(positions))


if __name__ == "__main__":
    logger.info("Starting AMMS paper trading bot")
    init_db()
    scheduler = build_scheduler(morning_scan, close_scan, daily_summary)
    scheduler.start()
    logger.info("Scheduler started — jobs: %s", [j.id for j in scheduler.get_jobs()])
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)
