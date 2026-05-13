import logging
import httpx
from config import settings

logger = logging.getLogger(__name__)

_BASE = f"https://api.telegram.org/bot{settings.TELEGRAM_TOKEN}"


def send_message(text: str) -> None:
    if not settings.TELEGRAM_ENABLED:
        return
    try:
        httpx.post(
            f"{_BASE}/sendMessage",
            json={
                "chat_id": settings.TELEGRAM_CHAT_ID,
                "text": text,
                "parse_mode": "HTML",
            },
            timeout=10,
        )
    except Exception as exc:
        logger.warning("Telegram send failed: %s", exc)


def send_trade_alert(symbol: str, side: str, qty: int, price: float, reason: str) -> None:
    icon = "BUY" if side == "buy" else "SELL"
    text = (
        f"<b>[{icon}] {symbol}</b>\n"
        f"Qty: {qty}  |  Price: ${price:.2f}\n"
        f"Reason: {reason}"
    )
    send_message(text)


def send_daily_summary(
    equity: float, cash: float, positions: list, daily_pnl: float
) -> None:
    direction = "+" if daily_pnl >= 0 else ""
    text = (
        f"<b>Daily Summary</b>\n"
        f"Equity:     ${equity:,.2f}\n"
        f"Cash:       ${cash:,.2f}\n"
        f"Positions:  {len(positions)}\n"
        f"Daily P&amp;L:  {direction}${daily_pnl:,.2f}"
    )
    send_message(text)
