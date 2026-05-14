from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import httpx

from amms.notifier.base import Notifier, NullNotifier

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TelegramNotifier:
    """Outbound-only Telegram alerts. Failures are logged, not raised, so a
    flaky Telegram never crashes the trading loop.
    """

    token: str
    chat_id: str
    timeout: float = 10.0

    def send(self, text: str) -> None:
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        try:
            resp = httpx.post(
                url,
                json={"chat_id": self.chat_id, "text": text},
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except Exception:
            logger.warning("Telegram send failed", exc_info=True)


def build_notifier() -> Notifier:
    """Return a TelegramNotifier when env is set; a NullNotifier otherwise."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return NullNotifier()
    return TelegramNotifier(token=token, chat_id=chat_id)
