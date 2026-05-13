from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Notifier(Protocol):
    def send(self, text: str) -> None:
        ...


class NullNotifier:
    """No-op notifier used when Telegram is not configured."""

    def send(self, text: str) -> None:
        return None
