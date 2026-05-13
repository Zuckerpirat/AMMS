from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

PAPER_ENV = {
    "ALPACA_API_KEY": "test-key",
    "ALPACA_API_SECRET": "test-secret",
    "ALPACA_BASE_URL": "https://paper-api.alpaca.markets",
    "ALPACA_DATA_URL": "https://data.alpaca.markets",
    "AMMS_LOG_LEVEL": "DEBUG",
}


@pytest.fixture
def paper_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[dict[str, str]]:
    env = {**PAPER_ENV, "AMMS_DB_PATH": str(tmp_path / "amms.sqlite")}
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    yield env
