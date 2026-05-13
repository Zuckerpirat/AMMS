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

MIN_CONFIG_YAML = """
watchlist:
  - AAPL
strategy:
  name: sma_cross
  params:
    fast: 3
    slow: 5
risk:
  max_open_positions: 5
  max_position_pct: 0.02
  daily_loss_pct: -0.03
scheduler:
  tick_seconds: 60
"""


@pytest.fixture
def paper_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[dict[str, str]]:
    env = {**PAPER_ENV, "AMMS_DB_PATH": str(tmp_path / "amms.sqlite")}
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    yield env


@pytest.fixture
def paper_env_with_config(
    paper_env: dict[str, str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[dict[str, str]]:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(MIN_CONFIG_YAML, encoding="utf-8")
    monkeypatch.setenv("AMMS_CONFIG_PATH", str(config_path))
    yield {**paper_env, "AMMS_CONFIG_PATH": str(config_path)}
