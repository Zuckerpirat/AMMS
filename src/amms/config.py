from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PAPER_HOST_MARKER = "paper-api"


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or unsafe."""


@dataclass(frozen=True)
class Settings:
    alpaca_api_key: str
    alpaca_api_secret: str
    alpaca_base_url: str
    alpaca_data_url: str
    db_path: Path
    log_level: str


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ConfigError(f"Missing required environment variable: {name}")
    return value


def load_settings(env_file: Path | None = None) -> Settings:
    """Load settings from environment variables (and an optional .env file).

    Refuses to return unless ALPACA_BASE_URL points at the paper endpoint.
    This is the single hard guarantee that the bot can never hit live trading.
    """
    if env_file is not None:
        if not env_file.exists():
            raise ConfigError(f"Env file not found: {env_file}")
        load_dotenv(env_file, override=False)
    else:
        load_dotenv(override=False)

    base_url = _require("ALPACA_BASE_URL")
    if PAPER_HOST_MARKER not in base_url:
        raise ConfigError(
            "Refusing to start: ALPACA_BASE_URL must point at the paper endpoint "
            f"(contain {PAPER_HOST_MARKER!r}). Got: {base_url!r}"
        )

    return Settings(
        alpaca_api_key=_require("ALPACA_API_KEY"),
        alpaca_api_secret=_require("ALPACA_API_SECRET"),
        alpaca_base_url=base_url.rstrip("/"),
        alpaca_data_url=os.environ.get(
            "ALPACA_DATA_URL", "https://data.alpaca.markets"
        ).rstrip("/"),
        db_path=Path(os.environ.get("AMMS_DB_PATH", "amms.sqlite")),
        log_level=os.environ.get("AMMS_LOG_LEVEL", "INFO").upper(),
    )
