from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from amms.filters.universe import UniverseFilter
from amms.risk.rules import RiskConfig

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


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    timeframe: str = "1Day"

    def __post_init__(self) -> None:
        if not self.timeframe.strip():
            raise ConfigError("strategy.timeframe must not be empty")


@dataclass(frozen=True)
class SchedulerConfig:
    tick_seconds: int = 60
    timezone: str = "America/New_York"


@dataclass(frozen=True)
class AppConfig:
    watchlist: tuple[str, ...]
    strategy: StrategyConfig
    risk: RiskConfig
    scheduler: SchedulerConfig
    universe: UniverseFilter = field(default_factory=UniverseFilter)

    def __post_init__(self) -> None:
        if not self.watchlist:
            raise ConfigError("watchlist must not be empty")
        seen: set[str] = set()
        for sym in self.watchlist:
            up = sym.upper()
            if up in seen:
                raise ConfigError(f"duplicate symbol in watchlist: {sym}")
            seen.add(up)


def _config_path_from_env(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    env = os.environ.get("AMMS_CONFIG_PATH", "").strip()
    return Path(env) if env else Path("config.yaml")


def load_app_config(path: Path | None = None) -> AppConfig:
    """Load `config.yaml` (watchlist, strategy, risk, scheduler)."""
    config_path = _config_path_from_env(path)
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    watchlist_raw = raw.get("watchlist")
    if not isinstance(watchlist_raw, list) or not all(isinstance(s, str) for s in watchlist_raw):
        raise ConfigError("watchlist must be a list of ticker strings")
    watchlist = tuple(s.upper() for s in watchlist_raw)

    strategy_raw = raw.get("strategy") or {}
    if "name" not in strategy_raw:
        raise ConfigError("strategy.name is required")
    strategy = StrategyConfig(
        name=str(strategy_raw["name"]),
        params=dict(strategy_raw.get("params") or {}),
        timeframe=str(strategy_raw.get("timeframe", "1Day")),
    )

    risk_raw = raw.get("risk") or {}
    max_buys_raw = risk_raw.get("max_buys_per_tick")
    max_buys_per_tick = int(max_buys_raw) if max_buys_raw is not None else None
    risk = RiskConfig(
        max_open_positions=int(risk_raw.get("max_open_positions", 5)),
        max_position_pct=float(risk_raw.get("max_position_pct", 0.02)),
        daily_loss_pct=float(risk_raw.get("daily_loss_pct", -0.03)),
        blackout_minutes_after_open=int(risk_raw.get("blackout_minutes_after_open", 5)),
        blackout_minutes_before_close=int(risk_raw.get("blackout_minutes_before_close", 5)),
        max_buys_per_tick=max_buys_per_tick,
        min_hold_days=int(risk_raw.get("min_hold_days", 0)),
    )

    sched_raw = raw.get("scheduler") or {}
    scheduler = SchedulerConfig(
        tick_seconds=int(sched_raw.get("tick_seconds", 60)),
        timezone=str(sched_raw.get("timezone", "America/New_York")),
    )

    universe_raw = raw.get("universe") or {}
    max_price_raw = universe_raw.get("max_price")
    universe = UniverseFilter(
        min_price=float(universe_raw.get("min_price", 0.0)),
        max_price=float(max_price_raw) if max_price_raw is not None else None,
        min_avg_dollar_volume=float(universe_raw.get("min_avg_dollar_volume", 0.0)),
        adv_lookback=int(universe_raw.get("adv_lookback", 20)),
    )

    return AppConfig(
        watchlist=watchlist,
        strategy=strategy,
        risk=risk,
        scheduler=scheduler,
        universe=universe,
    )
