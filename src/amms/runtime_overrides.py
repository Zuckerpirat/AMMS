"""Runtime config overrides for safe values the user may want to tune live.

Only a small whitelist of keys is exposed via Telegram so the user can
enable/tune safety features (stop-loss, trailing stop, max buys) without
SSH-ing in to edit ``config.yaml``. Overrides live in SQLite and survive
restarts. The scheduler applies them on top of the on-disk config at the
start of every tick.

Adding a new key:
  1. Add an entry to ``_ALLOWED`` with the dotted-path and parser.
  2. Update ``apply_to_config`` to route the value into the right field.
  3. Cover the new path with a test.
"""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any, Callable

from amms.config import AppConfig


# Whitelist of user-tunable settings.
# key (telegram-facing) -> (description, parser)
def _to_bool(raw: str) -> bool:
    s = raw.strip().lower()
    if s in {"1", "true", "yes", "on", "y"}:
        return True
    if s in {"0", "false", "no", "off", "n"}:
        return False
    raise ValueError(f"expected 1/0/true/false, got {raw!r}")


_ALLOWED: dict[str, tuple[str, Callable[[str], Any]]] = {
    "stop_loss": ("Stop-loss percentage (0–1, e.g. 0.05 = 5%)", float),
    "trailing_stop": ("Trailing-stop percentage (0–1)", float),
    "max_buys": ("Max buy orders per tick", int),
    "sentiment_weight": (
        "WSB hype bonus multiplier (0..1, e.g. 0.45 = up to +45% score)", float
    ),
    "wsb_enabled": ("Enable WSB Auto-Discovery watchlist expansion (1/0)", _to_bool),
    "wsb_top_n": ("Max tickers WSB Auto-Discovery may add", int),
    "wsb_min_mentions": ("Minimum WSB mentions needed to auto-add a ticker", int),
    "macro_enabled": ("Pause buys on VIXY stress (1/0)", _to_bool),
    "macro_day_threshold": (
        "VIXY 1d %% move that triggers macro-pause (default 5.0)", float
    ),
    "macro_week_threshold": (
        "VIXY 1w %% move that triggers macro-pause (default 15.0)", float
    ),
    "drawdown_alert": (
        "Drawdown %% below 30d peak that triggers an alert (default 5.0)", float
    ),
    "trading_mode": (
        "Active strategy mode: conservative | swing | meme | event", str
    ),
    "max_sector_pct": (
        "Max share of portfolio any single sector may hold (0..1, 0=off)", float
    ),
}


def allowed_keys() -> dict[str, str]:
    """Return user-facing key -> description mapping."""
    return {k: desc for k, (desc, _parser) in _ALLOWED.items()}


def parse_value(key: str, raw: str) -> Any:
    """Validate and convert a string value for a whitelisted key.

    Raises ``ValueError`` on unknown key or invalid value.
    """
    if key not in _ALLOWED:
        raise ValueError(f"unknown key '{key}'. Allowed: {', '.join(_ALLOWED)}")
    _, parser = _ALLOWED[key]
    try:
        value = parser(raw)
    except ValueError as e:
        raise ValueError(f"invalid value for {key}: {e}") from e
    if key in ("stop_loss", "trailing_stop"):
        if not 0 <= value < 1:
            raise ValueError(f"{key} must be in [0, 1), got {value}")
    if key == "sentiment_weight":
        if not 0 <= value <= 1:
            raise ValueError(f"sentiment_weight must be in [0, 1], got {value}")
    if key == "max_buys" and value < 0:
        raise ValueError("max_buys must be >= 0")
    if key == "wsb_top_n" and value <= 0:
        raise ValueError("wsb_top_n must be > 0")
    if key == "wsb_min_mentions" and value <= 0:
        raise ValueError("wsb_min_mentions must be > 0")
    if key in ("macro_day_threshold", "macro_week_threshold") and value <= 0:
        raise ValueError(f"{key} must be > 0")
    if key == "drawdown_alert" and value <= 0:
        raise ValueError("drawdown_alert must be > 0")
    if key == "trading_mode":
        allowed_modes = {"conservative", "swing", "meme", "event"}
        if value not in allowed_modes:
            raise ValueError(
                f"trading_mode must be one of: {', '.join(sorted(allowed_modes))}"
            )
    if key == "max_sector_pct":
        if not 0 <= value <= 1:
            raise ValueError("max_sector_pct must be in [0, 1]")
    return value


def ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS runtime_overrides ("
        "key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL"
        ")"
    )
    conn.commit()


def set_override(conn: sqlite3.Connection, key: str, raw_value: str) -> Any:
    """Validate and persist an override. Returns the parsed value."""
    value = parse_value(key, raw_value)
    ensure_table(conn)
    conn.execute(
        "INSERT INTO runtime_overrides(key, value, updated_at) VALUES (?, ?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
        "updated_at = excluded.updated_at",
        (key, str(value), datetime.now(UTC).isoformat()),
    )
    conn.commit()
    return value


def unset_override(conn: sqlite3.Connection, key: str) -> bool:
    """Remove a single override. Returns True if a row was deleted."""
    if key not in _ALLOWED:
        raise ValueError(f"unknown key '{key}'")
    ensure_table(conn)
    cur = conn.execute("DELETE FROM runtime_overrides WHERE key = ?", (key,))
    conn.commit()
    return cur.rowcount > 0


def get_overrides(conn: sqlite3.Connection) -> dict[str, Any]:
    """Return all currently active overrides parsed into their target types."""
    ensure_table(conn)
    rows = conn.execute(
        "SELECT key, value FROM runtime_overrides"
    ).fetchall()
    out: dict[str, Any] = {}
    for row in rows:
        try:
            key = row["key"] if isinstance(row, sqlite3.Row) else row[0]
            raw = row["value"] if isinstance(row, sqlite3.Row) else row[1]
        except (KeyError, IndexError):
            continue
        if key in _ALLOWED:
            try:
                out[key] = parse_value(key, raw)
            except ValueError:
                continue
    return out


def apply_to_strategy(strategy, conn: sqlite3.Connection):
    """Return the strategy with relevant overrides applied.

    Currently overrides ``sentiment_weight`` on any strategy that has
    that field (e.g. CompositeStrategy). Other strategy types are
    returned unchanged.
    """
    overrides = get_overrides(conn)
    if (
        "sentiment_weight" in overrides
        and hasattr(strategy, "sentiment_weight")
    ):
        try:
            return replace(strategy, sentiment_weight=overrides["sentiment_weight"])
        except (TypeError, ValueError):
            return strategy
    return strategy


def apply_to_config(config: AppConfig, conn: sqlite3.Connection) -> AppConfig:
    """Return a new AppConfig with stored overrides applied on top."""
    overrides = get_overrides(conn)
    if not overrides:
        return config
    risk_kwargs: dict[str, Any] = {}
    if "stop_loss" in overrides:
        risk_kwargs["stop_loss_pct"] = overrides["stop_loss"]
    if "trailing_stop" in overrides:
        risk_kwargs["trailing_stop_pct"] = overrides["trailing_stop"]
    if "max_buys" in overrides:
        risk_kwargs["max_buys_per_tick"] = overrides["max_buys"]
    if "max_sector_pct" in overrides:
        risk_kwargs["max_sector_pct"] = overrides["max_sector_pct"]
    wsb_kwargs: dict[str, Any] = {}
    if "wsb_enabled" in overrides:
        wsb_kwargs["enabled"] = overrides["wsb_enabled"]
    if "wsb_top_n" in overrides:
        wsb_kwargs["top_n"] = overrides["wsb_top_n"]
    if "wsb_min_mentions" in overrides:
        wsb_kwargs["min_mentions"] = overrides["wsb_min_mentions"]

    new_config = config
    if risk_kwargs:
        new_config = replace(new_config, risk=replace(new_config.risk, **risk_kwargs))
    if wsb_kwargs:
        new_config = replace(
            new_config,
            wsb_discovery=replace(new_config.wsb_discovery, **wsb_kwargs),
        )
    return new_config
