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
_ALLOWED: dict[str, tuple[str, Callable[[str], Any]]] = {
    "stop_loss": ("Stop-loss percentage (0–1, e.g. 0.05 = 5%)", float),
    "trailing_stop": ("Trailing-stop percentage (0–1)", float),
    "max_buys": ("Max buy orders per tick", int),
    "sentiment_weight": (
        "WSB hype bonus multiplier (0..1, e.g. 0.45 = up to +45% score)", float
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
    if risk_kwargs:
        new_risk = replace(config.risk, **risk_kwargs)
        return replace(config, risk=new_risk)
    return config
