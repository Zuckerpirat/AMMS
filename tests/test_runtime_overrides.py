from __future__ import annotations

import sqlite3

import pytest

from amms.config import AppConfig, SchedulerConfig, StrategyConfig
from amms.risk.rules import RiskConfig
from amms.runtime_overrides import (
    apply_to_config,
    get_overrides,
    parse_value,
    set_override,
    unset_override,
)


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    return c


def _base_config() -> AppConfig:
    return AppConfig(
        watchlist=("AAPL",),
        strategy=StrategyConfig(name="sma_cross"),
        scheduler=SchedulerConfig(),
        risk=RiskConfig(),
    )


def test_parse_value_validates_range() -> None:
    assert parse_value("stop_loss", "0.05") == 0.05
    with pytest.raises(ValueError):
        parse_value("stop_loss", "1.5")
    with pytest.raises(ValueError):
        parse_value("stop_loss", "-0.1")


def test_parse_value_rejects_unknown_key() -> None:
    with pytest.raises(ValueError):
        parse_value("some_random_key", "1")


def test_set_and_get_override_round_trip() -> None:
    conn = _conn()
    set_override(conn, "stop_loss", "0.05")
    set_override(conn, "trailing_stop", "0.10")
    overrides = get_overrides(conn)
    assert overrides == {"stop_loss": 0.05, "trailing_stop": 0.10}


def test_set_override_updates_existing() -> None:
    conn = _conn()
    set_override(conn, "stop_loss", "0.05")
    set_override(conn, "stop_loss", "0.07")
    assert get_overrides(conn) == {"stop_loss": 0.07}


def test_unset_override_removes_value() -> None:
    conn = _conn()
    set_override(conn, "stop_loss", "0.05")
    assert unset_override(conn, "stop_loss") is True
    assert get_overrides(conn) == {}


def test_unset_override_returns_false_when_missing() -> None:
    conn = _conn()
    assert unset_override(conn, "stop_loss") is False


def test_apply_to_config_replaces_risk_fields() -> None:
    conn = _conn()
    set_override(conn, "stop_loss", "0.05")
    set_override(conn, "trailing_stop", "0.10")
    set_override(conn, "max_buys", "3")
    cfg = apply_to_config(_base_config(), conn)
    assert cfg.risk.stop_loss_pct == 0.05
    assert cfg.risk.trailing_stop_pct == 0.10
    assert cfg.risk.max_buys_per_tick == 3


def test_apply_to_config_returns_same_when_no_overrides() -> None:
    conn = _conn()
    base = _base_config()
    cfg = apply_to_config(base, conn)
    assert cfg is base
