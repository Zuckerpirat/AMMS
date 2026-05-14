from __future__ import annotations

import sqlite3

import pytest

from amms.config import AppConfig, SchedulerConfig, StrategyConfig
from amms.data.wsb_discovery import WSBDiscoveryConfig
from amms.risk.rules import RiskConfig
from amms.runtime_overrides import (
    apply_to_config,
    apply_to_strategy,
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


def test_apply_to_strategy_overrides_sentiment_weight() -> None:
    from amms.strategy.composite import CompositeStrategy

    conn = _conn()
    set_override(conn, "sentiment_weight", "0.45")
    strat = CompositeStrategy(sentiment_weight=0.0)
    new_strat = apply_to_strategy(strat, conn)
    assert new_strat.sentiment_weight == 0.45


def test_apply_to_strategy_no_op_when_strategy_lacks_field() -> None:
    class _NoSentimentStrategy:
        name = "x"

    conn = _conn()
    set_override(conn, "sentiment_weight", "0.3")
    strat = _NoSentimentStrategy()
    assert apply_to_strategy(strat, conn) is strat


def test_apply_to_config_overrides_wsb_discovery_fields() -> None:
    conn = _conn()
    set_override(conn, "wsb_enabled", "1")
    set_override(conn, "wsb_top_n", "12")
    set_override(conn, "wsb_min_mentions", "100")
    cfg = apply_to_config(_base_config(), conn)
    assert cfg.wsb_discovery.enabled is True
    assert cfg.wsb_discovery.top_n == 12
    assert cfg.wsb_discovery.min_mentions == 100


def test_parse_wsb_enabled_accepts_truthy_and_falsy() -> None:
    assert parse_value("wsb_enabled", "1") is True
    assert parse_value("wsb_enabled", "true") is True
    assert parse_value("wsb_enabled", "yes") is True
    assert parse_value("wsb_enabled", "off") is False
    assert parse_value("wsb_enabled", "no") is False
    with pytest.raises(ValueError):
        parse_value("wsb_enabled", "maybe")


def test_parse_wsb_top_n_rejects_zero() -> None:
    with pytest.raises(ValueError):
        parse_value("wsb_top_n", "0")


def test_parse_sentiment_weight_validates_range() -> None:
    assert parse_value("sentiment_weight", "0.45") == 0.45
    assert parse_value("sentiment_weight", "0") == 0.0
    assert parse_value("sentiment_weight", "1") == 1.0
    with pytest.raises(ValueError):
        parse_value("sentiment_weight", "-0.1")
    with pytest.raises(ValueError):
        parse_value("sentiment_weight", "1.5")
