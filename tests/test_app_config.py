from __future__ import annotations

from pathlib import Path

import pytest

from amms.config import ConfigError, load_app_config

VALID_YAML = """
watchlist:
  - aapl
  - MSFT
  - nvda
strategy:
  name: sma_cross
  params:
    fast: 10
    slow: 30
risk:
  max_open_positions: 3
  max_position_pct: 0.05
  daily_loss_pct: -0.02
scheduler:
  tick_seconds: 30
  timezone: America/New_York
"""


def test_load_app_config_parses_full_yaml(tmp_path: Path) -> None:
    p = tmp_path / "config.yaml"
    p.write_text(VALID_YAML, encoding="utf-8")
    cfg = load_app_config(p)
    assert cfg.watchlist == ("AAPL", "MSFT", "NVDA")
    assert cfg.strategy.name == "sma_cross"
    assert cfg.strategy.params == {"fast": 10, "slow": 30}
    assert cfg.strategy.timeframe == "1Day"  # default when omitted
    assert cfg.risk.max_open_positions == 3
    assert cfg.risk.daily_loss_pct == pytest.approx(-0.02)
    assert cfg.scheduler.tick_seconds == 30


def test_load_app_config_reads_universe_filter(tmp_path: Path) -> None:
    yaml_text = """
watchlist: [AAPL]
strategy: {name: sma_cross}
universe:
  min_price: 1.0
  max_price: 500.0
  min_avg_dollar_volume: 1000000
  adv_lookback: 30
"""
    p = tmp_path / "config.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    cfg = load_app_config(p)
    assert cfg.universe.min_price == pytest.approx(1.0)
    assert cfg.universe.max_price == pytest.approx(500.0)
    assert cfg.universe.min_avg_dollar_volume == pytest.approx(1_000_000)
    assert cfg.universe.adv_lookback == 30


def test_load_app_config_universe_defaults_when_section_missing(tmp_path: Path) -> None:
    p = tmp_path / "config.yaml"
    p.write_text("watchlist: [AAPL]\nstrategy: {name: sma_cross}\n", encoding="utf-8")
    cfg = load_app_config(p)
    assert cfg.universe.min_price == 0.0
    assert cfg.universe.max_price is None
    assert cfg.universe.min_avg_dollar_volume == 0.0


def test_load_app_config_reads_strategy_timeframe(tmp_path: Path) -> None:
    yaml_text = """
watchlist: [AAPL]
strategy:
  name: sma_cross
  timeframe: 5Min
  params: {fast: 3, slow: 8}
"""
    p = tmp_path / "config.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    cfg = load_app_config(p)
    assert cfg.strategy.timeframe == "5Min"


def test_load_app_config_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_app_config(tmp_path / "nope.yaml")


def test_load_app_config_rejects_empty_watchlist(tmp_path: Path) -> None:
    p = tmp_path / "config.yaml"
    p.write_text("watchlist: []\nstrategy: {name: sma_cross}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="watchlist"):
        load_app_config(p)


def test_load_app_config_rejects_duplicate_symbols(tmp_path: Path) -> None:
    p = tmp_path / "config.yaml"
    p.write_text(
        "watchlist: [AAPL, aapl]\nstrategy: {name: sma_cross}\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="duplicate"):
        load_app_config(p)


def test_load_app_config_requires_strategy_name(tmp_path: Path) -> None:
    p = tmp_path / "config.yaml"
    p.write_text("watchlist: [AAPL]\nstrategy: {}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="strategy.name"):
        load_app_config(p)


def test_load_app_config_validates_risk_via_dataclass(tmp_path: Path) -> None:
    bad_yaml = """
watchlist: [AAPL]
strategy: {name: sma_cross}
risk:
  daily_loss_pct: 0.05
"""
    p = tmp_path / "config.yaml"
    p.write_text(bad_yaml, encoding="utf-8")
    with pytest.raises(ValueError, match="daily_loss_pct"):
        load_app_config(p)
