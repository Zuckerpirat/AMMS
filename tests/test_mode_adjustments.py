"""Tests for trading-mode risk adjustments."""

from __future__ import annotations

from amms.scheduler import _apply_mode_adjustments


def _make_config():
    from amms.config import AppConfig, SchedulerConfig, StrategyConfig
    from amms.data.wsb_discovery import WSBDiscoveryConfig
    from amms.filters.universe import UniverseFilter
    from amms.risk.rules import RiskConfig

    return AppConfig(
        watchlist=("AAPL",),
        risk=RiskConfig(max_position_pct=0.02, stop_loss_pct=0.05),
        universe=UniverseFilter(),
        strategy=StrategyConfig(name="composite"),
        scheduler=SchedulerConfig(),
        wsb_discovery=WSBDiscoveryConfig(),
    )


def test_swing_mode_no_adjustment():
    config = _make_config()
    adjusted = _apply_mode_adjustments(config, "swing")
    assert adjusted.risk.max_position_pct == config.risk.max_position_pct
    assert adjusted.risk.stop_loss_pct == config.risk.stop_loss_pct


def test_conservative_mode_reduces_position_size():
    config = _make_config()
    adjusted = _apply_mode_adjustments(config, "conservative")
    assert adjusted.risk.max_position_pct == 0.01  # halved from 0.02
    assert adjusted.risk.stop_loss_pct == 0.03  # tighter


def test_meme_mode_reduces_position_size_slightly():
    config = _make_config()
    adjusted = _apply_mode_adjustments(config, "meme")
    assert adjusted.risk.max_position_pct == 0.015


def test_event_mode_adjusts_stop_loss():
    config = _make_config()
    adjusted = _apply_mode_adjustments(config, "event")
    assert adjusted.risk.stop_loss_pct == 0.04


def test_unknown_mode_no_adjustment():
    config = _make_config()
    adjusted = _apply_mode_adjustments(config, "turbo_yolo")
    assert adjusted.risk.max_position_pct == config.risk.max_position_pct
