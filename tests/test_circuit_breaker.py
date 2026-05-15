from __future__ import annotations

import sqlite3

import pytest

from amms.risk.circuit_breaker import (
    CircuitBreakerConfig,
    CircuitBreakerState,
    ensure_table,
    is_open,
    load_state,
    record_trade_result,
    reset_circuit,
)


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    return conn


def test_load_state_fresh_db() -> None:
    conn = _conn()
    state = load_state(conn)
    assert state.daily_loss == 0.0
    assert state.consec_losses == 0
    assert not state.tripped
    assert not state.is_open


def test_record_win_resets_consec() -> None:
    conn = _conn()
    cfg = CircuitBreakerConfig()
    # First: a loss to build up consec
    record_trade_result(conn, pnl=-100.0, config=cfg, initial_equity=100_000.0)
    state = record_trade_result(conn, pnl=200.0, config=cfg, initial_equity=100_000.0)
    assert state.consec_losses == 0  # win resets streak


def test_record_loss_increments_consec() -> None:
    conn = _conn()
    cfg = CircuitBreakerConfig()
    record_trade_result(conn, pnl=-50.0, config=cfg, initial_equity=100_000.0)
    state = record_trade_result(conn, pnl=-50.0, config=cfg, initial_equity=100_000.0)
    assert state.consec_losses == 2
    assert state.daily_loss == pytest.approx(100.0)


def test_daily_loss_threshold_trips_circuit() -> None:
    conn = _conn()
    cfg = CircuitBreakerConfig(max_daily_loss_pct=3.0)
    # 3.1% loss on $100k = $3100
    state = record_trade_result(conn, pnl=-3100.0, config=cfg, initial_equity=100_000.0)
    assert state.tripped
    assert is_open(conn)


def test_consecutive_loss_threshold_trips_circuit() -> None:
    conn = _conn()
    cfg = CircuitBreakerConfig(max_consecutive_losses=3)
    for _ in range(3):
        state = record_trade_result(conn, pnl=-1.0, config=cfg, initial_equity=100_000.0)
    assert state.tripped


def test_disabled_circuit_never_trips() -> None:
    conn = _conn()
    cfg = CircuitBreakerConfig(max_daily_loss_pct=0.001, enabled=False)
    state = record_trade_result(conn, pnl=-999_999.0, config=cfg, initial_equity=100_000.0)
    assert not state.tripped


def test_reset_clears_state() -> None:
    conn = _conn()
    cfg = CircuitBreakerConfig(max_daily_loss_pct=1.0)
    record_trade_result(conn, pnl=-5000.0, config=cfg, initial_equity=100_000.0)
    assert is_open(conn)
    reset_circuit(conn)
    state = load_state(conn)
    assert not state.tripped
    assert state.daily_loss == 0.0
    assert state.consec_losses == 0


def test_tripped_circuit_stays_tripped() -> None:
    conn = _conn()
    cfg = CircuitBreakerConfig(max_daily_loss_pct=1.0)
    record_trade_result(conn, pnl=-5000.0, config=cfg, initial_equity=100_000.0)
    assert is_open(conn)
    # Winning trade should not untrip
    state = record_trade_result(conn, pnl=10000.0, config=cfg, initial_equity=100_000.0)
    assert state.tripped


def test_is_open_before_any_trades() -> None:
    conn = _conn()
    assert not is_open(conn)


def test_state_persists_across_calls() -> None:
    conn = _conn()
    cfg = CircuitBreakerConfig()
    record_trade_result(conn, pnl=-100.0, config=cfg, initial_equity=100_000.0)
    state = load_state(conn)
    assert state.daily_loss == pytest.approx(100.0)
    assert state.consec_losses == 1
