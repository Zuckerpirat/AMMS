"""Tests for MemePortfolio sandbox."""

from __future__ import annotations

import json
import pytest
from pathlib import Path


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_portfolio(tmp_path: Path, *, starting_cash: float = 5_000.0,
                    main_trader=None, max_allocation_pct: float = 1.0):
    """Helper that defaults max_allocation_pct=1.0 so tests aren't blocked by the cap."""
    from amms.execution.meme_portfolio import MemePortfolio, MemeConfig
    cfg = MemeConfig(starting_cash=starting_cash, max_allocation_pct=max_allocation_pct)
    return MemePortfolio(config=cfg, state_path=tmp_path / "meme.json", main_trader=main_trader)


def _make_main_trader(tmp_path: Path, *, starting_cash: float = 50_000.0):
    from amms.execution.paper_trader import PaperTrader
    return PaperTrader(starting_cash=starting_cash)


# ── Instantiation ─────────────────────────────────────────────────────────────

def test_load_returns_meme_portfolio(tmp_path):
    from amms.execution.meme_portfolio import MemePortfolio
    mp = MemePortfolio.load(state_path=tmp_path / "meme.json")
    assert mp is not None
    assert mp.name == "meme-sandbox"


def test_default_config_values(tmp_path):
    from amms.execution.meme_portfolio import MemeConfig
    cfg = MemeConfig()
    assert cfg.max_position_pct == pytest.approx(0.03)
    assert cfg.max_positions == 5
    assert cfg.max_allocation_pct == pytest.approx(0.10)
    assert cfg.cooldown_minutes == 120
    assert cfg.starting_cash == pytest.approx(5_000.0)


def test_uses_separate_state_file(tmp_path):
    main = _make_main_trader(tmp_path, starting_cash=100_000.0)
    from amms.execution.meme_portfolio import MemePortfolio, MemeConfig
    cfg = MemeConfig(starting_cash=5_000.0)
    mp = MemePortfolio(config=cfg, state_path=tmp_path / "meme.json", main_trader=main)
    mp.buy("GME", qty=1, price=10.0, reason="test")
    assert (tmp_path / "meme.json").exists()


# ── Buy enforcement ───────────────────────────────────────────────────────────

def test_buy_succeeds_within_limits(tmp_path):
    mp = _make_portfolio(tmp_path, starting_cash=5_000.0)
    trade = mp.buy("GME", qty=1, price=20.0, reason="wsb")
    assert trade is not None


def test_buy_prefixes_reason_with_meme(tmp_path):
    mp = _make_portfolio(tmp_path, starting_cash=5_000.0)
    mp.buy("GME", qty=1, price=20.0, reason="wsb spike")
    trades = mp.recent_trades(1)
    assert len(trades) == 1
    assert trades[0].reason.startswith("MEME:")


def test_buy_with_empty_reason(tmp_path):
    mp = _make_portfolio(tmp_path, starting_cash=5_000.0)
    mp.buy("GME", qty=1, price=20.0, reason="")
    trades = mp.recent_trades(1)
    assert trades[0].reason.startswith("MEME")


def test_buy_caps_position_size(tmp_path):
    mp = _make_portfolio(tmp_path, starting_cash=5_000.0)
    # Request 1000 shares at $10 = $10_000, but max is 3% of $5_000 = $150
    trade = mp.buy("GME", qty=1000, price=10.0, reason="oversized")
    assert trade is not None
    # Actual qty should be much less than 1000
    assert trade.qty < 20


def test_buy_blocks_at_max_positions(tmp_path):
    main = _make_main_trader(tmp_path, starting_cash=100_000.0)
    from amms.execution.meme_portfolio import MemePortfolio, MemeConfig
    # Max 2 positions, each buy $1 = well within 3% of $5000 ($150 limit)
    cfg = MemeConfig(max_positions=2, starting_cash=5_000.0, max_allocation_pct=1.0)
    mp = MemePortfolio(config=cfg, state_path=tmp_path / "meme.json", main_trader=main)
    mp.buy("AAA", qty=1, price=1.0, reason="a")
    mp.buy("BBB", qty=1, price=1.0, reason="b")
    result = mp.buy("CCC", qty=1, price=1.0, reason="c")
    assert result is None


def test_buy_returns_none_for_zero_qty(tmp_path):
    mp = _make_portfolio(tmp_path, starting_cash=5_000.0)
    result = mp.buy("GME", qty=0, price=10.0, reason="zero")
    assert result is None


def test_buy_returns_none_for_zero_price(tmp_path):
    mp = _make_portfolio(tmp_path, starting_cash=5_000.0)
    result = mp.buy("GME", qty=10, price=0.0, reason="zero price")
    assert result is None


# ── Allocation cap ────────────────────────────────────────────────────────────

def test_allocation_cap_blocks_when_meme_exceeds_limit(tmp_path):
    """When meme portfolio already exceeds 10% of combined, new buys blocked."""
    main = _make_main_trader(tmp_path, starting_cash=1_000.0)  # tiny main
    from amms.execution.meme_portfolio import MemePortfolio, MemeConfig
    # Meme starts at $5_000 — main $1_000 => meme is 83% of $6_000 > 10%
    cfg = MemeConfig(starting_cash=5_000.0, max_allocation_pct=0.10)
    mp = MemePortfolio(config=cfg, state_path=tmp_path / "meme.json", main_trader=main)
    result = mp.buy("GME", qty=1, price=10.0, reason="too large")
    assert result is None


def test_allocation_cap_allows_when_within_limit(tmp_path):
    main = _make_main_trader(tmp_path, starting_cash=100_000.0)  # large main
    from amms.execution.meme_portfolio import MemePortfolio, MemeConfig
    # Meme $5_000, main $100_000 → meme = ~4.7% < 10%
    cfg = MemeConfig(starting_cash=5_000.0, max_allocation_pct=0.10)
    mp = MemePortfolio(config=cfg, state_path=tmp_path / "meme.json", main_trader=main)
    result = mp.buy("GME", qty=1, price=1.0, reason="fine")
    assert result is not None


def test_combined_value_without_main_trader(tmp_path):
    mp = _make_portfolio(tmp_path)
    # No main trader — combined value is just meme portfolio value
    combined = mp._combined_value()
    snap = mp.snapshot()
    assert combined == pytest.approx(snap.portfolio_value, rel=1e-6)


def test_combined_value_with_main_trader(tmp_path):
    main = _make_main_trader(tmp_path, starting_cash=10_000.0)
    from amms.execution.meme_portfolio import MemePortfolio, MemeConfig
    cfg = MemeConfig(starting_cash=5_000.0)
    mp = MemePortfolio(config=cfg, state_path=tmp_path / "meme.json", main_trader=main)
    combined = mp._combined_value()
    assert combined == pytest.approx(15_000.0, rel=1e-4)


# ── Sell ──────────────────────────────────────────────────────────────────────

def test_sell_prefixes_reason_with_meme(tmp_path):
    mp = _make_portfolio(tmp_path, starting_cash=5_000.0)
    mp.buy("GME", qty=5, price=10.0, reason="in")
    mp.sell("GME", qty=5, price=12.0, reason="profit")
    trades = mp.recent_trades(5)
    sell_trade = [t for t in trades if t.side == "sell"]
    assert len(sell_trade) == 1
    assert sell_trade[0].reason.startswith("MEME:")


def test_close_position_prefixes_reason(tmp_path):
    mp = _make_portfolio(tmp_path, starting_cash=5_000.0)
    mp.buy("AMC", qty=3, price=5.0, reason="in")
    trade = mp.close_position("AMC", price=6.0, reason="close")
    assert trade is not None
    assert trade.reason.startswith("MEME:")


# ── Snapshot and persistence ──────────────────────────────────────────────────

def test_snapshot_returns_portfolio_snapshot(tmp_path):
    mp = _make_portfolio(tmp_path)
    snap = mp.snapshot()
    assert hasattr(snap, "cash")
    assert hasattr(snap, "portfolio_value")
    assert hasattr(snap, "positions")


def test_position_returns_none_when_not_held(tmp_path):
    mp = _make_portfolio(tmp_path)
    assert mp.position("GME") is None


def test_save_persists_state(tmp_path):
    state_path = tmp_path / "meme.json"
    mp = _make_portfolio(tmp_path, starting_cash=5_000.0)
    mp.buy("GME", qty=1, price=5.0, reason="test")
    mp.save()
    assert state_path.exists()
    data = json.loads(state_path.read_text())
    assert "cash" in data


# ── Status summary ────────────────────────────────────────────────────────────

def test_status_summary_contains_required_sections(tmp_path):
    mp = _make_portfolio(tmp_path)
    s = mp.status_summary()
    assert "Meme Sandbox Portfolio" in s
    assert "Cash" in s
    assert "Positions" in s
    assert "Value" in s
    assert "Max alloc" in s
    assert "Return" in s


def test_status_summary_shows_open_positions(tmp_path):
    mp = _make_portfolio(tmp_path, starting_cash=5_000.0)
    mp.buy("GME", qty=1, price=10.0, reason="test")
    s = mp.status_summary()
    assert "GME" in s


def test_status_summary_without_main_trader_no_crash(tmp_path):
    mp = _make_portfolio(tmp_path)
    s = mp.status_summary()
    assert isinstance(s, str)
    assert len(s) > 0


# ── Symbol normalization ──────────────────────────────────────────────────────

def test_buy_normalizes_symbol_to_upper(tmp_path):
    mp = _make_portfolio(tmp_path, starting_cash=5_000.0)
    mp.buy("gme", qty=1, price=10.0, reason="lower")
    assert mp.position("GME") is not None
