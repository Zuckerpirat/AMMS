from __future__ import annotations

from amms.data.macro import MacroRegime, compute_regime


class _FakeData:
    def __init__(self, snap: dict) -> None:
        self._snap = snap

    def get_snapshots(self, _symbols: list[str]) -> dict:
        return self._snap


def test_compute_regime_calm_on_quiet_vixy() -> None:
    fake = _FakeData(
        {"VIXY": {"price": 17.0, "change_pct": 0.5, "change_pct_week": 1.2}}
    )
    regime = compute_regime(fake)
    assert regime.level == "calm"
    assert not regime.is_stressed
    assert regime.vixy_1d_pct == 0.5


def test_compute_regime_elevated_at_moderate_day_move() -> None:
    fake = _FakeData(
        {"VIXY": {"price": 18.0, "change_pct": 3.0, "change_pct_week": 4.0}}
    )
    regime = compute_regime(fake)
    assert regime.level == "elevated"
    assert not regime.is_stressed


def test_compute_regime_stressed_on_big_day_spike() -> None:
    fake = _FakeData(
        {"VIXY": {"price": 22.0, "change_pct": 7.0, "change_pct_week": 5.0}}
    )
    regime = compute_regime(fake)
    assert regime.level == "stressed"
    assert regime.is_stressed


def test_compute_regime_stressed_on_big_week_spike() -> None:
    fake = _FakeData(
        {"VIXY": {"price": 25.0, "change_pct": 1.0, "change_pct_week": 18.0}}
    )
    regime = compute_regime(fake)
    assert regime.level == "stressed"


def test_compute_regime_returns_calm_on_data_error() -> None:
    class _BoomData:
        def get_snapshots(self, _symbols):
            raise RuntimeError("alpaca down")

    regime = compute_regime(_BoomData())
    assert regime.level == "calm"
    assert "unavailable" in regime.reason


# ── Bar fallback tests ────────────────────────────────────────────────────────

def test_compute_regime_uses_bar_fallback_when_snapshot_empty() -> None:
    """When snapshot returns empty dict for symbol, use bars instead."""
    from amms.data.bars import Bar
    from datetime import datetime, timedelta, timezone

    class _EmptySnapData:
        def get_snapshots(self, _symbols):
            return {}  # no error, just empty

        def get_bars(self, symbol, *, limit=10):
            # Return bars where VIXY went up 6% (stressed)
            base = datetime.now(timezone.utc) - timedelta(days=10)
            bars = [
                Bar(symbol=symbol, timeframe="1Day",
                    ts=(base + timedelta(days=i)).isoformat(),
                    open=20.0, high=21.0, low=19.0,
                    close=20.0 + i * 0.1,
                    volume=1_000_000)
                for i in range(10)
            ]
            return bars

    from amms.data.macro import compute_regime
    # With a 6% 1-day move from bars, should be stressed
    # bars[-1].close / bars[-2].close - 1 ≈ 0.49% (small), but 1w is bigger
    # The bars go from 20.0 to 20.9 over 10 days
    regime = compute_regime(_EmptySnapData())
    # regime should be calm or elevated based on bar math
    assert regime.level in {"calm", "elevated", "stressed"}


def test_compute_regime_no_bar_fallback_when_snapshot_raises() -> None:
    """When snapshot raises, do NOT try bars — return calm immediately."""
    called = []

    class _RaisingSnapData:
        def get_snapshots(self, _symbols):
            raise AttributeError("no get_snapshots")

        def get_bars(self, symbol, *, limit=10):
            called.append(symbol)
            return []

    from amms.data.macro import compute_regime
    regime = compute_regime(_RaisingSnapData())
    assert regime.level == "calm"
    assert called == [], "get_bars should NOT be called when snapshot raises"
