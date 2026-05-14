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
