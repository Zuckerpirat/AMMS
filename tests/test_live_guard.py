"""Tests for amms.execution.live_guard."""

from __future__ import annotations

import pytest

from amms.execution.live_guard import (
    LIVE_ACK_PHRASE,
    LiveTradingNotAllowed,
    assert_live_allowed,
    check_live_allowed,
    is_live_mode_url,
)


@pytest.fixture(autouse=True)
def clear_live_env(monkeypatch):
    """Each test starts with no live-trading env vars set."""
    for var in ("AMMS_LIVE_ACKNOWLEDGED", "AMMS_LIVE_MAX_ORDER_USD", "AMMS_LIVE_MAX_DAILY_USD"):
        monkeypatch.delenv(var, raising=False)


class TestDisabled:
    def test_default_disabled(self):
        lim = check_live_allowed()
        assert lim.enabled is False

    def test_disabled_reason_mentions_ack(self):
        lim = check_live_allowed()
        assert "AMMS_LIVE_ACKNOWLEDGED" in lim.reason_if_disabled

    def test_wrong_ack_phrase(self, monkeypatch):
        monkeypatch.setenv("AMMS_LIVE_ACKNOWLEDGED", "yes please")
        assert check_live_allowed().enabled is False

    def test_correct_ack_but_no_limits(self, monkeypatch):
        monkeypatch.setenv("AMMS_LIVE_ACKNOWLEDGED", LIVE_ACK_PHRASE)
        lim = check_live_allowed()
        assert lim.enabled is False
        assert "MAX_ORDER" in lim.reason_if_disabled

    def test_order_set_but_no_daily(self, monkeypatch):
        monkeypatch.setenv("AMMS_LIVE_ACKNOWLEDGED", LIVE_ACK_PHRASE)
        monkeypatch.setenv("AMMS_LIVE_MAX_ORDER_USD", "1000")
        lim = check_live_allowed()
        assert lim.enabled is False
        assert "DAILY" in lim.reason_if_disabled

    def test_invalid_order_value(self, monkeypatch):
        monkeypatch.setenv("AMMS_LIVE_ACKNOWLEDGED", LIVE_ACK_PHRASE)
        monkeypatch.setenv("AMMS_LIVE_MAX_ORDER_USD", "not a number")
        monkeypatch.setenv("AMMS_LIVE_MAX_DAILY_USD", "5000")
        assert check_live_allowed().enabled is False

    def test_negative_order_value(self, monkeypatch):
        monkeypatch.setenv("AMMS_LIVE_ACKNOWLEDGED", LIVE_ACK_PHRASE)
        monkeypatch.setenv("AMMS_LIVE_MAX_ORDER_USD", "-100")
        monkeypatch.setenv("AMMS_LIVE_MAX_DAILY_USD", "5000")
        assert check_live_allowed().enabled is False


class TestEnabled:
    def test_all_conditions_met(self, monkeypatch):
        monkeypatch.setenv("AMMS_LIVE_ACKNOWLEDGED", LIVE_ACK_PHRASE)
        monkeypatch.setenv("AMMS_LIVE_MAX_ORDER_USD", "500")
        monkeypatch.setenv("AMMS_LIVE_MAX_DAILY_USD", "2000")
        lim = check_live_allowed()
        assert lim.enabled is True
        assert lim.max_order_usd == 500.0
        assert lim.max_daily_usd == 2000.0


class TestAssertion:
    def test_assert_raises_when_disabled(self):
        with pytest.raises(LiveTradingNotAllowed):
            assert_live_allowed()

    def test_assert_passes_when_enabled(self, monkeypatch):
        monkeypatch.setenv("AMMS_LIVE_ACKNOWLEDGED", LIVE_ACK_PHRASE)
        monkeypatch.setenv("AMMS_LIVE_MAX_ORDER_USD", "100")
        monkeypatch.setenv("AMMS_LIVE_MAX_DAILY_USD", "500")
        lim = assert_live_allowed()
        assert lim.enabled is True


class TestUrlDetection:
    def test_paper_url_not_live(self):
        assert is_live_mode_url("https://paper-api.alpaca.markets") is False

    def test_live_url_detected(self):
        assert is_live_mode_url("https://api.alpaca.markets") is True

    def test_random_url_not_live(self):
        assert is_live_mode_url("https://example.com") is False

    def test_case_insensitive(self):
        assert is_live_mode_url("HTTPS://API.ALPACA.MARKETS") is True
