from __future__ import annotations

import pytest

from amms.config import ConfigError, load_settings


def test_load_settings_succeeds_for_paper_endpoint(paper_env: dict[str, str]) -> None:
    settings = load_settings()
    assert settings.alpaca_api_key == "test-key"
    assert settings.alpaca_base_url == "https://paper-api.alpaca.markets"
    assert "paper-api" in settings.alpaca_base_url


def test_load_settings_refuses_live_endpoint(
    paper_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ALPACA_BASE_URL", "https://api.alpaca.markets")
    with pytest.raises(ConfigError, match="paper"):
        load_settings()


def test_load_settings_requires_api_key(
    paper_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="ALPACA_API_KEY"):
        load_settings()


def test_load_settings_requires_api_secret(
    paper_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ALPACA_API_SECRET", raising=False)
    with pytest.raises(ConfigError, match="ALPACA_API_SECRET"):
        load_settings()
