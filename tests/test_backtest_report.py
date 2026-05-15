from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from amms.backtest.engine import BacktestConfig, BacktestResult, Portfolio, Trade
from amms.backtest.report import (
    format_report_summary,
    load_report_history,
    save_backtest_report,
)
from amms.risk import RiskConfig
from amms.strategy import SmaCross


def _make_result(
    trades=None, equity_curve=None, initial=100_000.0
) -> BacktestResult:
    config = BacktestConfig(
        start=date(2025, 1, 1),
        end=date(2025, 12, 31),
        symbols=("AAPL",),
        initial_equity=initial,
        risk=RiskConfig(),
        strategy=SmaCross(),
    )
    return BacktestResult(
        config=config,
        portfolio=Portfolio(cash=initial),
        trades=trades or [],
        equity_curve=equity_curve or [("2025-01-01", initial), ("2025-12-31", initial)],
    )


def test_save_backtest_report_creates_file(tmp_path: Path) -> None:
    result = _make_result()
    path = save_backtest_report(result, report_dir=tmp_path, label="test_run")
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["label"] == "test_run"
    assert "stats" in data
    assert "config" in data


def test_save_backtest_report_stats_correct(tmp_path: Path) -> None:
    result = _make_result(
        equity_curve=[("2025-01-01", 100_000.0), ("2025-12-31", 110_000.0)]
    )
    path = save_backtest_report(result, report_dir=tmp_path)
    data = json.loads(path.read_text())
    assert data["stats"]["total_return_pct"] == pytest.approx(10.0)


def test_load_report_history_empty(tmp_path: Path) -> None:
    reports = load_report_history(report_dir=tmp_path)
    assert reports == []


def test_load_report_history_nonexistent_dir(tmp_path: Path) -> None:
    reports = load_report_history(report_dir=tmp_path / "does_not_exist")
    assert reports == []


def test_load_report_history_returns_newest_first(tmp_path: Path) -> None:
    for i in range(3):
        result = _make_result(
            equity_curve=[("2025-01-01", 100_000.0), ("2025-12-31", 100_000.0 + i * 5_000)]
        )
        save_backtest_report(result, report_dir=tmp_path, label=f"run_{i}")
    reports = load_report_history(report_dir=tmp_path)
    assert len(reports) == 3
    # Newest first = highest return (run_2)
    assert reports[0]["label"] == "run_2"


def test_load_report_history_respects_limit(tmp_path: Path) -> None:
    for i in range(5):
        save_backtest_report(_make_result(), report_dir=tmp_path, label=f"r{i}")
    reports = load_report_history(report_dir=tmp_path, limit=3)
    assert len(reports) == 3


def test_format_report_summary_basic(tmp_path: Path) -> None:
    result = _make_result(
        equity_curve=[("2025-01-01", 100_000.0), ("2025-12-31", 115_000.0)]
    )
    path = save_backtest_report(result, report_dir=tmp_path, label="my_run")
    data = json.loads(path.read_text())
    summary = format_report_summary(data)
    assert "my_run" in summary
    assert "15.00%" in summary or "+15.00" in summary
    assert "WR" in summary


def test_save_creates_reports_dir_automatically(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "c"
    result = _make_result()
    path = save_backtest_report(result, report_dir=nested)
    assert path.exists()


def test_report_includes_strategy_name(tmp_path: Path) -> None:
    result = _make_result()
    path = save_backtest_report(result, report_dir=tmp_path)
    data = json.loads(path.read_text())
    assert data["config"]["strategy"] == "SmaCross"
