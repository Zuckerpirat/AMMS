"""Backtest report writer.

Saves BacktestStats + config metadata to a JSON file so results persist
across runs and can be compared later. Provides a simple history loader.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from amms.backtest.engine import BacktestConfig, BacktestResult
from amms.backtest.stats import BacktestStats, compute_stats

logger = logging.getLogger(__name__)

_DEFAULT_REPORT_DIR = Path("reports") / "backtest"


def _report_path(report_dir: Path, label: str) -> Path:
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    safe = label.replace(" ", "_").replace("/", "-")[:40]
    return report_dir / f"{ts}_{safe}.json"


def save_backtest_report(
    result: BacktestResult,
    *,
    report_dir: Path | None = None,
    label: str = "backtest",
) -> Path:
    """Compute stats and write a JSON report.  Returns the path written."""
    stats = compute_stats(result)
    report_dir = Path(report_dir) if report_dir else _DEFAULT_REPORT_DIR
    report_dir.mkdir(parents=True, exist_ok=True)
    path = _report_path(report_dir, label)

    config = result.config
    payload = {
        "label": label,
        "generated_at": datetime.now(UTC).isoformat(),
        "config": {
            "start": str(config.start),
            "end": str(config.end),
            "symbols": list(config.symbols),
            "initial_equity": config.initial_equity,
            "strategy": config.strategy.__class__.__name__,
        },
        "stats": asdict(stats),
        "num_equity_points": len(result.equity_curve),
        "num_trades": len(result.trades),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("backtest report written to %s", path)
    return path


def load_report_history(
    report_dir: Path | None = None, *, limit: int = 10
) -> list[dict]:
    """Return the last ``limit`` backtest reports as dicts, newest first."""
    report_dir = Path(report_dir) if report_dir else _DEFAULT_REPORT_DIR
    if not report_dir.exists():
        return []
    files = sorted(report_dir.glob("*.json"), reverse=True)[:limit]
    reports = []
    for f in files:
        try:
            reports.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            logger.warning("could not parse report %s", f)
    return reports


def format_report_summary(report: dict) -> str:
    """One-line summary for a report dict (for /backhist display)."""
    s = report.get("stats", {})
    cfg = report.get("config", {})
    label = report.get("label", "?")
    start = cfg.get("start", "?")
    end = cfg.get("end", "?")
    ret = s.get("total_return_pct", 0.0)
    dd = s.get("max_drawdown_pct", 0.0)
    wr = s.get("win_rate", 0.0)
    trades = s.get("num_trades", 0)
    return (
        f"{label}  {start}→{end}  "
        f"return {ret:+.2f}%  DD {dd:.2f}%  "
        f"WR {wr:.0%}  {trades} trades"
    )
