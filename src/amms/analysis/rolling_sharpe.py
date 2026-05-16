"""Rolling Sharpe Ratio Analyser.

Computes Sharpe ratio over a rolling window using bar close-to-close returns,
providing stability metrics and trend in risk-adjusted performance.

Sharpe = (mean_return - risk_free) / std_return × sqrt(annualize)

Also computes:
  - Sortino ratio (downside deviation only)
  - Calmar ratio (mean return / max drawdown)
  - Rolling Sharpe trend (improving / worsening / stable)
  - Percentile of current Sharpe vs history
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SharpeSnapshot:
    bar_idx: int
    sharpe: float
    sortino: float
    mean_return: float
    vol: float


@dataclass(frozen=True)
class RollingSharpeReport:
    symbol: str
    window: int
    annualize: int

    current_sharpe: float
    current_sortino: float
    current_vol: float          # annualised vol %
    current_mean_return: float  # annualised mean return %

    sharpe_trend: str           # "improving", "worsening", "stable"
    sharpe_percentile: float    # 0-100: current vs rolling history

    max_sharpe: float
    min_sharpe: float
    avg_sharpe: float

    calmar: float | None        # mean return / max drawdown (proxy)
    max_drawdown_pct: float     # of the return series itself (not equity)

    history: list[SharpeSnapshot]
    current_price: float
    bars_used: int
    verdict: str


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _std(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = _mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def _sharpe(returns: list[float], rf: float, annualize: int) -> float:
    if len(returns) < 2:
        return 0.0
    m = _mean(returns) - rf
    s = _std(returns)
    if s < 1e-12:
        return 0.0
    return m / s * math.sqrt(annualize)


def _sortino(returns: list[float], rf: float, annualize: int) -> float:
    if len(returns) < 2:
        return 0.0
    m = _mean(returns) - rf
    downside = [min(r, 0.0) for r in returns]
    ds_std = math.sqrt(sum(d ** 2 for d in downside) / max(1, len(downside) - 1))
    if ds_std < 1e-12:
        return 0.0
    return m / ds_std * math.sqrt(annualize)


def analyze(
    bars: list,
    *,
    symbol: str = "",
    window: int = 30,
    annualize: int = 252,
    rf_daily: float = 0.0,
    history_bars: int = 50,
) -> RollingSharpeReport | None:
    """Compute rolling Sharpe ratio from bar close returns.

    bars: bar objects with .close attribute.
    window: rolling window for Sharpe calculation.
    annualize: trading days per year for annualisation.
    rf_daily: risk-free rate per bar (default 0).
    history_bars: number of rolling Sharpe snapshots to return.
    """
    if not bars or len(bars) < window + 5:
        return None

    try:
        closes = [float(b.close) for b in bars]
    except (AttributeError, TypeError, ValueError):
        return None

    current = closes[-1]
    if current <= 0:
        return None

    n = len(closes)
    returns = [closes[i] / closes[i - 1] - 1.0 for i in range(1, n) if closes[i - 1] > 0]

    if len(returns) < window:
        return None

    # Build rolling Sharpe series
    sharpe_series: list[float] = []
    sortino_series: list[float] = []
    mean_series: list[float] = []
    vol_series: list[float] = []
    snapshots: list[SharpeSnapshot] = []

    ret_start = max(0, len(returns) - history_bars - window)

    for k in range(window, len(returns) + 1):
        window_rets = returns[k - window:k]
        sh = _sharpe(window_rets, rf_daily, annualize)
        so = _sortino(window_rets, rf_daily, annualize)
        m = _mean(window_rets) * annualize * 100.0  # annualised %
        v = _std(window_rets) * math.sqrt(annualize) * 100.0  # annualised vol %
        sharpe_series.append(sh)
        sortino_series.append(so)
        mean_series.append(m)
        vol_series.append(v)

        # Build snapshot for history window
        bar_idx = k  # returns[k-1] corresponds to closes[k]
        if k >= len(returns) - history_bars:
            snapshots.append(SharpeSnapshot(
                bar_idx=bar_idx,
                sharpe=round(sh, 3),
                sortino=round(so, 3),
                mean_return=round(m, 3),
                vol=round(v, 3),
            ))

    if not sharpe_series:
        return None

    cur_sharpe = sharpe_series[-1]
    cur_sortino = sortino_series[-1]
    cur_vol = vol_series[-1]
    cur_mean = mean_series[-1]

    # Trend: compare recent vs prior quarter
    qlen = max(1, len(sharpe_series) // 4)
    recent_avg = _mean(sharpe_series[-qlen:])
    earlier_avg = _mean(sharpe_series[-2 * qlen:-qlen]) if len(sharpe_series) >= 2 * qlen else recent_avg

    if recent_avg > earlier_avg + 0.1:
        trend = "improving"
    elif recent_avg < earlier_avg - 0.1:
        trend = "worsening"
    else:
        trend = "stable"

    # Percentile of current Sharpe
    below = sum(1 for s in sharpe_series if s <= cur_sharpe)
    sharpe_pct = below / len(sharpe_series) * 100.0

    max_sh = max(sharpe_series)
    min_sh = min(sharpe_series)
    avg_sh = _mean(sharpe_series)

    # Max drawdown of return cumulative
    cum = [1.0]
    for r in returns[-min(252, len(returns)):]:
        cum.append(cum[-1] * (1 + r))
    peak = cum[0]
    max_dd = 0.0
    for v in cum:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        if dd > max_dd:
            max_dd = dd
    max_dd_pct = max_dd * 100.0

    # Calmar proxy
    calmar = (cur_mean / max_dd_pct) if max_dd_pct > 0 else None

    # Verdict
    if cur_sharpe >= 1.5:
        quality = "excellent risk-adjusted returns"
    elif cur_sharpe >= 0.5:
        quality = "acceptable risk-adjusted returns"
    elif cur_sharpe >= 0.0:
        quality = "marginal risk-adjusted returns"
    else:
        quality = "negative risk-adjusted returns"

    verdict = (
        f"Rolling Sharpe({window}): {cur_sharpe:.2f} — {quality}. "
        f"Trend: {trend}. Sortino: {cur_sortino:.2f}. Vol: {cur_vol:.1f}%."
    )

    return RollingSharpeReport(
        symbol=symbol,
        window=window,
        annualize=annualize,
        current_sharpe=round(cur_sharpe, 3),
        current_sortino=round(cur_sortino, 3),
        current_vol=round(cur_vol, 2),
        current_mean_return=round(cur_mean, 2),
        sharpe_trend=trend,
        sharpe_percentile=round(sharpe_pct, 1),
        max_sharpe=round(max_sh, 3),
        min_sharpe=round(min_sh, 3),
        avg_sharpe=round(avg_sh, 3),
        calmar=round(calmar, 3) if calmar is not None else None,
        max_drawdown_pct=round(max_dd_pct, 2),
        history=snapshots,
        current_price=round(current, 4),
        bars_used=n,
        verdict=verdict,
    )
