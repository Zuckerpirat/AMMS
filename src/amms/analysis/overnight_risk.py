"""Overnight gap risk analysis.

Quantifies how often and how severely a symbol gaps at the open
relative to the prior close.  Used to assess risk of holding positions
overnight or over the weekend.

Metrics computed from recent bars:
  - gap_frequency_pct: % of days with a gap > threshold
  - avg_gap_pct:       average absolute gap size (%)
  - max_gap_pct:       largest single gap (absolute)
  - gap_down_pct:      average of downside gaps only
  - risk_score:        0-100 (higher = more overnight risk)
  - risk_label:        "low"|"moderate"|"elevated"|"high"

Gap definition: |open - prev_close| / prev_close * 100
Threshold: gaps < 0.1% are considered noise and excluded.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OvernightRisk:
    symbol: str
    gap_frequency_pct: float   # % of bars with meaningful gap
    avg_gap_pct: float         # avg absolute gap size
    avg_gap_down_pct: float    # avg downside gap (0 if none)
    max_gap_pct: float         # largest absolute gap
    risk_score: float          # 0-100
    risk_label: str            # "low"|"moderate"|"elevated"|"high"
    bars_analyzed: int
    n_gaps: int


def analyze(bars: list, *, min_gap_pct: float = 0.1) -> OvernightRisk | None:
    """Compute overnight gap risk from bar data.

    bars: list[Bar] — needs at least 5 bars
    min_gap_pct: minimum gap size to count (default 0.1%)
    Returns None if insufficient data.
    """
    if len(bars) < 5:
        return None

    symbol = bars[0].symbol
    n = len(bars)

    gaps_abs: list[float] = []
    gaps_down: list[float] = []

    for i in range(1, n):
        prev_close = bars[i - 1].close
        open_price = bars[i].open
        if prev_close <= 0:
            continue
        gap_pct = (open_price - prev_close) / prev_close * 100
        abs_gap = abs(gap_pct)
        if abs_gap >= min_gap_pct:
            gaps_abs.append(abs_gap)
            if gap_pct < 0:
                gaps_down.append(abs_gap)

    n_gaps = len(gaps_abs)
    gap_freq = n_gaps / (n - 1) * 100 if n > 1 else 0.0
    avg_gap = sum(gaps_abs) / n_gaps if n_gaps else 0.0
    avg_gap_down = sum(gaps_down) / len(gaps_down) if gaps_down else 0.0
    max_gap = max(gaps_abs) if gaps_abs else 0.0

    # Risk score: weighted combination
    # frequency (0-40): 100% freq = 40pts, 50% = 20pts
    # avg gap (0-40): 3%+ = 40pts, 1% = 13pts
    # max gap (0-20): 10%+ = 20pts
    freq_score = min(40.0, gap_freq * 0.4)
    avg_score = min(40.0, avg_gap * 13.0)
    max_score = min(20.0, max_gap * 2.0)
    risk_score = freq_score + avg_score + max_score

    if risk_score < 20:
        risk_label = "low"
    elif risk_score < 40:
        risk_label = "moderate"
    elif risk_score < 65:
        risk_label = "elevated"
    else:
        risk_label = "high"

    return OvernightRisk(
        symbol=symbol,
        gap_frequency_pct=round(gap_freq, 1),
        avg_gap_pct=round(avg_gap, 2),
        avg_gap_down_pct=round(avg_gap_down, 2),
        max_gap_pct=round(max_gap, 2),
        risk_score=round(risk_score, 1),
        risk_label=risk_label,
        bars_analyzed=n,
        n_gaps=n_gaps,
    )
