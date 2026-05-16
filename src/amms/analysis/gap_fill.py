"""Gap Fill Probability Estimator.

Analyses the history of overnight gaps in a bar series and estimates:
  - Gap fill rate: % of historical gaps that were eventually filled
  - Avg time to fill: how many bars it typically takes
  - Size-conditional fill rate: do small gaps fill more reliably than big ones?
  - Current gap: if the most recent bar opened with a gap, show fill probability

An "overnight gap" is detected when:
  open[i] > close[i-1] * (1 + threshold)  → gap up
  open[i] < close[i-1] * (1 - threshold)  → gap down

A gap is "filled" when price returns to close[i-1] within N bars.

Interpretation:
  Fill rate > 70%: reliable gap-fill strategy
  Fill rate 50-70%: moderate, size-dependent
  Fill rate < 50%: gaps tend to hold direction (momentum)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Gap:
    bar_idx: int
    kind: str           # "up" / "down"
    prev_close: float
    open_price: float
    gap_pct: float      # signed: positive = gap up
    filled: bool
    bars_to_fill: int | None   # None if not filled within window


@dataclass(frozen=True)
class GapFillReport:
    symbol: str
    gaps: list[Gap]              # all detected gaps
    n_up_gaps: int
    n_down_gaps: int
    fill_rate: float             # % of all gaps filled
    up_fill_rate: float          # % of up-gaps filled
    down_fill_rate: float        # % of down-gaps filled
    avg_bars_to_fill: float      # median bars for filled gaps
    small_fill_rate: float       # fill rate for |gap| < 1%
    large_fill_rate: float       # fill rate for |gap| >= 1%
    current_gap: Gap | None      # most recent gap if still unfilled
    current_gap_fill_prob: float # estimated probability (from historical)
    bars_analysed: int
    gap_threshold_pct: float
    verdict: str


def _detect_gaps(bars: list, threshold_pct: float) -> list[Gap]:
    gaps = []
    for i in range(1, len(bars)):
        try:
            prev_close = float(bars[i - 1].close)
            open_i = float(bars[i].open) if hasattr(bars[i], "open") else float(bars[i].close)
        except Exception:
            continue

        if prev_close <= 0:
            continue

        gap_pct = (open_i - prev_close) / prev_close * 100.0
        if abs(gap_pct) < threshold_pct:
            continue

        kind = "up" if gap_pct > 0 else "down"
        gaps.append((i, kind, prev_close, open_i, gap_pct))

    return gaps


def _check_fill(bars: list, gap_idx: int, prev_close: float, kind: str, max_bars: int) -> tuple[bool, int | None]:
    """Check if the gap at gap_idx was filled within max_bars."""
    for j in range(gap_idx, min(gap_idx + max_bars, len(bars))):
        try:
            low = float(bars[j].low)
            high = float(bars[j].high)
        except Exception:
            continue
        if kind == "up" and low <= prev_close:
            return True, j - gap_idx
        if kind == "down" and high >= prev_close:
            return True, j - gap_idx
    return False, None


def analyze(
    bars: list,
    *,
    symbol: str = "",
    gap_threshold_pct: float = 0.3,
    max_fill_bars: int = 10,
) -> GapFillReport | None:
    """Analyse gap fill probability from bars.

    bars: list[Bar] with .close and optionally .open .high .low — at least 20.
    symbol: ticker for display.
    gap_threshold_pct: minimum gap size to count (default 0.3%).
    max_fill_bars: bars within which to look for fill (default 10).
    """
    if not bars or len(bars) < 20:
        return None

    try:
        raw_gaps = _detect_gaps(bars, gap_threshold_pct)
    except Exception:
        return None

    if not raw_gaps:
        return GapFillReport(
            symbol=symbol,
            gaps=[],
            n_up_gaps=0,
            n_down_gaps=0,
            fill_rate=0.0,
            up_fill_rate=0.0,
            down_fill_rate=0.0,
            avg_bars_to_fill=0.0,
            small_fill_rate=0.0,
            large_fill_rate=0.0,
            current_gap=None,
            current_gap_fill_prob=0.0,
            bars_analysed=len(bars),
            gap_threshold_pct=gap_threshold_pct,
            verdict="No gaps detected in the analysed period.",
        )

    gaps: list[Gap] = []
    for idx, kind, prev_close, open_price, gap_pct in raw_gaps:
        filled, btf = _check_fill(bars, idx, prev_close, kind, max_fill_bars)
        gaps.append(Gap(
            bar_idx=idx,
            kind=kind,
            prev_close=round(prev_close, 4),
            open_price=round(open_price, 4),
            gap_pct=round(gap_pct, 3),
            filled=filled,
            bars_to_fill=btf,
        ))

    # Statistics
    up_gaps = [g for g in gaps if g.kind == "up"]
    down_gaps = [g for g in gaps if g.kind == "down"]
    filled_gaps = [g for g in gaps if g.filled]
    filled_up = [g for g in up_gaps if g.filled]
    filled_down = [g for g in down_gaps if g.filled]

    total = len(gaps)
    fill_rate = len(filled_gaps) / total * 100.0 if total else 0.0
    up_fill = len(filled_up) / len(up_gaps) * 100.0 if up_gaps else 0.0
    down_fill = len(filled_down) / len(down_gaps) * 100.0 if down_gaps else 0.0

    btfs = [g.bars_to_fill for g in filled_gaps if g.bars_to_fill is not None]
    avg_btf = sum(btfs) / len(btfs) if btfs else 0.0

    # Size-conditional
    small = [g for g in gaps if abs(g.gap_pct) < 1.0]
    large = [g for g in gaps if abs(g.gap_pct) >= 1.0]
    small_fill = sum(1 for g in small if g.filled) / len(small) * 100.0 if small else 0.0
    large_fill = sum(1 for g in large if g.filled) / len(large) * 100.0 if large else 0.0

    # Current gap: check if last gap is recent and unfilled
    current_gap: Gap | None = None
    current_prob = fill_rate  # default: historical rate
    if gaps:
        last = gaps[-1]
        bars_since = len(bars) - 1 - last.bar_idx
        if bars_since <= max_fill_bars and not last.filled:
            current_gap = last
            # Use size-specific rate if available
            current_prob = small_fill if abs(last.gap_pct) < 1.0 else large_fill

    # Verdict
    if fill_rate > 70:
        tendency = "strong gap-fill tendency"
    elif fill_rate > 50:
        tendency = "moderate gap-fill tendency"
    else:
        tendency = "gaps tend to hold (momentum bias)"

    size_note = (
        f"  Small gaps fill {small_fill:.0f}% vs large {large_fill:.0f}%."
        if small and large else ""
    )
    current_note = (
        f"  Current {'up' if current_gap.kind == 'up' else 'down'} gap "
        f"({current_gap.gap_pct:+.2f}%): fill prob {current_prob:.0f}%."
        if current_gap else ""
    )
    verdict = (
        f"Gap fill rate: {fill_rate:.0f}% of {total} gaps — {tendency}. "
        f"Avg fill time: {avg_btf:.1f} bars.{size_note}{current_note}"
    )

    return GapFillReport(
        symbol=symbol,
        gaps=gaps,
        n_up_gaps=len(up_gaps),
        n_down_gaps=len(down_gaps),
        fill_rate=round(fill_rate, 1),
        up_fill_rate=round(up_fill, 1),
        down_fill_rate=round(down_fill, 1),
        avg_bars_to_fill=round(avg_btf, 1),
        small_fill_rate=round(small_fill, 1),
        large_fill_rate=round(large_fill, 1),
        current_gap=current_gap,
        current_gap_fill_prob=round(current_prob, 1),
        bars_analysed=len(bars),
        gap_threshold_pct=gap_threshold_pct,
        verdict=verdict,
    )
