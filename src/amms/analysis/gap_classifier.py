"""Gap Classifier.

Identifies price gaps between consecutive bars and classifies them by type:
  - Common gap:    Small gap within a sideways range; high fill probability
  - Breakaway gap: Gap at start of a new trend (from consolidation); lower fill prob
  - Runaway gap:   Gap mid-trend confirming continuation; moderate fill prob
  - Exhaustion gap: Gap near trend end, high volume; very high fill probability
  - Island reversal: Gaps both sides of a price island (gap up then gap down)

Also tracks which past gaps have been filled and estimates fill probability.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Gap:
    bar_idx: int           # bar where gap opened
    direction: str         # "up" or "down"
    gap_open: float        # lower bound of the gap
    gap_close_: float      # upper bound of the gap (note: 'close_' avoids confusion)
    gap_size: float        # gap_close_ - gap_open
    gap_pct: float         # gap_size / prior_close * 100
    gap_type: str          # "common", "breakaway", "runaway", "exhaustion", "island", "unknown"
    filled: bool           # has the gap been subsequently filled?
    fill_bar: int | None   # bar index when filled (if filled)
    fill_probability: float  # estimated 0-1


@dataclass(frozen=True)
class GapReport:
    symbol: str

    all_gaps: list[Gap]
    up_gaps: list[Gap]
    down_gaps: list[Gap]

    recent_gap: Gap | None    # most recent gap
    open_gaps: list[Gap]      # unfilled gaps

    total_gaps: int
    filled_gaps: int
    fill_rate: float           # filled / total

    avg_gap_pct: float
    largest_gap: Gap | None

    current_price: float
    bars_used: int
    verdict: str


def _is_in_range(values: list[float], window: int) -> bool:
    """True if values are in a tight range (consolidation)."""
    if len(values) < window:
        return False
    subset = values[-window:]
    rng = max(subset) - min(subset)
    mid = sum(subset) / len(subset)
    return rng / mid < 0.05 if mid > 0 else False


def _classify_gap(
    direction: str,
    gap_pct: float,
    bar_idx: int,
    closes: list[float],
    highs: list[float],
    lows: list[float],
    volumes: list[float] | None,
    avg_vol: float,
) -> tuple[str, float]:
    """Classify a gap and estimate fill probability. Returns (type, fill_prob)."""
    prior_closes = closes[:bar_idx]
    cur_vol = volumes[bar_idx] if volumes else avg_vol

    in_consolidation = _is_in_range(prior_closes, 15) if len(prior_closes) >= 15 else False

    # Detect if we're mid-trend
    if len(prior_closes) >= 20:
        sma20 = sum(prior_closes[-20:]) / 20
        sma5 = sum(prior_closes[-5:]) / 5
        trending_up = sma5 > sma20 * 1.01
        trending_down = sma5 < sma20 * 0.99
    else:
        trending_up = trending_down = False

    # High volume = more significant gap
    high_vol = cur_vol > avg_vol * 1.5 if avg_vol > 0 else False

    # Large gap = more significant
    large_gap = gap_pct > 1.5

    # Classification logic
    if in_consolidation and large_gap:
        # Gap out of consolidation = breakaway
        return "breakaway", 0.3

    if (trending_up and direction == "up") or (trending_down and direction == "down"):
        if large_gap and high_vol:
            # Could be exhaustion or runaway
            # Exhaustion: very large gap at end of extended trend
            if len(prior_closes) >= 30:
                long_move = (prior_closes[-1] - prior_closes[-30]) / prior_closes[-30] * 100.0
                if abs(long_move) > 15:
                    return "exhaustion", 0.8
            return "runaway", 0.45
        elif not large_gap:
            return "runaway", 0.4

    # Common gap: small, no clear context
    if gap_pct < 0.5:
        return "common", 0.75

    return "unknown", 0.5


def analyze(
    bars: list,
    *,
    symbol: str = "",
    min_gap_pct: float = 0.1,
    lookback: int = 60,
) -> GapReport | None:
    """Find and classify gaps in bar history.

    bars: bar objects with .open, .high, .low, .close (and optionally .volume).
    min_gap_pct: minimum gap size as % of price to qualify.
    lookback: how many bars to scan.
    """
    if not bars or len(bars) < 5:
        return None

    try:
        opens = [float(b.open) for b in bars]
        highs = [float(b.high) for b in bars]
        lows = [float(b.low) for b in bars]
        closes = [float(b.close) for b in bars]
    except (AttributeError, TypeError, ValueError):
        return None

    try:
        volumes = [float(b.volume) for b in bars]
    except AttributeError:
        volumes = None

    current = closes[-1]
    if current <= 0:
        return None

    n = len(bars)
    avg_vol = sum(volumes) / len(volumes) if volumes else 0.0
    scan_start = max(1, n - lookback)

    gaps: list[Gap] = []

    for i in range(scan_start, n):
        prev_high = highs[i - 1]
        prev_low = lows[i - 1]
        prev_close = closes[i - 1]
        cur_open = opens[i]
        cur_high = highs[i]
        cur_low = lows[i]

        # Up gap: today's low > yesterday's high
        if cur_low > prev_high:
            gap_size = cur_low - prev_high
            gap_pct = gap_size / prev_close * 100.0 if prev_close > 0 else 0.0
            if gap_pct >= min_gap_pct:
                gtype, fp = _classify_gap("up", gap_pct, i, closes[:i], highs, lows, volumes, avg_vol)
                # Check if subsequently filled
                filled = False
                fill_bar = None
                for j in range(i + 1, n):
                    if lows[j] <= prev_high:
                        filled = True
                        fill_bar = j
                        break
                gaps.append(Gap(
                    bar_idx=i, direction="up",
                    gap_open=round(prev_high, 4), gap_close_=round(cur_low, 4),
                    gap_size=round(gap_size, 4), gap_pct=round(gap_pct, 2),
                    gap_type=gtype, filled=filled, fill_bar=fill_bar,
                    fill_probability=fp,
                ))

        # Down gap: today's high < yesterday's low
        elif cur_high < prev_low:
            gap_size = prev_low - cur_high
            gap_pct = gap_size / prev_close * 100.0 if prev_close > 0 else 0.0
            if gap_pct >= min_gap_pct:
                gtype, fp = _classify_gap("down", gap_pct, i, closes[:i], highs, lows, volumes, avg_vol)
                filled = False
                fill_bar = None
                for j in range(i + 1, n):
                    if highs[j] >= prev_low:
                        filled = True
                        fill_bar = j
                        break
                gaps.append(Gap(
                    bar_idx=i, direction="down",
                    gap_open=round(cur_high, 4), gap_close_=round(prev_low, 4),
                    gap_size=round(gap_size, 4), gap_pct=round(gap_pct, 2),
                    gap_type=gtype, filled=filled, fill_bar=fill_bar,
                    fill_probability=fp,
                ))

    up_gaps = [g for g in gaps if g.direction == "up"]
    down_gaps = [g for g in gaps if g.direction == "down"]
    open_gaps = [g for g in gaps if not g.filled]
    filled_count = sum(1 for g in gaps if g.filled)
    fill_rate = filled_count / len(gaps) if gaps else 0.0
    avg_gap_pct = sum(g.gap_pct for g in gaps) / len(gaps) if gaps else 0.0
    largest = max(gaps, key=lambda g: g.gap_pct) if gaps else None
    recent = gaps[-1] if gaps else None

    # Verdict
    if not gaps:
        verdict = f"No qualifying gaps found in last {lookback} bars (min {min_gap_pct}% threshold)."
    else:
        open_tag = f"{len(open_gaps)} open" if open_gaps else "all filled"
        verdict = (
            f"{len(gaps)} gap(s) found: {len(up_gaps)} up, {len(down_gaps)} down. "
            f"Fill rate: {fill_rate * 100:.0f}% ({open_tag}). "
            f"Avg size: {avg_gap_pct:.2f}%."
        )
        if recent:
            verdict += f" Last: {recent.gap_type} {recent.direction} ({recent.gap_pct:.2f}%)."

    return GapReport(
        symbol=symbol,
        all_gaps=gaps,
        up_gaps=up_gaps,
        down_gaps=down_gaps,
        recent_gap=recent,
        open_gaps=open_gaps,
        total_gaps=len(gaps),
        filled_gaps=filled_count,
        fill_rate=round(fill_rate, 3),
        avg_gap_pct=round(avg_gap_pct, 2),
        largest_gap=largest,
        current_price=round(current, 4),
        bars_used=n,
        verdict=verdict,
    )
