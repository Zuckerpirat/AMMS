"""Relative strength ranking.

Ranks symbols by their performance relative to a benchmark
(or to the portfolio average if no benchmark is provided).

Metrics:
  - abs_return_pct:  symbol return over the lookback window
  - rel_return_pct:  symbol return minus benchmark return
  - rs_score:        0-100 percentile rank among peers (higher = stronger)
  - trend:           "outperforming" | "neutral" | "underperforming"
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RSRow:
    symbol: str
    abs_return_pct: float    # raw % return over window
    rel_return_pct: float    # return vs benchmark (or portfolio avg)
    rs_score: float          # 0-100 percentile among peers
    trend: str               # "outperforming"|"neutral"|"underperforming"
    bars_used: int


@dataclass(frozen=True)
class RSRanking:
    rows: list[RSRow]          # sorted by rs_score descending
    benchmark_return_pct: float
    leader: RSRow | None
    laggard: RSRow | None
    lookback: int


def rank(
    bars_map: dict[str, list],
    *,
    lookback: int = 20,
    benchmark_bars: list | None = None,
) -> RSRanking | None:
    """Compute relative strength ranking for a set of symbols.

    bars_map: dict[symbol -> list[Bar]]
    lookback: return window in bars (default 20)
    benchmark_bars: optional Bar list for benchmark return; if None,
                    uses portfolio-average return as benchmark

    Returns None if bars_map is empty or no valid data.
    """
    if not bars_map:
        return None

    # Compute return for each symbol
    returns: dict[str, tuple[float, int]] = {}  # sym -> (pct_return, bars_used)
    for sym, bars in bars_map.items():
        if not bars or len(bars) < 2:
            continue
        window = bars[-lookback:] if len(bars) >= lookback else bars
        start = window[0].close
        end = window[-1].close
        if start > 0:
            ret = (end - start) / start * 100
            returns[sym] = (ret, len(window))

    if not returns:
        return None

    # Compute benchmark return
    if benchmark_bars and len(benchmark_bars) >= 2:
        bw = benchmark_bars[-lookback:] if len(benchmark_bars) >= lookback else benchmark_bars
        bs, be = bw[0].close, bw[-1].close
        bench_ret = (be - bs) / bs * 100 if bs > 0 else 0.0
    else:
        # Use portfolio average as benchmark
        bench_ret = sum(r for r, _ in returns.values()) / len(returns)

    # Relative returns
    rel_rets = {sym: ret - bench_ret for sym, (ret, _) in returns.items()}

    # Percentile rank (mid-point formula)
    all_abs = sorted(returns.keys(), key=lambda s: returns[s][0])
    n = len(all_abs)

    rows: list[RSRow] = []
    for sym in returns:
        abs_ret, bars_used = returns[sym]
        rel_ret = rel_rets[sym]

        n_below = sum(1 for s in returns if returns[s][0] < abs_ret)
        n_equal = sum(1 for s in returns if returns[s][0] == abs_ret)
        rs_score = (n_below + 0.5 * n_equal) / n * 100

        if rel_ret > 2.0:
            trend = "outperforming"
        elif rel_ret < -2.0:
            trend = "underperforming"
        else:
            trend = "neutral"

        rows.append(RSRow(
            symbol=sym,
            abs_return_pct=round(abs_ret, 2),
            rel_return_pct=round(rel_ret, 2),
            rs_score=round(rs_score, 1),
            trend=trend,
            bars_used=bars_used,
        ))

    rows.sort(key=lambda r: -r.rs_score)
    leader = rows[0] if rows else None
    laggard = rows[-1] if rows else None

    return RSRanking(
        rows=rows,
        benchmark_return_pct=round(bench_ret, 2),
        leader=leader,
        laggard=laggard,
        lookback=lookback,
    )
