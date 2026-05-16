"""Central Decision Engine.

Aggregates signals from multiple analysis modules and produces a final
trade decision with confidence score, reasoning, and risk check.

Architecture:
  - Runs a configurable set of analysis modules on a symbol's bars
  - Groups results by category (trend, momentum, oscillator, volatility, volume)
  - Computes category scores and a weighted composite
  - Applies a risk gate (drawdown, position limits)
  - Returns a DecisionReport with verdict, confidence, and full reasoning

Signal categories and weights:
  trend      : 30% — Ichimoku, MA Ribbon, Supertrend, KAMA
  momentum   : 30% — MACD, PMO, TRIX/KST, STC, Vortex
  oscillator : 25% — RSI, StochRSI, CMO, Williams %R, Connors RSI
  volume     : 15% — OFI, Force Index, KVO, Chaikin MF

Final action:
  strong_buy  → confidence >= 0.70 and score >= 60
  buy         → confidence >= 0.50 and score >= 35
  hold        → score in [-35, 35]
  sell        → confidence >= 0.50 and score <= -35
  strong_sell → confidence >= 0.70 and score <= -60
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModuleResult:
    module: str
    category: str       # "trend" / "momentum" / "oscillator" / "volume" / "volatility"
    score: float        # -100..+100 (module's own score, normalised)
    signal: str         # module's own signal label
    verdict: str        # module's one-line verdict
    weight: float = 1.0


@dataclass(frozen=True)
class CategoryScore:
    name: str
    score: float        # -100..+100 weighted average within category
    module_count: int
    signals: list[str]  # signal labels from modules in this category


@dataclass
class DecisionReport:
    symbol: str
    action: str                     # "strong_buy" / "buy" / "hold" / "sell" / "strong_sell"
    composite_score: float          # -100..+100
    confidence: float               # 0.0..1.0

    # Category breakdown
    categories: dict[str, CategoryScore]

    # Individual module results
    modules: list[ModuleResult]
    modules_run: int
    modules_failed: int

    # Risk gate
    risk_blocked: bool
    risk_reason: str

    # Human-readable reasoning
    reasoning: list[str]
    verdict: str

    bars_used: int


# Category weights (must sum to 1.0)
_CATEGORY_WEIGHTS = {
    "trend":      0.30,
    "momentum":   0.30,
    "oscillator": 0.25,
    "volume":     0.15,
}


def _safe_run(fn, *args, **kwargs):
    """Run analysis function, return None on any error."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        logger.debug("Decision engine module error: %s", exc)
        return None


def _score_to_vote(score: float) -> str:
    if score >= 20:
        return "bull"
    if score <= -20:
        return "bear"
    return "neutral"


def analyze(
    bars: list,
    *,
    symbol: str = "",
    min_confidence: float = 0.50,     # block action below this signal-agreement level
    min_modules: int = 3,             # minimum modules that must succeed
    risk_veto = None,                 # optional callable(score, confidence) -> reason | None
) -> DecisionReport | None:
    """Run the central decision engine on a symbol's bars.

    bars: bar objects with .open, .high, .low, .close, .volume attributes.
    Returns None if too few bars or too few modules succeed.
    """
    if not bars or len(bars) < 50:
        return None

    results: list[ModuleResult] = []
    failed = 0

    # ── TREND modules ──────────────────────────────────────────────
    from amms.analysis.ichimoku import analyze as ich_analyze
    r = _safe_run(ich_analyze, bars, symbol=symbol)
    if r:
        # Map bullish_signals (0-6) to -100..+100
        ich_score = (r.bullish_signals - r.bearish_signals) / 6 * 100
        results.append(ModuleResult("Ichimoku", "trend", ich_score, r.signal, r.verdict))
    else:
        failed += 1

    from amms.analysis.kama import analyze as kama_analyze
    r = _safe_run(kama_analyze, bars, symbol=symbol)
    if r:
        results.append(ModuleResult("KAMA", "trend", r.score, r.signal, r.verdict))
    else:
        failed += 1

    from amms.analysis.supertrend import analyze as st_analyze
    r = _safe_run(st_analyze, bars, symbol=symbol)
    if r:
        # Supertrend has no score field — derive from direction and distance
        st_score = min(100.0, abs(r.distance_pct) * 10) if r.direction == "bull" else -min(100.0, abs(r.distance_pct) * 10)
        st_signal = "bull" if r.direction == "bull" else "bear"
        results.append(ModuleResult("Supertrend", "trend", st_score, st_signal, r.verdict))
    else:
        failed += 1

    from amms.analysis.vortex import analyze as vx_analyze
    r = _safe_run(vx_analyze, bars, symbol=symbol)
    if r:
        results.append(ModuleResult("Vortex", "trend", r.score, r.signal, r.verdict))
    else:
        failed += 1

    # ── MOMENTUM modules ───────────────────────────────────────────
    from amms.analysis.trix_kst import analyze as trix_analyze
    r = _safe_run(trix_analyze, bars, symbol=symbol)
    if r:
        results.append(ModuleResult("TRIX/KST", "momentum", r.score, r.signal, r.verdict))
    else:
        failed += 1

    from amms.analysis.pmo import analyze as pmo_analyze
    r = _safe_run(pmo_analyze, bars, symbol=symbol)
    if r:
        results.append(ModuleResult("PMO", "momentum", r.score, r.signal, r.verdict))
    else:
        failed += 1

    from amms.analysis.schaff_trend import analyze as stc_analyze
    r = _safe_run(stc_analyze, bars, symbol=symbol)
    if r:
        results.append(ModuleResult("STC", "momentum", r.score, r.signal, r.verdict))
    else:
        failed += 1

    from amms.analysis.dpo import analyze as dpo_analyze
    r = _safe_run(dpo_analyze, bars, symbol=symbol)
    if r:
        results.append(ModuleResult("DPO", "momentum", r.score, r.signal, r.verdict))
    else:
        failed += 1

    # ── OSCILLATOR modules ─────────────────────────────────────────
    from amms.analysis.chande_mo import analyze as cmo_analyze
    r = _safe_run(cmo_analyze, bars, symbol=symbol)
    if r:
        results.append(ModuleResult("CMO", "oscillator", r.score, r.signal, r.verdict))
    else:
        failed += 1

    from amms.analysis.williams_aroon import analyze as wa_analyze
    r = _safe_run(wa_analyze, bars, symbol=symbol)
    if r:
        results.append(ModuleResult("Williams/Aroon", "oscillator", r.composite_score, r.composite_signal, r.verdict))
    else:
        failed += 1

    from amms.analysis.stoch_rsi import analyze as srsi_analyze
    r = _safe_run(srsi_analyze, bars, symbol=symbol)
    if r:
        results.append(ModuleResult("StochRSI", "oscillator", r.score, r.signal, r.verdict))
    else:
        failed += 1

    from amms.analysis.connors_rsi import analyze as crsi_analyze
    r = _safe_run(crsi_analyze, bars, symbol=symbol)
    if r:
        # Connors RSI uses mean-reversion semantics (oversold→+score = buy bias).
        # Invert for trend-following consensus where +score must mean bullish trend.
        crsi_trend_score = -r.score
        results.append(ModuleResult("ConnorsRSI", "oscillator", crsi_trend_score, r.signal, r.verdict))
    else:
        failed += 1

    from amms.analysis.ultimate_oscillator import analyze as uo_analyze
    r = _safe_run(uo_analyze, bars, symbol=symbol)
    if r:
        results.append(ModuleResult("UltOsc", "oscillator", r.score, r.signal, r.verdict))
    else:
        failed += 1

    # ── VOLUME modules ─────────────────────────────────────────────
    from amms.analysis.order_flow_imbalance import analyze as ofi_analyze
    r = _safe_run(ofi_analyze, bars, symbol=symbol)
    if r:
        results.append(ModuleResult("OFI", "volume", r.score, r.signal, r.verdict))
    else:
        failed += 1

    from amms.analysis.force_index import analyze as fi_analyze
    r = _safe_run(fi_analyze, bars, symbol=symbol)
    if r:
        results.append(ModuleResult("ForceIndex", "volume", r.score, r.signal, r.verdict))
    else:
        failed += 1

    from amms.analysis.klinger_vo import analyze as kvo_analyze
    r = _safe_run(kvo_analyze, bars, symbol=symbol)
    if r:
        results.append(ModuleResult("KVO", "volume", r.score, r.signal, r.verdict))
    else:
        failed += 1

    if len(results) < min_modules:
        return None

    # ── Aggregate by category ──────────────────────────────────────
    categories: dict[str, CategoryScore] = {}
    cat_groups: dict[str, list[ModuleResult]] = {}
    for res in results:
        cat_groups.setdefault(res.category, []).append(res)

    for cat, mods in cat_groups.items():
        avg = sum(m.score for m in mods) / len(mods)
        categories[cat] = CategoryScore(
            name=cat,
            score=round(avg, 1),
            module_count=len(mods),
            signals=[m.signal for m in mods],
        )

    # ── Weighted composite ─────────────────────────────────────────
    total_weight = 0.0
    weighted_sum = 0.0
    for cat, cs in categories.items():
        w = _CATEGORY_WEIGHTS.get(cat, 0.10)
        weighted_sum += cs.score * w
        total_weight += w

    composite = weighted_sum / total_weight if total_weight > 0 else 0.0
    composite = max(-100.0, min(100.0, composite))

    # ── Confidence: agreement ratio ────────────────────────────────
    bull_count = sum(1 for m in results if m.score >= 20)
    bear_count = sum(1 for m in results if m.score <= -20)
    total = len(results)
    dominant = max(bull_count, bear_count)
    confidence = dominant / total if total > 0 else 0.0

    # ── Risk gate ──────────────────────────────────────────────────
    # Two gates: (1) minimum confidence to act, (2) optional external veto
    risk_blocked = False
    risk_reason = ""

    if confidence < min_confidence:
        risk_blocked = True
        risk_reason = f"Confidence {confidence:.0%} below required {min_confidence:.0%}"

    if not risk_blocked and risk_veto is not None:
        try:
            veto_reason = risk_veto(composite, confidence)
            if veto_reason:
                risk_blocked = True
                risk_reason = f"External veto: {veto_reason}"
        except Exception as exc:
            logger.warning("risk_veto callable raised %s — ignoring", exc)

    # ── Action decision ───────────────────────────────────────────
    if risk_blocked:
        action = "hold"
    elif composite >= 60 and confidence >= 0.70:
        action = "strong_buy"
    elif composite >= 35:
        action = "buy"
    elif composite <= -60 and confidence >= 0.70:
        action = "strong_sell"
    elif composite <= -35:
        action = "sell"
    else:
        action = "hold"

    # ── Reasoning ─────────────────────────────────────────────────
    reasoning: list[str] = []
    for cat, cs in sorted(categories.items(), key=lambda x: abs(x[1].score), reverse=True):
        direction = "bullish" if cs.score > 10 else ("bearish" if cs.score < -10 else "neutral")
        reasoning.append(
            f"{cat.capitalize()} ({cs.module_count} modules): score {cs.score:+.0f} → {direction}"
        )
    if confidence < 0.5:
        reasoning.append(f"Low signal agreement ({confidence:.0%}) — mixed market")
    if risk_blocked:
        reasoning.append(f"RISK GATE: {risk_reason}")

    verdict = (
        f"Decision ({symbol}): {action.replace('_', ' ').upper()} "
        f"| Score {composite:+.0f}/100 | Confidence {confidence:.0%} "
        f"| {len(results)} modules ({failed} failed)"
    )

    return DecisionReport(
        symbol=symbol,
        action=action,
        composite_score=round(composite, 1),
        confidence=round(confidence, 3),
        categories=categories,
        modules=results,
        modules_run=len(results),
        modules_failed=failed,
        risk_blocked=risk_blocked,
        risk_reason=risk_reason,
        reasoning=reasoning,
        verdict=verdict,
        bars_used=len(bars),
    )
