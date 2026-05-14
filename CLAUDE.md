# AMMS — Long-term Architecture & Principles

This file is the authoritative reference for AMMS's long-term direction.
Read this before suggesting major changes. Build incrementally toward it.

## Vision

AMMS is evolving into a **modular AI-assisted financial research and paper
trading platform**. It is NOT a collection of disconnected AI scripts and
NOT a fake "money printer".

Long-term goals:
- autonomous financial research
- probabilistic decision support
- macro and geopolitical awareness
- multi-strategy experimentation
- explainable market analysis
- safe paper trading
- modular and maintainable architecture

The system must stay: **explainable, testable, modular, risk-aware,
maintainable, scalable.**

## Core architecture principles

Every module must have:
- a clearly defined responsibility
- clearly defined inputs/outputs
- structured logging
- independent testability
- minimal coupling to unrelated systems

No "god modules". A single module must NEVER do all of: fetch data, analyze
markets, manage risk, execute trades, and generate reports.

## Required system layers

1. **Data layer** — APIs, market data, news ingestion, social scraping,
   macro datasets, caching, normalization. No trading logic.
2. **Analysis layer** — indicators, sentiment, macro scoring, volatility,
   fundamentals, regime detection. No trade execution.
3. **Strategy layer** — strategy-specific logic, trade hypotheses,
   entry/exit ideas, holding-period classification. Strategies stay
   modular and independent.
4. **Risk layer** — exposure, drawdown protection, position sizing,
   volatility adjustment, stress handling, emergency shutdown. **Risk
   layer has veto power over strategy signals.**
5. **Execution layer** — paper trade execution, simulation, portfolio
   tracking, trade journaling. No strategy generation.
6. **Reporting & AI research layer** — dashboards, explanations, reports,
   summaries, watchlists, strategy evaluations.

## Central decision engine

There must be a central decision engine that:
- aggregates outputs from all modules
- resolves conflicting signals
- weighs confidence levels
- applies portfolio/risk constraints
- determines whether a trade is allowed

**No isolated module ever executes trades on its own.**

## Multi-strategy / multi-mode

The architecture must support multiple independent strategy modes that
can be switched dynamically (not all need to be implemented immediately).

### Mode 1 — Conservative Research
Long-term research, macro analysis, quality companies, low volatility,
defensive during stress. Slow, low turnover, fundamentals + macro heavy.

### Mode 2 — Swing / Momentum
Momentum trades, technical analysis, medium-term holds, sentiment +
trend following. Active but controlled, dynamic stops.

### Mode 3 — High-Risk Meme / Experimental
Meme stocks, retail hype, social spikes, unusual volume. Short holds,
aggressive momentum.
**MUST remain sandboxed:** separate capital allocation, separate risk
limits, separate performance tracking, separate trade journal, separate
scoring logic. Treat as highly experimental.

### Mode 4 — Event-Driven
Earnings, Fed, tariffs, geopolitical, macro shocks. Reactive
positioning, hedging, increased cash during uncertainty, sector rotation.

## Future macro & geopolitical intelligence

Eventually classify geopolitical risk, estimate sector exposure, adjust
risk dynamically, trigger defensive behavior during uncertainty.

Examples to be aware of: Fed decisions, CPI, unemployment, bond yields,
oil, USD, VIX, tariffs, sanctions, China/Taiwan, conflicts, trade
restrictions, sector-specific policy.

## Future AI research agent

An AI-powered research assistant that:
- summarizes market conditions
- explains portfolio decisions
- analyzes sectors and geopolitical risks
- generates monthly research reports, watchlists, opportunity/risk
  summaries

Requirements: must explain uncertainty, avoid fake certainty,
distinguish facts from speculation, cite sources.

## Explainability requirement

Every trade decision must include:
- reasoning
- confidence level
- relevant macro conditions
- sentiment factors
- technical signals
- risk considerations
- expected holding horizon

**No "AI says buy" black-box behavior.**

## Quality control

Implement structured logging, module-level tests, backtesting, performance
benchmarking, config validation, trade journaling, auditability.

Track: Sharpe, max drawdown, win rate, profit factor, benchmark
comparison, regime-specific performance.

## Development rules

- Prefer simplicity over complexity
- Avoid overengineering and fake AI sophistication
- Avoid hidden coupling between modules
- Build incrementally
- Never implement too many systems simultaneously
- Prioritize robustness over feature count
- Prefer maintainable architecture over rapid expansion

**Before adding major new features:** review existing architecture,
identify redundancy/overlap/unnecessary complexity.

## Communication style with the user

- The user is not a programmer. Use simple language, German when the
  user writes German.
- Keep responses short and concrete to preserve token budget.
- Always end with a clear next step the user can take.
- Explain *what* a change does in business terms, not implementation
  details.
