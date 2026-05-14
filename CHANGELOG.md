# Changelog

## Phase 0 — repo skeleton

- pyproject + ruff + pytest scaffolding
- Docker (non-root user, /data volume) + docker-compose
- Typer CLI stub
- `.env.example`, `config.example.yaml`, `.gitignore`, `Makefile`, CI

## Phase 1 — broker + data plumbing

- Alpaca paper client (httpx): account, positions, submit/get/cancel order
- Hard `paper-api` guard at config + client level
- SQLite + migrations (bars, orders, signals, equity_snapshots)
- Alpaca market-data v2 bars fetcher
- CLI: `status`, `buy`, `sell`, `fetch-bars`, `init-db`

## Phase 2a — strategy + risk + tick

- `Strategy` protocol + `Signal` dataclass
- `SmaCross` strategy
- `RiskConfig` with self-validation; `check_buy` decision
- `config.yaml` loader (`AppConfig`)
- `amms tick` (dry-run by default; `--execute` to act)

## Phase 2b — autonomous loop + Telegram

- Market clock + `ClockStatus`
- Notifier protocol + outbound Telegram
- `executor.run_tick`, `executor.build_daily_summary`
- `scheduler.run_loop` with SIGINT/SIGTERM, daily summary on close
- `amms run` becomes the autonomous loop (default dry-run)
- docker-compose mounts `config.yaml`, restart: unless-stopped

## Phase 3 — backtester

- Event-driven engine: signals at close, fills at next bar's open
- Stats: total return, max drawdown, FIFO win rate
- CSV exports for trades + equity curve
- `amms backtest --from --to --symbols --output --fetch`

## Phase 4a — features + composite strategy

- features/: `n_day_return`, `rsi`, `atr`, `realized_vol`, `relative_volume`
- `CompositeStrategy`: multi-filter long-only screen
- Strategy interface refactored to `list[Bar]`; SMA + executor + backtest
  threaded through

## Phase 4b — feature persistence + top-N ranking

- Migration 002: `features` table + `signals.score` column
- `features.standard_features` snapshot
- `RiskConfig.max_buys_per_tick`; executor sorts by score and trims
- Executor persists features per tick

## Phase 4c — Reddit sentiment + LLM summary

- Built-in tiny sentiment lexicon (no external deps)
- `RedditSentimentCollector` (OAuth client_credentials)
- `set_sentiment_overlay` consumed by `CompositeStrategy`
- Optional `augment_summary` (Anthropic API) on the daily Telegram message

## Phase 5 — exits + filters + day-trade safety

- Executor processes sell signals; `min_hold_days` gate (live + backtest)
- `Account.daytrade_count`; PDT guard refuses sells creating day-trades
  when `equity < pdt_min_equity`
- `UniverseFilter`: `min_price`, `max_price`, `min_avg_dollar_volume`,
  `require_tradable` (Alpaca /v2/assets)
- `force_close_minutes_before_close` in scheduler

## Phase 6 — operations

- Inbound Telegram: `/status`, `/positions`, `/equity`, `/signals`,
  `/lastorders`, `/pause`, `/resume`, `/help`
- Auto-pause circuit breaker after 5 consecutive tick errors
- Prometheus `/metrics` + `/healthz` (`AMMS_METRICS_PORT`)
- `amms_ticks_total`, `amms_equity_dollars`, `amms_cash_dollars`,
  `amms_daytrade_count`, `amms_last_tick_unix`, `amms_orders_total{side}`

## Phase 7 — backtester expansion + ergonomics

- Walk-forward harness (`run_walk_forward`, CLI `walk-forward`)
- Intraday backtester (`run_intraday_backtest`, `--intraday`)
- `MeanReversion` + `Breakout` strategies; registry-driven
- CLI: `compare-strategies` (all registered), `list-strategies`,
  `signals` / `orders` / `equity-log` (read DB), `summary-today` (`--llm`),
  `doctor` (pre-flight self-check), `watchlist {show,add,remove}`,
  `preview-signal SYMBOL`, `order-status ORDER_ID`, `cancel-all` / `close-all`
  (emergency), `apply-profile NAME`, `vacuum`, `reset-db`

## Docs + repo polish

- `docs/SETUP.md` — non-technical VPS walkthrough
- `docs/PROFILES.md` — three example config.yaml shapes
- `docs/OPS.md` — daily checks, backups, metrics, profile switching,
  circuit-breaker response
- `profiles/swing.yaml`, `profiles/day-trade.yaml`, `profiles/penny.yaml`
  ready to `cp profiles/X.yaml config.yaml`
- `scripts/backup.sh` — SQLite online .backup out of the container
- MIT LICENSE with paper-only disclaimer
- 198 tests; ruff clean
