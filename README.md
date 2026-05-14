# AMMS

AI-assisted paper trading bot for US equities. Paper-only, long-only, no
margin, no leverage, no options, no shorting. Runs in Docker on a small Ubuntu
VPS and is controlled from a Windows PC over SSH and Telegram.

## Documentation

- **[`docs/SETUP.md`](docs/SETUP.md)** — step-by-step setup guide for non-technical
  users (VPS, SSH, Docker, deployment, first run). Start here if you want to
  actually run the bot.
- [`docs/PROFILES.md`](docs/PROFILES.md) — three example `config.yaml` shapes
  (swing, day-trade, penny-stock) you can paste into `config.yaml`.
- [`docs/OPS.md`](docs/OPS.md) — daily checks, backups, metrics, updates.
- [`docs/DESIGN.md`](docs/DESIGN.md) — full architecture and phased plan.

## Quick start (local dev)

```sh
make install        # pip install -e ".[dev]"
make test           # pytest
make lint           # ruff check
amms --help
amms tick           # one strategy pass (dry-run; needs .env + config.yaml)
amms run            # autonomous loop (dry-run; add --execute to place orders)
```

## Quick start (Docker on a VPS)

See `docs/SETUP.md` for the full walkthrough. The short version:

```sh
cp .env.example .env             # fill in your Alpaca paper keys
cp config.example.yaml config.yaml
docker compose build
docker compose up -d             # runs autonomously, restart unless-stopped
docker compose logs -f           # watch what it does
```

## CLI commands

| Command | What it does |
|---|---|
| `amms status` | Show paper equity, open positions, record an equity snapshot |
| `amms tick` | One pass of the strategy (dry-run by default; `--execute` to act) |
| `amms run` | The autonomous loop (dry-run by default; `--execute` to act) |
| `amms buy SYM N` | Manually place a paper BUY order |
| `amms sell SYM N` | Manually place a paper SELL order (only sells what you hold) |
| `amms fetch-bars SYM --start --end` | Cache historical daily bars in SQLite |
| `amms backtest --from --to --symbols` | Run a historical backtest |
| `amms walk-forward --from --to ...` | Walk-forward (rolling out-of-sample) backtest |
| `amms signals [--symbol] [--limit]` | Recent strategy signals from the DB |
| `amms orders [--limit]` | Recent orders from the DB |
| `amms equity-log [--limit]` | Recent equity snapshots |
| `amms init-db` | Apply DB migrations explicitly |
| `amms doctor` | Pre-flight self-check (env, config, DB, Alpaca, Telegram) |
| `amms compare-strategies --from --to` | A/B SMA vs Composite on the same window |
| `amms wsb-scan` | Discover trending tickers on r/wallstreetbets (mentions + sentiment) |

## What it does

- **Strategies**: SMA crossover, multi-feature composite (momentum + RSI +
  volatility + relative volume + optional Reddit sentiment overlay), Bollinger
  mean reversion, Donchian breakout. Pluggable: `register_strategy("name", cls)`.
- **Features**: 20-day momentum, RSI, ATR, realized vol, relative volume,
  Reddit/WSB sentiment with built-in tiny lexicon.
- **Risk**: configurable max positions, max position pct, daily-loss kill
  switch, top-N buys per tick, min hold days, PDT-rule guard for sub-$25k
  accounts, force-close-before-close for day trading.
- **Universe filter**: min/max price, min average dollar volume, optional
  "must be tradable on Alpaca" check.
- **Backtesters**: daily-stepping (default) and timestamp-stepping
  (intraday, `--intraday`). Walk-forward harness for out-of-sample
  validation. `amms compare-strategies` A/Bs all registered strategies on
  the same window.
- **Operations**: Telegram outbound alerts, inbound commands (`/status`,
  `/positions`, `/signals`, `/lastorders`, `/equity`, `/pause`, `/resume`,
  `/help`), Prometheus `/metrics` + `/healthz`, auto-pause circuit breaker
  after 5 consecutive tick errors, optional LLM-augmented daily summary
  via Anthropic API.
- **Data**: Alpaca Paper Trading API for orders, Alpaca market data v2
  for bars, SEC EDGAR for fundamentals, Reddit OAuth for sentiment.
  Local SQLite holds bars, orders, signals (with score), equity snapshots,
  features audit trail.

## Safety guarantees

- The bot refuses to start if `ALPACA_BASE_URL` does not contain `paper-api`.
- The Alpaca client refuses to be instantiated against a non-paper endpoint.
- The DB schema constrains order side to `buy` or `sell` (no shorting at the
  data layer either).
- `amms sell` refuses if requested quantity exceeds the held position.
