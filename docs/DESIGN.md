# AMMS — AI-Assisted Paper Trading Bot

Paper-only trading bot for US stocks and penny stocks. Alpaca Paper API, Python,
Docker, SQLite, optional Telegram alerts. Runs on a cheap Ubuntu VPS, controlled
from a Windows PC over SSH.

**Hard constraints (never violated):**

- Paper trading only — `paper-api.alpaca.markets`, never the live endpoint.
- No margin, no leverage, no options, no shorting.
- Long equity only. Order sides: `buy` and `sell` (to close) only.
- All signals are advisory until the strategy is explicitly enabled.

---

## 1. Realistic architecture

The bot is one Python process inside Docker, structured as a set of cooperating
services that share a SQLite database. Keep it boring: no message bus, no
microservices, no Kubernetes. One container, one DB file, one config file.

```
                +----------------------------+
                |   Windows PC (your laptop) |
                |   SSH / scp / Telegram     |
                +-------------+--------------+
                              | SSH
                              v
+---------------------------------------------------------------+
|                  Ubuntu VPS  (1 vCPU, 1–2 GB RAM)             |
|                                                               |
|   docker compose up                                           |
|   +--------------------------------------------------------+  |
|   |                amms (Python container)                 |  |
|   |                                                        |  |
|   |   scheduler  --->  data_ingest  ---> SQLite (vol)      |  |
|   |        |              |  ^                             |  |
|   |        v              v  |                             |  |
|   |     strategy  --->  signals  --->  risk  --->  broker  |  |
|   |        |                                       (Alpaca)|  |
|   |        v                                               |  |
|   |     notifier  ---------------> Telegram (optional)     |  |
|   |                                                        |  |
|   +--------------------------------------------------------+  |
|                                                               |
|   docker volume: /data  -> amms.sqlite, logs/                 |
+---------------------------------------------------------------+

         External APIs:
          - Alpaca Paper Trading (orders, positions, account)
          - Alpaca Market Data    (bars, quotes)
          - Telegram Bot API      (alerts; optional)
          - Reddit / Pushshift    (sentiment; phase 4)
```

**Why this shape:**

- One process keeps state and timing easy. The scheduler ticks; each tick
  pulls data, evaluates strategy, applies risk rules, places paper orders.
- SQLite is enough for a single-writer workload. WAL mode, one connection per
  thread. Migrate later if you outgrow it — schema lives in `migrations/`.
- Docker makes the VPS reproducible: rebuild from `git pull && docker compose up -d`.
- Control from Windows is just SSH + Telegram commands. No web UI in MVP.

**Module responsibilities:**

| Module | Job | Talks to |
|---|---|---|
| `config` | Load env vars + `config.yaml`, validate at startup | — |
| `broker` | Thin Alpaca client (paper). Buy/sell/cancel/positions/account | Alpaca |
| `data` | Pull bars/quotes, store in SQLite, expose recent-window queries | Alpaca, DB |
| `strategy` | Pure functions: bars → signal (`buy` / `sell` / `hold`) | DB (read) |
| `risk` | Position sizing, max exposure, daily loss cap, cooldowns | DB (read) |
| `executor` | Translates approved signals into paper orders, logs fills | Broker, DB |
| `scheduler` | Cron-like loop; runs jobs on market hours | All |
| `notifier` | Telegram alerts and a small command set (`/status`, `/positions`) | Telegram |
| `cli` | `amms backtest`, `amms run`, `amms status` — for SSH control | All |

---

## 2. Simple MVP plan

The MVP must prove the loop works end-to-end before any AI/sentiment work.

**MVP scope (Phase 1–2 only):**

1. Configure Alpaca paper keys via `.env`.
2. Pull 1-minute and daily bars for a small watchlist (~20 tickers).
3. Run one trivial strategy: SMA(10) vs SMA(30) crossover, long-only.
4. Risk rules: max 5 open positions, max 2% of equity per position, daily
   loss cap −3%, no trades in first/last 5 minutes of session.
5. Place paper market orders during regular US market hours.
6. Log every signal, order, and fill to SQLite.
7. Telegram alert on order placed and on daily summary.
8. `docker compose up -d` on the VPS runs everything.

**Out of scope for MVP:**

- Reddit/WSB sentiment, fundamentals, ML, multi-strategy, web dashboard,
  options, shorting, after-hours, complex order types, multiple accounts.

**Definition of done for MVP:**

- Bot runs 5 consecutive trading days on the VPS without crashing.
- Every order in the Alpaca paper account has a matching row in `orders` and
  `fills` tables.
- A Telegram `/status` command returns equity, open positions, P&L today.
- `amms backtest --from 2025-01-01 --to 2025-12-31 --symbols AAPL,MSFT` runs
  on historical bars and produces a summary.

---

## 3. Folder structure

Flat enough to navigate, structured enough to test. Single package.

```
AMMS/
├── README.md
├── docs/
│   ├── DESIGN.md              <- this file
│   └── runbook.md             <- VPS setup, deploy, recovery (phase 2)
├── docker/
│   ├── Dockerfile
│   └── entrypoint.sh
├── docker-compose.yml
├── .env.example               <- never commit real .env
├── config.example.yaml        <- watchlist, risk caps, strategy params
├── pyproject.toml             <- ruff + pytest + deps
├── Makefile                   <- make run / test / lint / shell
├── migrations/
│   └── 001_init.sql
├── src/amms/
│   ├── __init__.py
│   ├── config.py
│   ├── db.py                  <- SQLite connection, migrations
│   ├── broker/
│   │   ├── __init__.py
│   │   └── alpaca.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── bars.py
│   │   └── universe.py
│   ├── strategy/
│   │   ├── __init__.py
│   │   ├── base.py            <- Strategy protocol
│   │   └── sma_cross.py       <- MVP strategy
│   ├── risk/
│   │   ├── __init__.py
│   │   └── rules.py
│   ├── executor.py
│   ├── scheduler.py
│   ├── notifier/
│   │   ├── __init__.py
│   │   └── telegram.py
│   ├── backtest/
│   │   ├── __init__.py
│   │   └── engine.py
│   └── cli.py                 <- entry point: `amms ...`
└── tests/
    ├── test_risk_rules.py
    ├── test_sma_cross.py
    ├── test_backtest_engine.py
    └── fixtures/
```

---

## 4. Required APIs

| API | Purpose | Auth | Cost | Phase |
|---|---|---|---|---|
| **Alpaca Paper Trading** (`paper-api.alpaca.markets`) | Account, orders, positions | API key + secret | Free | 1 |
| **Alpaca Market Data v2** | Historical + delayed real-time bars/quotes (IEX feed is free) | Same key | Free tier | 1 |
| **Telegram Bot API** | Alerts and `/status` style commands | Bot token + chat ID | Free | 2 |
| **Reddit API** (PRAW) | Pull r/wallstreetbets, r/pennystocks posts | Reddit app client ID/secret | Free, rate-limited | 4 |
| **Pushshift (or arctic-shift mirror)** | Historical Reddit if PRAW is too slow | Public / API key | Free | 4 (optional) |
| **SEC EDGAR** | Fundamentals, filings | None (just User-Agent) | Free | 5 |
| **Financial Modeling Prep / yfinance** | Backup fundamentals, ratios | Free tier key (FMP) | Free tier | 5 (optional) |

Notes:

- Stick to Alpaca's free IEX data feed in MVP. SIP feed costs money and isn't
  needed for paper trading or research.
- All keys live in `.env` on the VPS, loaded via `python-dotenv`. Never commit.
- Add a `BROKER_BASE_URL` env var and assert it contains `paper-api` at startup
  so the bot physically cannot hit the live endpoint.

---

## 5. Exact development phases

Each phase ends with a deployable, demonstrable artifact. Don't start phase N+1
until phase N is running on the VPS.

### Phase 0 — Repo skeleton (½ day)
- `pyproject.toml`, ruff, pytest, basic CI.
- `Dockerfile` + `docker-compose.yml` that boots an empty container.
- `.env.example`, `config.example.yaml`.
- `amms --help` CLI stub.
- **Done when:** `docker compose up` runs and prints "amms ready".

### Phase 1 — Broker + data plumbing (2–3 days)
- Alpaca paper client: `get_account`, `get_positions`, `submit_order`, `cancel_order`.
- Startup guard that refuses to run unless URL is the paper endpoint.
- Market-data fetcher for daily and 1-minute bars; persist to SQLite.
- `amms status` prints account equity and positions over SSH.
- **Done when:** you can place one paper order from the CLI and see it in the DB.

### Phase 2 — MVP strategy loop + Telegram (3–4 days)
- SMA crossover strategy on a 20-symbol watchlist.
- Risk module: position size, max positions, daily loss cap, trading hours.
- Scheduler that runs every minute during US market hours.
- Telegram alerts on order + daily 16:05 ET summary.
- Deploy to VPS, run for one full trading week.
- **Done when:** MVP "definition of done" above is met.

### Phase 3 — Backtester (3–4 days)
- Event-driven backtester that consumes the same `Strategy` interface as live.
- Stats: equity curve, max drawdown, win rate, Sharpe-ish, trade log CSV.
- Walk-forward harness so a strategy is validated on out-of-sample data
  before going live.
- **Done when:** same strategy code runs in backtest and live with identical signals.

### Phase 4 — Sentiment, momentum, volatility, unusual volume (1–2 weeks)
Add as independent feature modules; each writes scores to a `features` table.
- `features/momentum.py` — N-day return, RSI, MACD histogram.
- `features/volatility.py` — ATR, realized vol, vol-of-vol.
- `features/volume.py` — relative volume vs 20-day average, volume spikes.
- `features/reddit.py` — pull WSB/pennystocks daily, ticker mention counts,
  simple sentiment (VADER first; upgrade later).
- A new `strategy/composite.py` combines features into a score and only trades
  the top N. Still long-only, still paper.
- **Done when:** composite strategy beats SMA baseline in backtest *and*
  forward-tests for 2 weeks on paper.

### Phase 5 — Fundamentals + smarter filters (1 week)
- SEC EDGAR pull for basics: market cap, shares outstanding, recent filings.
- Penny-stock filters: price ≥ $1, ADV ≥ $1M, no recent reverse split warnings.
- Halt/circuit-breaker awareness; skip symbols flagged today.

### Phase 6 — AI assist (optional, after everything above works)
- LLM summarizer for end-of-day report: "Here's why we bought X today."
- LLM as a *filter only*, never as the decision-maker — it can veto a trade,
  it cannot create one. Cache responses to avoid spend.

### Phase 7 — Ops polish
- Postgres migration if SQLite hurts (probably won't for one bot).
- Prometheus + Grafana on the VPS for equity, signals/min, error rate.
- Restore-from-backup drill: blow away the VPS, redeploy, verify state.

---

## 6. What to build first to avoid overcomplicating

Build in this exact order. Each step is small and forces the next to be honest.

1. **`.env` + Alpaca client + "hello account"** — one script that prints your
   paper equity. If this is hard, nothing else will be easy.
2. **SQLite schema + one writer** — tables for `bars`, `orders`, `fills`,
   `signals`, `equity_snapshots`. Write a fake order; read it back.
3. **One paper order via CLI** — `amms buy AAPL 1`. Confirm in Alpaca dashboard.
   This is your "the wiring works" milestone.
4. **One scheduled tick** — a job that runs every minute during market hours
   and writes a heartbeat row. No strategy yet.
5. **The dumbest strategy that can lose money** — SMA crossover. It will
   underperform; that's fine, the point is the loop, not the alpha.
6. **Risk rules before anything clever** — caps and cooldowns. A boring bot
   that respects limits is infinitely better than a smart one that doesn't.
7. **Telegram alerts** — only after the bot trades correctly without them.
   Alerts are a UX feature, not a safety feature.
8. **Backtester** — only once a live strategy exists, so the two share an
   interface from day one. Building the backtester first is a classic trap
   that produces beautiful backtests of strategies that can't run live.
9. **Then, and only then, sentiment / fundamentals / AI.** Each new feature
   must beat the SMA baseline in backtest *and* forward-test on paper before
   it gets to influence orders.

**Anti-goals (things to refuse to add until they're truly needed):**

- A web dashboard. SSH + Telegram is enough for one user.
- Multiple brokers. Alpaca is enough.
- A plugin system. You have one user; just edit the code.
- Async everywhere. The bot is I/O-light; threads + a scheduler are fine.
- Postgres / Redis / Kafka. SQLite handles this workload for years.
- "Configurable everything." Hardcode sensible defaults; expose only what you
  actually tune.

If a feature doesn't fit one of the phases above, it goes in `docs/icebox.md`
and waits.
