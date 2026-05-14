# Trading Profiles

Three example `config.yaml` shapes for the three trading styles AMMS supports.
You switch profiles by editing `config.yaml` — no code changes.

After editing, restart:
```sh
docker compose down && docker compose up -d
```

---

## Swing profile (recommended starting point)

3-4 day holds on big-cap names. Daily bars. Forgiving of small wobbles.

```yaml
watchlist:
  - AAPL
  - MSFT
  - NVDA
  - AMD
  - GOOGL
  - META
  - TSLA
  - AMZN

strategy:
  name: composite
  timeframe: 1Day
  params:
    momentum_n: 20
    momentum_min: 0.05
    rsi_max: 70
    vol_max: 0.40
    rvol_min: 1.2

risk:
  max_open_positions: 5
  max_position_pct: 0.05
  daily_loss_pct: -0.03
  min_hold_days: 2
  max_buys_per_tick: 2
  force_close_minutes_before_close: 0

scheduler:
  tick_seconds: 3600
```

---

## Day-trading profile

Intraday 5-minute bars. Flatten 15 minutes before close so nothing carries
overnight. PDT guard active so the bot will not lock itself out below \$25k.

```yaml
watchlist:
  - SPY
  - QQQ
  - NVDA
  - TSLA

strategy:
  name: composite
  timeframe: 5Min
  params:
    momentum_n: 12
    momentum_min: 0.005
    rsi_max: 75
    vol_max: 1.0
    rvol_min: 1.5

risk:
  max_open_positions: 3
  max_position_pct: 0.02
  daily_loss_pct: -0.02
  min_hold_days: 0
  max_buys_per_tick: 1
  pdt_min_equity: 25000
  pdt_max_day_trades: 3
  force_close_minutes_before_close: 15

scheduler:
  tick_seconds: 60
```

Note: Alpaca's free IEX feed has ~15min delay; for real day-trading you
want the SIP feed (\$9/mo). On paper it still validates the logic.

---

## Penny-stock profile

Strict liquidity filter and minimum price, smaller positions, longer holds.
Composite strategy with sentiment enabled.

```yaml
watchlist:
  - SNDL
  - PLUG
  - RIOT
  - MARA

strategy:
  name: composite
  timeframe: 1Day
  params:
    momentum_n: 10
    momentum_min: 0.08
    rsi_max: 75
    vol_max: 0.80
    rvol_min: 2.0
    sentiment_weight: 0.5
    sentiment_min: -0.3

risk:
  max_open_positions: 4
  max_position_pct: 0.01
  daily_loss_pct: -0.02
  min_hold_days: 1
  max_buys_per_tick: 2

universe:
  min_price: 1.00
  min_avg_dollar_volume: 1000000
  adv_lookback: 20

scheduler:
  tick_seconds: 1800
```

For sentiment to do anything, set `REDDIT_CLIENT_ID` and
`REDDIT_CLIENT_SECRET` in `.env` (free Reddit app credentials). Without them
the bot still trades — it just skips the sentiment overlay.

---

## WSB profile

Curated WSB favorites (large-cap memes + crypto-adjacent + classic tickers),
composite strategy weighted toward Reddit sentiment, and **per-position
stop-loss + trailing stop are enabled by default** because these names move
fast in both directions.

```yaml
watchlist:
  - NVDA
  - TSLA
  - AMD
  - PLTR
  - SMCI
  - MSTR
  - COIN
  - MARA
  - RIOT
  - GME
  - AMC
  - SOFI
  - RIVN

strategy:
  name: composite
  timeframe: 1Day
  params:
    momentum_n: 10
    momentum_min: 0.04
    rsi_max: 75
    vol_max: 0.80
    rvol_min: 1.5
    sentiment_weight: 0.4
    sentiment_min: -0.2

risk:
  max_open_positions: 5
  max_position_pct: 0.03
  daily_loss_pct: -0.04
  min_hold_days: 1
  max_buys_per_tick: 2
  stop_loss_pct: 0.06
  trailing_stop_pct: 0.12

universe:
  min_price: 2.00
  min_avg_dollar_volume: 5000000
  adv_lookback: 20
  require_tradable: true

scheduler:
  tick_seconds: 1800
```

Use `amms wsb-scan` periodically to discover *additional* trending tickers
worth adding to the watchlist by hand.

---

## How to choose

- Just starting? **Swing.** Lets the bot trade ~1-3 times per week per
  symbol, easy to follow what it's doing.
- Account at \$25k+ and you want excitement? **Day-trade.** PDT rule no
  longer locks you out.
- Hunting volatility? **Penny.** Smaller positions, strict filters, accept
  more whipsaw.
- Want WSB-driven trades with safety nets? **WSB.** Stop-loss baked in,
  sentiment-weighted scoring, ~30-min check interval.

Always run a week in dry-run mode first (`amms run` without `--execute`).
