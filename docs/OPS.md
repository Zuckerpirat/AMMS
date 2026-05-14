# Operations runbook

How to keep the bot alive on the VPS once it's deployed. Assumes you've
worked through `docs/SETUP.md` already.

---

## Daily checks (~2 minutes)

```sh
ssh root@DEINE_IP
cd /opt/amms

docker compose ps                       # container up?
docker compose logs --tail=200          # any tick failures?
docker compose run --rm amms status     # equity sane?
docker compose run --rm amms orders --limit 10
```

If `/pause` was flipped by the circuit breaker, the logs will tell you why
and Telegram will have pinged you. Investigate, then `/resume` via Telegram.

---

## Backups

`scripts/backup.sh` runs SQLite's online `.backup` against the live DB
and pulls the snapshot out of the container.

```sh
./scripts/backup.sh                 # ./backups/amms-<UTC>.sqlite
```

Recommended cron (host crontab):
```
0 5 * * * cd /opt/amms && ./scripts/backup.sh /var/backups/amms >/dev/null 2>&1
```

Keep one week of dailies. Restore by stopping the bot, copying the snapshot
back to `/data/amms.sqlite` in the volume, and starting again.

---

## Metrics + alerts

Set `AMMS_METRICS_PORT=9100` in `.env` to expose Prometheus metrics on the
container's port 9100. Useful gauges:

- `amms_equity_dollars` — alert if it drops more than X% intraday
- `amms_last_tick_unix` — alert if `time() - amms_last_tick_unix > 300`
  (bot is silent during market hours)
- `amms_daytrade_count` — watch for approaching PDT limit
- `amms_orders_total{side="buy"}` / `{side="sell"}`

A `/healthz` endpoint on the same port returns `ok` for liveness checks.

Quick Grafana setup: scrape `http://<vps-ip>:9100/metrics` every 30s. Add
the host to UFW only after restricting source IPs.

---

## Updating the bot

```sh
cd /opt/amms
git fetch origin
git checkout main            # or the active dev branch
git pull
docker compose build
docker compose up -d
docker compose logs -f
```

The DB schema migrates automatically on startup. Backups before major
upgrades are still a good idea.

---

## Switching profiles

Edit `config.yaml` (see `docs/PROFILES.md` for the three canonical
shapes), then:

```sh
docker compose down
docker compose up -d
```

You do not need to rebuild the image just to change config — only when
the source changes.

---

## When the bot pauses itself

`MAX_CONSECUTIVE_TICK_ERRORS = 5` in `scheduler.py`. After 5 ticks in a
row throw, the bot:

1. Sets the PauseFlag (no new orders).
2. Pings Telegram with a clear message.
3. Keeps ticking so you can read `/status` and tail logs.

To resume: investigate logs, fix the root cause, then send `/resume`
from your Telegram chat. The counter resets on the next successful tick.

---

## When you want to fully stop

```sh
docker compose down
```

That's it. The DB volume is persistent. Restart any time with
`docker compose up -d`. The container starts in dry-run by default — you
still need the `--execute` flag in `command:` in `docker-compose.yml` to
re-arm live paper trading after a stop/start.
