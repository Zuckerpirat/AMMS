# AMMS

AI-assisted paper trading bot for US equities. Paper-only, long-only, no
margin, no leverage, no options, no shorting. Runs in Docker on a small Ubuntu
VPS and is controlled from a Windows PC over SSH and Telegram.

See `docs/DESIGN.md` for the full architecture and phased plan.

## Quick start (local dev)

```sh
make install        # pip install -e ".[dev]"
make test           # pytest
make lint           # ruff check
amms --help
amms run            # prints "amms <version> ready"
```

## Quick start (Docker)

```sh
cp .env.example .env             # fill in keys when needed (Phase 1+)
cp config.example.yaml config.yaml
make build
make up                          # docker compose up
```

The container prints `amms <version> ready` and exits. The real loop arrives
in Phase 1 (broker wiring) and Phase 2 (strategy + scheduler).
