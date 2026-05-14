from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from amms import __version__, db
from amms.backtest import (
    BacktestConfig,
    BacktestStats,
    compute_stats,
    run_backtest,
    run_intraday_backtest,
    run_walk_forward,
    write_trades_csv,
)
from amms.backtest.stats import write_equity_curve_csv
from amms.broker import AlpacaClient
from amms.config import ConfigError, load_app_config, load_settings
from amms.data import MarketDataClient, upsert_bars
from amms.executor import TickResult, run_tick
from amms.scheduler import run_loop
from amms.strategy import Signal, build_strategy

app = typer.Typer(
    help="AI-assisted paper trading bot for US equities (paper-only, long-only).",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _settings_or_die() -> object:
    try:
        return load_settings()
    except ConfigError as e:
        console.print(f"[red]Configuration error:[/red] {e}")
        raise typer.Exit(code=2) from e


def _app_config_or_die() -> object:
    try:
        return load_app_config()
    except ConfigError as e:
        console.print(f"[red]Configuration error:[/red] {e}")
        raise typer.Exit(code=2) from e


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@app.command()
def run(
    execute: bool = typer.Option(
        False,
        "--execute",
        help="Place real paper orders for approved buys. Default is a dry run.",
    ),
    bars_back: int = typer.Option(
        90,
        "--bars-back",
        help="Number of daily bars to pull per symbol for the strategy.",
    ),
) -> None:
    """Start the autonomous trading loop.

    Ticks during US market hours, idles while the market is closed, and sends
    a daily summary via Telegram (if configured) once per session.
    """
    settings = _settings_or_die()
    config = _app_config_or_die()
    _configure_logging(settings.log_level)
    console.print(
        f"amms {__version__} starting ({'live (paper)' if execute else 'dry-run'})"
    )
    run_loop(settings, config, execute=execute, bars_back=bars_back)


@app.command(name="init-db")
def init_db() -> None:
    """Create or migrate the local SQLite database."""
    settings = _settings_or_die()
    conn = db.connect(settings.db_path)
    try:
        applied = db.migrate(conn)
    finally:
        conn.close()
    console.print(f"Applied [bold]{applied}[/bold] migration(s) to {settings.db_path}")


@app.command()
def status() -> None:
    """Print account equity, open positions, and record an equity snapshot."""
    settings = _settings_or_die()
    conn = db.connect(settings.db_path)
    db.migrate(conn)
    try:
        with AlpacaClient(
            settings.alpaca_api_key,
            settings.alpaca_api_secret,
            settings.alpaca_base_url,
        ) as client:
            account = client.get_account()
            positions = client.get_positions()
        db.insert_equity_snapshot(conn, account)
    finally:
        conn.close()

    acc_table = Table(title="Account (paper)")
    acc_table.add_column("equity", justify="right")
    acc_table.add_column("cash", justify="right")
    acc_table.add_column("buying power", justify="right")
    acc_table.add_column("status")
    acc_table.add_row(
        f"${account.equity:,.2f}",
        f"${account.cash:,.2f}",
        f"${account.buying_power:,.2f}",
        account.status,
    )
    console.print(acc_table)

    if not positions:
        console.print("No open positions.")
        return

    pos_table = Table(title="Open positions")
    pos_table.add_column("symbol")
    pos_table.add_column("qty", justify="right")
    pos_table.add_column("avg entry", justify="right")
    pos_table.add_column("mkt value", justify="right")
    pos_table.add_column("unrealized P&L", justify="right")
    for p in positions:
        pos_table.add_row(
            p.symbol,
            f"{p.qty:g}",
            f"${p.avg_entry_price:,.2f}",
            f"${p.market_value:,.2f}",
            f"${p.unrealized_pl:,.2f}",
        )
    console.print(pos_table)


@app.command()
def buy(
    symbol: str = typer.Argument(..., help="Ticker symbol, e.g. AAPL"),
    qty: float = typer.Argument(..., help="Number of shares (> 0)"),
) -> None:
    """Place a paper market BUY order and record it in the DB."""
    settings = _settings_or_die()
    conn = db.connect(settings.db_path)
    db.migrate(conn)
    try:
        with AlpacaClient(
            settings.alpaca_api_key,
            settings.alpaca_api_secret,
            settings.alpaca_base_url,
        ) as client:
            order = client.submit_order(symbol, qty, "buy")
        db.upsert_order(conn, order)
    finally:
        conn.close()
    console.print(
        f"[green]BUY[/green] {order.qty:g} {order.symbol}: "
        f"order [bold]{order.id}[/bold] status={order.status}"
    )


@app.command()
def sell(
    symbol: str = typer.Argument(..., help="Ticker symbol, e.g. AAPL"),
    qty: float = typer.Argument(..., help="Number of shares to sell (> 0)"),
) -> None:
    """Place a paper market SELL order to close (part of) a long position.

    Refuses if the requested quantity exceeds the position you actually hold.
    This bot never shorts.
    """
    settings = _settings_or_die()
    conn = db.connect(settings.db_path)
    db.migrate(conn)
    try:
        with AlpacaClient(
            settings.alpaca_api_key,
            settings.alpaca_api_secret,
            settings.alpaca_base_url,
        ) as client:
            positions = {p.symbol: p for p in client.get_positions()}
            held = positions.get(symbol.upper())
            held_qty = held.qty if held is not None else 0.0
            if held_qty < qty:
                console.print(
                    f"[red]Refusing to SELL {qty:g} {symbol.upper()}: "
                    f"only {held_qty:g} held. This bot does not short.[/red]"
                )
                raise typer.Exit(code=1)
            order = client.submit_order(symbol, qty, "sell")
        db.upsert_order(conn, order)
    finally:
        conn.close()
    console.print(
        f"[yellow]SELL[/yellow] {order.qty:g} {order.symbol}: "
        f"order [bold]{order.id}[/bold] status={order.status}"
    )


@app.command(name="fetch-bars")
def fetch_bars(
    symbol: str = typer.Argument(..., help="Ticker symbol"),
    start: str = typer.Option(..., "--start", help="ISO date/time, e.g. 2025-01-01"),
    end: str = typer.Option(..., "--end", help="ISO date/time, e.g. 2025-12-31"),
    timeframe: str = typer.Option("1Day", "--timeframe", help="e.g. 1Min, 5Min, 1Day"),
    feed: str = typer.Option("iex", "--feed", help="iex (free) or sip"),
) -> None:
    """Fetch OHLCV bars from Alpaca market data and store them in the DB."""
    settings = _settings_or_die()
    conn = db.connect(settings.db_path)
    db.migrate(conn)
    try:
        with MarketDataClient(
            settings.alpaca_api_key,
            settings.alpaca_api_secret,
            settings.alpaca_data_url,
        ) as client:
            bars = client.get_bars(symbol, timeframe, start, end, feed=feed)
        written = upsert_bars(conn, bars)
    finally:
        conn.close()
    console.print(
        f"Fetched [bold]{len(bars)}[/bold] {timeframe} bars for {symbol.upper()}, "
        f"wrote {written} to DB."
    )


@app.command()
def tick(
    execute: bool = typer.Option(
        False,
        "--execute",
        help="Place paper orders for approved buys. Without this flag, tick is a dry run.",
    ),
    bars_back: int = typer.Option(
        90,
        "--bars-back",
        help="Number of daily bars to pull per symbol for the strategy.",
    ),
) -> None:
    """Run a single pass of the strategy over the watchlist.

    Without --execute this is a dry run: it prints what it would do but
    submits no orders. With --execute, approved BUY signals are placed as
    paper market orders.
    """
    settings = _settings_or_die()
    config = _app_config_or_die()
    strategy = build_strategy(config.strategy.name, config.strategy.params)

    conn = db.connect(settings.db_path)
    db.migrate(conn)
    try:
        with (
            AlpacaClient(
                settings.alpaca_api_key,
                settings.alpaca_api_secret,
                settings.alpaca_base_url,
            ) as broker,
            MarketDataClient(
                settings.alpaca_api_key,
                settings.alpaca_api_secret,
                settings.alpaca_data_url,
            ) as data,
        ):
            result = run_tick(
                broker=broker,
                data=data,
                conn=conn,
                config=config,
                strategy=strategy,
                bars_back=bars_back,
                execute=execute,
            )
    finally:
        conn.close()

    _render_tick_result(result, execute=execute)


def _render_tick_result(result: TickResult, *, execute: bool) -> None:
    table = Table(title="Signals")
    table.add_column("symbol")
    table.add_column("signal")
    table.add_column("price", justify="right")
    table.add_column("reason")
    for s in result.signals:
        color = {"buy": "green", "sell": "yellow", "hold": "dim"}[s.kind]
        table.add_row(
            s.symbol,
            f"[{color}]{s.kind}[/{color}]",
            f"${s.price:,.2f}",
            s.reason,
        )
    console.print(table)

    for order_id in result.placed_order_ids:
        console.print(f"  [green]Placed BUY[/green] order {order_id}")
    for symbol, reason in result.blocked:
        console.print(f"  [red]BLOCKED[/red] {symbol}: {reason}")

    if not execute and not result.placed_order_ids:
        console.print(
            "[dim]Dry run. Re-run with --execute to place approved buys.[/dim]"
        )


def _render_signals(signals: list[Signal]) -> None:
    table = Table(title="Signals")
    table.add_column("symbol")
    table.add_column("signal")
    table.add_column("price", justify="right")
    table.add_column("reason")
    for s in signals:
        color = {"buy": "green", "sell": "yellow", "hold": "dim"}[s.kind]
        table.add_row(s.symbol, f"[{color}]{s.kind}[/{color}]", f"${s.price:,.2f}", s.reason)
    console.print(table)


@app.command()
def backtest(
    start: str = typer.Option(..., "--from", help="Start date, ISO format, e.g. 2024-01-01"),
    end: str = typer.Option(..., "--to", help="End date, ISO format, e.g. 2025-12-31"),
    symbols: str | None = typer.Option(
        None,
        "--symbols",
        help="Comma-separated tickers. Defaults to config.yaml watchlist.",
    ),
    initial_equity: float = typer.Option(
        100_000.0, "--initial-equity", help="Starting cash for the backtest."
    ),
    output: Path | None = typer.Option(
        None, "--output", help="Write the trade log to this CSV path."
    ),
    equity_output: Path | None = typer.Option(
        None, "--equity-output", help="Write the equity curve to this CSV path."
    ),
    fetch: bool = typer.Option(
        False,
        "--fetch",
        help="Fetch missing daily bars from Alpaca before running the backtest.",
    ),
    intraday: bool = typer.Option(
        False,
        "--intraday",
        help="Use the timestamp-stepping engine (required for 5Min/15Min timeframes).",
    ),
) -> None:
    """Backtest the configured strategy on historical bars stored in SQLite."""
    config = _app_config_or_die()
    symbols_tuple = (
        tuple(s.strip().upper() for s in symbols.split(",") if s.strip())
        if symbols
        else config.watchlist
    )

    bt_config = BacktestConfig(
        start=date.fromisoformat(start),
        end=date.fromisoformat(end),
        symbols=symbols_tuple,
        initial_equity=initial_equity,
        risk=config.risk,
        strategy=build_strategy(config.strategy.name, config.strategy.params),
        timeframe=config.strategy.timeframe,
        universe=config.universe,
    )

    settings = _settings_or_die()
    conn = db.connect(settings.db_path)
    db.migrate(conn)
    try:
        if fetch:
            with MarketDataClient(
                settings.alpaca_api_key,
                settings.alpaca_api_secret,
                settings.alpaca_data_url,
            ) as data_client:
                for sym in symbols_tuple:
                    bars = data_client.get_bars(sym, config.strategy.timeframe, start, end)
                    upsert_bars(conn, bars)
                    console.print(
                        f"Fetched [bold]{len(bars)}[/bold] "
                        f"{config.strategy.timeframe} bars for {sym}"
                    )

        runner = run_intraday_backtest if intraday else run_backtest
        try:
            result = runner(bt_config, conn)
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(code=1) from e
    finally:
        conn.close()

    stats = compute_stats(result)
    _render_backtest_stats(stats, symbols_tuple, start, end)

    if output is not None:
        n = write_trades_csv(result.trades, output)
        console.print(f"Wrote {n} trades to {output}")
    if equity_output is not None:
        n = write_equity_curve_csv(result.equity_curve, equity_output)
        console.print(f"Wrote {n} equity points to {equity_output}")


def _render_backtest_stats(
    stats: BacktestStats,
    symbols: tuple[str, ...],
    start: str,
    end: str,
) -> None:
    table = Table(title=f"Backtest {start} → {end}  ({', '.join(symbols)})")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("initial equity", f"${stats.initial_equity:,.2f}")
    table.add_row("final equity", f"${stats.final_equity:,.2f}")
    table.add_row("total return", f"{stats.total_return_pct:+.2f}%")
    table.add_row("max drawdown", f"{stats.max_drawdown_pct:.2f}%")
    table.add_row("trades", str(stats.num_trades))
    table.add_row("buys / sells", f"{stats.num_buys} / {stats.num_sells}")
    table.add_row("closed round-trips", str(stats.closed_round_trips))
    table.add_row("win rate", f"{stats.win_rate:.2%}")
    console.print(table)


@app.command(name="walk-forward")
def walk_forward(
    start: str = typer.Option(..., "--from", help="Overall ISO start date"),
    end: str = typer.Option(..., "--to", help="Overall ISO end date"),
    symbols: str | None = typer.Option(None, "--symbols"),
    initial_equity: float = typer.Option(100_000.0, "--initial-equity"),
    train_days: int = typer.Option(180, "--train-days"),
    test_days: int = typer.Option(30, "--test-days"),
    step_days: int | None = typer.Option(None, "--step-days"),
) -> None:
    """Run a walk-forward backtest: rolling (train, test) windows; report
    stats per test window so a strategy that only works in one regime gets
    exposed."""
    config = _app_config_or_die()
    settings = _settings_or_die()
    symbols_tuple = (
        tuple(s.strip().upper() for s in symbols.split(",") if s.strip())
        if symbols
        else config.watchlist
    )
    base = BacktestConfig(
        start=date.fromisoformat(start),
        end=date.fromisoformat(end),
        symbols=symbols_tuple,
        initial_equity=initial_equity,
        risk=config.risk,
        strategy=build_strategy(config.strategy.name, config.strategy.params),
        timeframe=config.strategy.timeframe,
        universe=config.universe,
    )
    conn = db.connect(settings.db_path)
    db.migrate(conn)
    try:
        windows = run_walk_forward(
            base,
            conn,
            train_days=train_days,
            test_days=test_days,
            step_days=step_days,
        )
    finally:
        conn.close()
    if not windows:
        console.print("[red]No windows produced; check date range vs bars in DB.[/red]")
        raise typer.Exit(code=1)
    table = Table(title=f"Walk-forward {start} → {end}")
    table.add_column("test_start")
    table.add_column("test_end")
    table.add_column("return", justify="right")
    table.add_column("max_dd", justify="right")
    table.add_column("trades", justify="right")
    table.add_column("win_rate", justify="right")
    for w in windows:
        table.add_row(
            w.test_start.isoformat(),
            w.test_end.isoformat(),
            f"{w.stats.total_return_pct:+.2f}%",
            f"{w.stats.max_drawdown_pct:.2f}%",
            str(w.stats.num_trades),
            f"{w.stats.win_rate:.0%}",
        )
    console.print(table)


@app.command()
def signals(
    symbol: str | None = typer.Option(None, "--symbol"),
    limit: int = typer.Option(20, "--limit"),
) -> None:
    """Show recent signal rows from the DB."""
    settings = _settings_or_die()
    conn = db.connect(settings.db_path)
    db.migrate(conn)
    try:
        if symbol:
            rows = conn.execute(
                "SELECT ts, symbol, strategy, signal, score, reason FROM signals "
                "WHERE symbol = ? ORDER BY ts DESC LIMIT ?",
                (symbol.upper(), limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT ts, symbol, strategy, signal, score, reason FROM signals "
                "ORDER BY ts DESC LIMIT ?",
                (limit,),
            ).fetchall()
    finally:
        conn.close()
    if not rows:
        console.print("No signals recorded yet.")
        return
    table = Table(title="signals")
    for col in ("ts", "symbol", "strategy", "signal", "score", "reason"):
        table.add_column(col)
    for r in rows:
        score = r["score"] if r["score"] is not None else 0.0
        table.add_row(
            r["ts"], r["symbol"], r["strategy"], r["signal"],
            f"{score:.4f}", r["reason"] or "",
        )
    console.print(table)


@app.command()
def orders(limit: int = typer.Option(20, "--limit")) -> None:
    """Show recent orders from the DB."""
    settings = _settings_or_die()
    conn = db.connect(settings.db_path)
    db.migrate(conn)
    try:
        rows = conn.execute(
            "SELECT submitted_at, symbol, side, qty, status, filled_avg_price "
            "FROM orders ORDER BY submitted_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        console.print("No orders recorded yet.")
        return
    table = Table(title="orders")
    for col in ("submitted_at", "symbol", "side", "qty", "status", "fill_price"):
        table.add_column(col)
    for r in rows:
        fp = r["filled_avg_price"]
        table.add_row(
            r["submitted_at"], r["symbol"], r["side"],
            f"{r['qty']:g}", r["status"], f"${fp:.2f}" if fp is not None else "—",
        )
    console.print(table)


@app.command(name="equity-log")
def equity_log(limit: int = typer.Option(20, "--limit")) -> None:
    """Show recent equity snapshots."""
    settings = _settings_or_die()
    conn = db.connect(settings.db_path)
    db.migrate(conn)
    try:
        rows = conn.execute(
            "SELECT ts, equity, cash, buying_power FROM equity_snapshots "
            "ORDER BY ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        console.print("No equity snapshots recorded yet.")
        return
    table = Table(title="equity_snapshots")
    for col in ("ts", "equity", "cash", "buying_power"):
        table.add_column(col, justify="right" if col != "ts" else "left")
    for r in rows:
        table.add_row(
            r["ts"], f"${r['equity']:,.2f}", f"${r['cash']:,.2f}",
            f"${r['buying_power']:,.2f}",
        )
    console.print(table)


@app.command(name="compare-strategies")
def compare_strategies(
    start: str = typer.Option(..., "--from"),
    end: str = typer.Option(..., "--to"),
    symbols: str | None = typer.Option(None, "--symbols"),
    initial_equity: float = typer.Option(100_000.0, "--initial-equity"),
    only: str | None = typer.Option(
        None,
        "--only",
        help="Comma-separated subset of strategy names; default = all registered.",
    ),
) -> None:
    """A/B all registered strategies over the same window."""
    from amms.strategy.base import registered_strategies

    config = _app_config_or_die()
    settings = _settings_or_die()
    symbols_tuple = (
        tuple(s.strip().upper() for s in symbols.split(",") if s.strip())
        if symbols
        else config.watchlist
    )
    # Touch build_strategy to lazy-register the built-ins, then read.
    _ = build_strategy("sma_cross", {})
    names = list(registered_strategies())
    if only:
        wanted = {n.strip() for n in only.split(",")}
        names = [n for n in names if n in wanted]
    conn = db.connect(settings.db_path)
    db.migrate(conn)
    rows: list[tuple[str, BacktestStats]] = []
    for name in sorted(names):
        try:
            strat = build_strategy(name, {})
        except Exception as e:
            console.print(f"[red]{name}: construct failed: {e}[/red]")
            continue
        bt = BacktestConfig(
            start=date.fromisoformat(start),
            end=date.fromisoformat(end),
            symbols=symbols_tuple,
            initial_equity=initial_equity,
            risk=config.risk,
            strategy=strat,
            timeframe=config.strategy.timeframe,
            universe=config.universe,
        )
        try:
            result = run_backtest(bt, conn)
        except ValueError as e:
            console.print(f"[red]{name}: {e}[/red]")
            continue
        rows.append((name, compute_stats(result)))
    conn.close()
    if not rows:
        raise typer.Exit(code=1)
    table = Table(title=f"Comparison {start} → {end}")
    table.add_column("strategy")
    table.add_column("return", justify="right")
    table.add_column("max_dd", justify="right")
    table.add_column("trades", justify="right")
    table.add_column("win_rate", justify="right")
    for label, s in rows:
        table.add_row(
            label,
            f"{s.total_return_pct:+.2f}%",
            f"{s.max_drawdown_pct:.2f}%",
            str(s.num_trades),
            f"{s.win_rate:.0%}",
        )
    console.print(table)


@app.command(name="summary-today")
def summary_today(llm: bool = typer.Option(False, "--llm")) -> None:
    """Print the plain daily summary that the bot would send via Telegram."""
    from amms.executor import build_daily_summary

    settings = _settings_or_die()
    conn = db.connect(settings.db_path)
    db.migrate(conn)
    try:
        with AlpacaClient(
            settings.alpaca_api_key,
            settings.alpaca_api_secret,
            settings.alpaca_base_url,
        ) as broker:
            plain = build_daily_summary(broker, conn)
        if llm:
            from datetime import UTC, datetime

            from amms.notifier.llm_summary import augment_summary

            today = datetime.now(UTC).date().isoformat()
            rows = conn.execute(
                "SELECT symbol, side, qty, status, filled_avg_price "
                "FROM orders WHERE substr(submitted_at,1,10) = ?",
                (today,),
            ).fetchall()
            plain = augment_summary(plain, trades_today=[dict(r) for r in rows], conn=conn)
    finally:
        conn.close()
    console.print(plain)


watchlist_app = typer.Typer(help="Inspect / mutate the watchlist in config.yaml.")
app.add_typer(watchlist_app, name="watchlist")


@watchlist_app.command("show")
def watchlist_show() -> None:
    """Print the active watchlist."""
    cfg = _app_config_or_die()
    for sym in cfg.watchlist:
        console.print(sym)


@watchlist_app.command("add")
def watchlist_add(symbol: str = typer.Argument(...)) -> None:
    """Append a symbol to config.yaml's watchlist (uppercased, deduped)."""
    _watchlist_mutate(lambda lst: _dedupe_append(lst, symbol.upper()))


@watchlist_app.command("remove")
def watchlist_remove(symbol: str = typer.Argument(...)) -> None:
    """Remove a symbol from config.yaml's watchlist (case-insensitive)."""
    upper = symbol.upper()
    _watchlist_mutate(lambda lst: [s for s in lst if s.upper() != upper])


def _dedupe_append(lst: list[str], sym: str) -> list[str]:
    seen = {s.upper() for s in lst}
    if sym in seen:
        return lst
    return [*lst, sym]


def _watchlist_mutate(fn) -> None:
    import os

    import yaml

    path = os.environ.get("AMMS_CONFIG_PATH", "").strip() or "config.yaml"
    p = Path(path)
    if not p.exists():
        console.print(f"[red]config not found: {p}[/red]")
        raise typer.Exit(code=1)
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    current = list(raw.get("watchlist") or [])
    new_list = fn(current)
    if new_list == current:
        console.print("watchlist unchanged.")
        return
    raw["watchlist"] = new_list
    p.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    console.print(f"watchlist now {len(new_list)} symbols: {new_list}")
    console.print(
        "[yellow]Note:[/yellow] restart the bot to pick up the change "
        "(`docker compose down && docker compose up -d`)."
    )


@app.command(name="preview-signal")
def preview_signal(
    symbol: str = typer.Argument(...),
    bars_back: int = typer.Option(90, "--bars-back"),
) -> None:
    """Run the configured strategy on cached bars for one symbol. No order."""
    config = _app_config_or_die()
    settings = _settings_or_die()
    strat = build_strategy(config.strategy.name, config.strategy.params)
    conn = db.connect(settings.db_path)
    db.migrate(conn)
    try:
        rows = conn.execute(
            "SELECT ts, open, high, low, close, volume FROM bars "
            "WHERE symbol = ? AND timeframe = ? ORDER BY ts DESC LIMIT ?",
            (symbol.upper(), config.strategy.timeframe, bars_back),
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        console.print(
            f"[red]No cached bars for {symbol.upper()} at "
            f"timeframe {config.strategy.timeframe!r}. "
            "Run `amms fetch-bars` first.[/red]"
        )
        raise typer.Exit(code=1)
    from amms.data.bars import Bar

    bars = list(
        reversed(
            [
                Bar(
                    symbol.upper(),
                    config.strategy.timeframe,
                    r["ts"], r["open"], r["high"], r["low"], r["close"], r["volume"],
                )
                for r in rows
            ]
        )
    )
    signal = strat.evaluate(symbol.upper(), bars)
    color = {"buy": "green", "sell": "yellow", "hold": "dim"}.get(signal.kind, "white")
    console.print(
        f"[{color}]{signal.kind.upper()}[/{color}] {signal.symbol} "
        f"@ ${signal.price:,.2f}  score={signal.score:.4f}\n  {signal.reason}"
    )


@app.command(name="apply-profile")
def apply_profile(
    name: str = typer.Argument(..., help="swing | day-trade | penny | <path>"),
) -> None:
    """Replace config.yaml with one of the bundled profiles (or any path)."""
    import os
    import shutil

    candidate = Path(f"profiles/{name}.yaml")
    if candidate.exists():
        src = candidate
    else:
        src = Path(name)
        if not src.exists():
            console.print(f"[red]profile not found: {name}[/red]")
            raise typer.Exit(code=1)
    dst_env = os.environ.get("AMMS_CONFIG_PATH", "").strip() or "config.yaml"
    dst = Path(dst_env)
    if dst.exists():
        backup = dst.with_suffix(dst.suffix + ".bak")
        shutil.copy2(dst, backup)
        console.print(f"backed up existing config to {backup}")
    shutil.copy2(src, dst)
    console.print(f"[green]applied profile {src.name} → {dst}[/green]")
    console.print(
        "[yellow]Restart to pick up the change:[/yellow] "
        "docker compose down && docker compose up -d"
    )


@app.command(name="order-status")
def order_status(order_id: str = typer.Argument(...)) -> None:
    """Fetch the current status of a paper order from Alpaca."""
    settings = _settings_or_die()
    with AlpacaClient(
        settings.alpaca_api_key,
        settings.alpaca_api_secret,
        settings.alpaca_base_url,
    ) as broker:
        order = broker.get_order(order_id)
    table = Table(title=f"order {order.id}")
    table.add_column("field")
    table.add_column("value")
    fp = order.filled_avg_price
    rows = [
        ("symbol", order.symbol),
        ("side", order.side),
        ("qty", f"{order.qty:g}"),
        ("type", order.type),
        ("status", order.status),
        ("submitted_at", order.submitted_at),
        ("filled_at", order.filled_at or "—"),
        ("fill_price", f"${fp:.2f}" if fp is not None else "—"),
    ]
    for k, v in rows:
        table.add_row(k, str(v))
    console.print(table)


@app.command()
def vacuum() -> None:
    """Run SQLite VACUUM to reclaim space + analyze the schema."""
    settings = _settings_or_die()
    conn = db.connect(settings.db_path)
    try:
        conn.execute("VACUUM")
        conn.execute("ANALYZE")
    finally:
        conn.close()
    console.print(f"[green]VACUUM + ANALYZE done on {settings.db_path}[/green]")


@app.command(name="reset-db")
def reset_db(yes: bool = typer.Option(False, "--yes", help="Skip confirmation.")) -> None:
    """Destroy the SQLite DB and re-run migrations.

    Useful only on a fresh, never-traded paper account. Strongly refuses
    without --yes since this is irreversible.
    """
    settings = _settings_or_die()
    if not yes:
        console.print(
            "[yellow]This will delete the DB file. Re-run with --yes to confirm.[/yellow]"
        )
        raise typer.Exit(code=1)
    try:
        settings.db_path.unlink(missing_ok=True)  # type: ignore[arg-type]
    except Exception as e:
        console.print(f"[red]could not unlink: {e}[/red]")
        raise typer.Exit(code=1) from e
    conn = db.connect(settings.db_path)
    applied = db.migrate(conn)
    conn.close()
    console.print(f"[green]reset done; {applied} migration(s) applied[/green]")


@app.command(name="list-strategies")
def list_strategies() -> None:
    """List registered strategies and their parameters."""
    import contextlib

    from amms.strategy.base import build_strategy, registered_strategies

    # Touch build_strategy to lazy-register the built-ins.
    with contextlib.suppress(Exception):
        build_strategy("sma_cross", {})
    reg = registered_strategies()
    table = Table(title="strategies")
    table.add_column("name")
    table.add_column("class")
    table.add_column("fields")
    import dataclasses

    for name in sorted(reg):
        cls = reg[name]
        try:
            field_names = ", ".join(f.name for f in dataclasses.fields(cls))
        except TypeError:
            field_names = "—"
        table.add_row(name, cls.__name__, field_names)
    console.print(table)


@app.command()
def doctor() -> None:
    """Pre-flight self-check. Validates env, config, DB, and Alpaca reach.

    Run this once after `cp .env.example .env` and `cp config.example.yaml
    config.yaml`. Reports each check as ✓ or ✗ with a clear reason.
    """
    failures = 0

    # 1. Settings
    try:
        settings = load_settings()
        console.print(f"[green]✓[/green] env loaded; broker URL = {settings.alpaca_base_url}")
    except ConfigError as e:
        console.print(f"[red]✗[/red] env: {e}")
        raise typer.Exit(code=1) from e

    # 2. Paper guard
    if "paper-api" in settings.alpaca_base_url:
        console.print("[green]✓[/green] paper endpoint (live-trading guard intact)")
    else:
        console.print(f"[red]✗[/red] paper guard failed: {settings.alpaca_base_url}")
        failures += 1

    # 3. AppConfig
    try:
        cfg = load_app_config()
        console.print(
            f"[green]✓[/green] config.yaml: {len(cfg.watchlist)} symbols, "
            f"strategy={cfg.strategy.name}@{cfg.strategy.timeframe}"
        )
    except ConfigError as e:
        console.print(f"[red]✗[/red] config: {e}")
        failures += 1
        cfg = None

    # 4. DB
    try:
        conn = db.connect(settings.db_path)
        applied = db.migrate(conn)
        tables = {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        conn.close()
        console.print(
            f"[green]✓[/green] DB at {settings.db_path}: "
            f"{applied} migration(s) applied; tables={len(tables)}"
        )
    except Exception as e:
        console.print(f"[red]✗[/red] DB: {e}")
        failures += 1

    # 5. Alpaca reachability (HEAD-equivalent: just /v2/account)
    try:
        with AlpacaClient(
            settings.alpaca_api_key,
            settings.alpaca_api_secret,
            settings.alpaca_base_url,
        ) as broker:
            account = broker.get_account()
        console.print(
            f"[green]✓[/green] Alpaca reachable; "
            f"equity ${account.equity:,.2f}, status {account.status}"
        )
    except Exception as e:
        console.print(f"[red]✗[/red] Alpaca: {e}")
        failures += 1

    # 6. Telegram (optional)
    import os

    if os.environ.get("TELEGRAM_BOT_TOKEN", "").strip() and os.environ.get(
        "TELEGRAM_CHAT_ID", ""
    ).strip():
        console.print("[green]✓[/green] Telegram configured")
    else:
        console.print("[dim]·[/dim] Telegram not configured (optional)")

    # 7. Sanity: strategy timeframe vs backtester limitation
    if cfg is not None and cfg.strategy.timeframe != "1Day":
        console.print(
            f"[yellow]⚠[/yellow] strategy.timeframe={cfg.strategy.timeframe} — "
            "default backtester groups bars by day. Use `amms backtest --intraday` "
            "or run_intraday_backtest for proper intraday simulation."
        )

    # 8. Stale bars
    if cfg is not None:
        try:
            from datetime import UTC, datetime

            conn2 = db.connect(settings.db_path)
            row = conn2.execute(
                "SELECT max(ts) FROM bars WHERE timeframe = ?",
                (cfg.strategy.timeframe,),
            ).fetchone()
            conn2.close()
            latest = row[0] if row else None
            if latest:
                latest_dt = datetime.fromisoformat(latest.replace("Z", "+00:00"))
                age_days = (datetime.now(UTC) - latest_dt).days
                if age_days <= 2:
                    console.print(f"[green]✓[/green] bars fresh (newest {latest})")
                else:
                    console.print(
                        f"[yellow]⚠[/yellow] newest bar is {age_days}d old "
                        f"({latest}); run `amms fetch-bars` per symbol."
                    )
            else:
                console.print(
                    "[dim]·[/dim] no bars cached yet; first `amms tick` will fetch."
                )
        except Exception as e:
            console.print(f"[yellow]⚠[/yellow] bars-age check failed: {e}")

    if failures:
        console.print(f"\n[red]doctor: {failures} failure(s).[/red]")
        raise typer.Exit(code=1)
    console.print("\n[green]doctor: all good.[/green]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
