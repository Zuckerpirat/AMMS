from __future__ import annotations

from datetime import UTC, datetime, timedelta

import typer
from rich.console import Console
from rich.table import Table

from amms import __version__, db
from amms.broker import AlpacaClient
from amms.config import ConfigError, load_app_config, load_settings
from amms.data import MarketDataClient, upsert_bars
from amms.risk import check_buy
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


@app.command()
def run() -> None:
    """Start the trading loop. Phase 0 stub: prints a readiness banner and exits."""
    typer.echo(f"amms {__version__} ready")


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
    """Run one pass of the strategy over the watchlist.

    Without --execute this is a dry run: it prints what it would do but
    submits no orders. With --execute, approved BUY signals are placed as
    paper market orders.
    """
    settings = _settings_or_die()
    try:
        app_config = load_app_config()
    except ConfigError as e:
        console.print(f"[red]Configuration error:[/red] {e}")
        raise typer.Exit(code=2) from e

    strategy = build_strategy(app_config.strategy.name, app_config.strategy.params)

    conn = db.connect(settings.db_path)
    db.migrate(conn)
    try:
        end = datetime.now(UTC)
        start = end - timedelta(days=bars_back * 2)  # weekends/holidays cushion
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
            account = broker.get_account()
            positions = {p.symbol: p for p in broker.get_positions()}

            signals: list[Signal] = []
            for symbol in app_config.watchlist:
                bars = data.get_bars(
                    symbol,
                    "1Day",
                    start.date().isoformat(),
                    end.date().isoformat(),
                    limit=bars_back,
                )
                upsert_bars(conn, bars)
                closes = [b.close for b in bars]
                signal = strategy.evaluate(symbol, closes)
                signals.append(signal)
                _record_signal(conn, strategy.name, signal)

            _render_signals(signals)

            buys = [s for s in signals if s.kind == "buy"]
            if not buys:
                console.print("No buy signals this tick.")
                return

            for signal in buys:
                decision = check_buy(
                    equity=account.equity,
                    price=signal.price,
                    cash=account.cash,
                    open_positions=len(positions),
                    daily_pnl_pct=0.0,
                    already_holds=signal.symbol in positions,
                    config=app_config.risk,
                )
                tag = "[green]ALLOWED[/green]" if decision.allowed else "[red]BLOCKED[/red]"
                console.print(
                    f"{tag} {signal.symbol}: {decision.reason}"
                )
                if execute and decision.allowed:
                    order = broker.submit_order(signal.symbol, decision.qty, "buy")
                    db.upsert_order(conn, order)
                    positions[signal.symbol] = order  # type: ignore[assignment]
                    console.print(
                        f"  [green]Placed BUY[/green] {decision.qty} {signal.symbol} "
                        f"(order {order.id}, status={order.status})"
                    )

            if not execute:
                console.print(
                    "[dim]Dry run. Re-run with --execute to place approved buys.[/dim]"
                )
    finally:
        conn.close()


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


def _record_signal(conn, strategy_name: str, signal: Signal) -> None:
    ts = datetime.now(UTC).isoformat()
    conn.execute(
        """
        INSERT INTO signals(ts, symbol, strategy, signal, reason)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(ts, symbol, strategy) DO UPDATE SET
            signal = excluded.signal,
            reason = excluded.reason
        """,
        (ts, signal.symbol, strategy_name, signal.kind, signal.reason),
    )


@app.command()
def backtest() -> None:
    """Run a historical backtest against the configured strategy."""
    raise NotImplementedError("backtest is implemented in Phase 3.")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
