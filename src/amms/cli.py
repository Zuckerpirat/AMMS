from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from amms import __version__, db
from amms.broker import AlpacaClient
from amms.config import ConfigError, load_settings
from amms.data import MarketDataClient, upsert_bars

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
def backtest() -> None:
    """Run a historical backtest against the configured strategy."""
    raise NotImplementedError("backtest is implemented in Phase 3.")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
