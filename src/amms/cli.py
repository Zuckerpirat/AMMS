from __future__ import annotations

import typer

from amms import __version__

app = typer.Typer(
    help="AI-assisted paper trading bot for US equities (paper-only, long-only).",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def run() -> None:
    """Start the trading loop. Phase 0 stub: prints a readiness banner and exits."""
    typer.echo(f"amms {__version__} ready")


@app.command()
def status() -> None:
    """Print account equity, open positions, and today's P&L."""
    raise NotImplementedError("status is implemented in Phase 1.")


@app.command()
def backtest() -> None:
    """Run a historical backtest against the configured strategy."""
    raise NotImplementedError("backtest is implemented in Phase 3.")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
