from typer.testing import CliRunner

from amms import __version__
from amms.cli import app

runner = CliRunner()


def test_help_works() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "paper trading" in result.stdout.lower()


def test_run_prints_ready_banner() -> None:
    result = runner.invoke(app, ["run"])
    assert result.exit_code == 0
    assert f"amms {__version__} ready" in result.stdout


def test_status_not_implemented_yet() -> None:
    result = runner.invoke(app, ["status"])
    assert result.exit_code != 0
    assert isinstance(result.exception, NotImplementedError)


def test_backtest_not_implemented_yet() -> None:
    result = runner.invoke(app, ["backtest"])
    assert result.exit_code != 0
    assert isinstance(result.exception, NotImplementedError)
