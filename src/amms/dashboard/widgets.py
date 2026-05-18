"""Widget registry. Each widget is a small self-contained dashboard tile.

A widget definition declares metadata (title, description, default size).
The actual rendering lives in `templates/widgets/<type>.html`. Adding a new
widget means: add an entry here + add a template file.
"""

from __future__ import annotations

from dataclasses import dataclass

SIZES = ("sm", "md", "lg")


@dataclass(frozen=True)
class WidgetDef:
    type: str
    title: str
    description: str
    default_size: str = "md"


WIDGET_REGISTRY: dict[str, WidgetDef] = {
    "equity": WidgetDef(
        type="equity",
        title="Gesamtvermögen",
        description="Aktueller Portfoliowert (Equity).",
        default_size="md",
    ),
    "cash": WidgetDef(
        type="cash",
        title="Cash",
        description="Verfügbares Bargeld auf dem Paper-Trading-Konto.",
        default_size="sm",
    ),
    "buying_power": WidgetDef(
        type="buying_power",
        title="Kaufkraft",
        description="Maximale Kaufsumme inkl. Margin (paper only).",
        default_size="sm",
    ),
    "day_change": WidgetDef(
        type="day_change",
        title="Tagesveränderung",
        description="Heutige Performance in Euro/USD und Prozent.",
        default_size="md",
    ),
    "positions_count": WidgetDef(
        type="positions_count",
        title="Offene Positionen",
        description="Anzahl der aktuell gehaltenen Positionen.",
        default_size="sm",
    ),
    "unrealized_pl": WidgetDef(
        type="unrealized_pl",
        title="Unrealisierter Gewinn",
        description="Summe aller noch nicht realisierten Gewinne/Verluste.",
        default_size="md",
    ),
    "top_winner": WidgetDef(
        type="top_winner",
        title="Bestes Asset",
        description="Position mit größtem unrealisierten Gewinn.",
        default_size="sm",
    ),
    "top_loser": WidgetDef(
        type="top_loser",
        title="Schwächstes Asset",
        description="Position mit größtem unrealisierten Verlust.",
        default_size="sm",
    ),
    "equity_sparkline": WidgetDef(
        type="equity_sparkline",
        title="Equity-Verlauf",
        description="Chart über 30 Tage mit Achsen und Umschalter (Wert / Δ % / Δ $).",
        default_size="lg",
    ),
    "positions_table": WidgetDef(
        type="positions_table",
        title="Positionen",
        description="Tabelle aller offenen Positionen mit Symbol, Stück, P/L.",
        default_size="lg",
    ),
    "index_sp500": WidgetDef(
        type="index_sp500",
        title="S&P 500",
        description="US-Leitindex (500 größte börsennotierte Unternehmen).",
        default_size="md",
    ),
    "index_nasdaq": WidgetDef(
        type="index_nasdaq",
        title="Nasdaq",
        description="US-Tech-lastiger Composite-Index.",
        default_size="md",
    ),
    "index_dow": WidgetDef(
        type="index_dow",
        title="Dow Jones",
        description="30 große US-Industriewerte.",
        default_size="md",
    ),
    "index_dax": WidgetDef(
        type="index_dax",
        title="DAX",
        description="Deutscher Leitindex (40 größte Unternehmen).",
        default_size="md",
    ),
    "index_ftse": WidgetDef(
        type="index_ftse",
        title="FTSE 100",
        description="UK-Leitindex.",
        default_size="md",
    ),
    "index_nikkei": WidgetDef(
        type="index_nikkei",
        title="Nikkei 225",
        description="Japans Leitindex.",
        default_size="md",
    ),
    "index_vix": WidgetDef(
        type="index_vix",
        title="VIX",
        description="Volatilitätsindex — 'Angstbarometer' des Marktes.",
        default_size="md",
    ),
}


def is_known(widget_type: str) -> bool:
    return widget_type in WIDGET_REGISTRY


def get(widget_type: str) -> WidgetDef:
    return WIDGET_REGISTRY[widget_type]
