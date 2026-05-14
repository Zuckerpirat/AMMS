from amms.data.bars import Bar
from amms.features.momentum import n_day_return, rsi
from amms.features.volatility import atr, realized_vol
from amms.features.volume import relative_volume

__all__ = [
    "atr",
    "n_day_return",
    "realized_vol",
    "relative_volume",
    "rsi",
    "standard_features",
]


def standard_features(bars: list[Bar]) -> dict[str, float]:
    """Compute the default feature snapshot used by the bot.

    Returns only features for which there's enough history. Caller can persist
    the result so we have an audit trail of what the bot saw each tick.
    """
    out: dict[str, float] = {}
    m20 = n_day_return(bars, 20)
    if m20 is not None:
        out["momentum_20d"] = m20
    r14 = rsi(bars, 14)
    if r14 is not None:
        out["rsi_14"] = r14
    a14 = atr(bars, 14)
    if a14 is not None:
        out["atr_14"] = a14
    v20 = realized_vol(bars, 20)
    if v20 is not None:
        out["realized_vol_20d"] = v20
    rv20 = relative_volume(bars, 20)
    if rv20 is not None:
        out["rvol_20"] = rv20
    return out
