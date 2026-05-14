from __future__ import annotations

from amms.data.bars import Bar


def relative_volume(bars: list[Bar], n: int = 20) -> float | None:
    """Last bar's volume divided by the mean of the prior ``n`` bars' volume.

    A value > 1 means today's volume is higher than the recent average. Returns
    None if there's not enough history or the prior average is zero.
    """
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    if len(bars) < n + 1:
        return None
    last_volume = bars[-1].volume
    prior_volumes = [b.volume for b in bars[-(n + 1) : -1]]
    avg = sum(prior_volumes) / len(prior_volumes)
    if avg <= 0:
        return None
    return last_volume / avg
