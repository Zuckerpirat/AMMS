"""Behavioral Loss Aversion and Disposition Effect Analyzer.

Measures two classic behavioral biases in trading:

1. **Disposition Effect** (Shefrin & Statman, 1985):
   Tendency to sell winners too early and hold losers too long.
   Measured by comparing median hold time of winning vs. losing trades.

   Hold Ratio = median_hold_loser / median_hold_winner
   > 1 → holding losers longer (disposition effect present)
   ≈ 1 → balanced
   < 1 → cutting losers quickly (stop-loss discipline)

2. **Loss Aversion Ratio**:
   Measured by comparing avg |loss| to avg win.
   Standard loss aversion theory: losses feel ~2× more painful than gains.

   Loss Multiplier = avg |loss_pct| / avg win_pct
   > 1 → losses larger than wins (insufficient risk management)
   < 1 → wins larger than losses (positive expectancy setup)

3. **Premature Exit Rate**:
   % of winning trades closed before reaching average win level.
   High rate → cutting winners too early.

4. **Max Adverse Excursion proxy**:
   For losing trades, how much did the loss grow vs the entry?
   Identifies trades held through deep drawdowns.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LossAversionReport:
    # Disposition effect
    median_hold_winner_min: float    # minutes
    median_hold_loser_min: float     # minutes
    hold_ratio: float                # loser / winner median hold
    disposition_effect: bool         # True if hold_ratio > 1.2
    disposition_strength: str        # "strong" / "moderate" / "mild" / "none"

    # Loss aversion ratio
    avg_win_pct: float
    avg_loss_pct: float              # positive value
    loss_multiplier: float           # avg_loss / avg_win
    loss_aversion_level: str         # "excessive" / "high" / "normal" / "low"

    # Premature exit
    premature_exit_rate: float       # % of winners closed below avg win
    avg_winning_trade_pct: float

    # Counts
    n_winners: int
    n_losers: int
    n_trades: int

    verdict: str


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 == 1 else (s[mid - 1] + s[mid]) / 2.0


def _parse_duration_minutes(entered_at: str, closed_at: str) -> float | None:
    """Approximate duration in minutes between two ISO timestamps."""
    try:
        def _to_min(ts: str) -> float:
            t = str(ts).replace("T", " ").strip()
            parts = t.split(" ")
            date_p = parts[0].split("-")
            time_p = parts[1].split(":") if len(parts) > 1 else ["0", "0"]
            y, mo, d = int(date_p[0]), int(date_p[1]), int(date_p[2])
            h, mi = int(time_p[0]), int(time_p[1])
            return ((y - 2000) * 365 + (mo - 1) * 30 + d) * 1440.0 + h * 60.0 + mi

        return max(0.0, _to_min(closed_at) - _to_min(entered_at))
    except Exception:
        return None


def compute(conn, *, limit: int = 500) -> LossAversionReport | None:
    """Analyse loss aversion and disposition effect from closed trade history.

    conn: SQLite connection.
    limit: max trades to analyse.
    """
    try:
        rows = conn.execute("""
            SELECT pnl_pct, entered_at, closed_at
            FROM trades
            WHERE status = 'closed'
              AND pnl_pct IS NOT NULL
              AND entered_at IS NOT NULL
              AND closed_at IS NOT NULL
            ORDER BY closed_at DESC
            LIMIT ?
        """, (limit,)).fetchall()
    except Exception:
        return None

    if len(rows) < 5:
        return None

    win_holds: list[float] = []
    loss_holds: list[float] = []
    win_pcts: list[float] = []
    loss_pcts: list[float] = []  # absolute values

    for pnl, entered, closed in rows:
        pnl = float(pnl)
        dur = _parse_duration_minutes(str(entered), str(closed))
        if dur is None:
            continue
        if pnl > 0:
            win_holds.append(dur)
            win_pcts.append(pnl)
        elif pnl < 0:
            loss_holds.append(dur)
            loss_pcts.append(abs(pnl))

    if not win_holds or not loss_holds:
        return None

    n_winners = len(win_holds)
    n_losers = len(loss_holds)
    n_trades = n_winners + n_losers

    med_win = _median(win_holds)
    med_loss = _median(loss_holds)
    hold_ratio = med_loss / med_win if med_win > 0 else 1.0

    # Disposition effect classification
    if hold_ratio > 2.0:
        disp_strength = "strong"
    elif hold_ratio > 1.4:
        disp_strength = "moderate"
    elif hold_ratio > 1.2:
        disp_strength = "mild"
    else:
        disp_strength = "none"
    disp_effect = hold_ratio > 1.2

    # Loss aversion ratio
    avg_win = sum(win_pcts) / len(win_pcts)
    avg_loss = sum(loss_pcts) / len(loss_pcts)
    loss_mult = avg_loss / avg_win if avg_win > 0 else 1.0

    if loss_mult > 2.5:
        la_level = "excessive"
    elif loss_mult > 1.5:
        la_level = "high"
    elif loss_mult > 0.8:
        la_level = "normal"
    else:
        la_level = "low"  # wins much larger than losses — good!

    # Premature exit: winners closed below average win
    premature = sum(1 for p in win_pcts if p < avg_win) / len(win_pcts) * 100.0

    # Verdict
    notes = []
    if disp_strength in ("strong", "moderate"):
        notes.append(
            f"disposition effect ({disp_strength}): holding losers "
            f"{hold_ratio:.1f}× longer than winners"
        )
    if la_level in ("excessive", "high"):
        notes.append(
            f"loss aversion ({la_level}): avg loss {avg_loss:.2f}% "
            f"vs avg win {avg_win:.2f}% ({loss_mult:.1f}×)"
        )
    if premature > 60:
        notes.append(f"premature exits: {premature:.0f}% of winners closed below avg")

    if notes:
        verdict = "Behavioural biases detected — " + "; ".join(notes) + "."
    else:
        verdict = (
            f"No significant behavioural biases. "
            f"Hold ratio {hold_ratio:.2f}, loss mult {loss_mult:.2f}×."
        )

    return LossAversionReport(
        median_hold_winner_min=round(med_win, 1),
        median_hold_loser_min=round(med_loss, 1),
        hold_ratio=round(hold_ratio, 3),
        disposition_effect=disp_effect,
        disposition_strength=disp_strength,
        avg_win_pct=round(avg_win, 3),
        avg_loss_pct=round(avg_loss, 3),
        loss_multiplier=round(loss_mult, 3),
        loss_aversion_level=la_level,
        premature_exit_rate=round(premature, 1),
        avg_winning_trade_pct=round(avg_win, 3),
        n_winners=n_winners,
        n_losers=n_losers,
        n_trades=n_trades,
        verdict=verdict,
    )
