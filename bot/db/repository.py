from bot.db.models import get_connection


def insert_trade(
    symbol: str, side: str, qty: int, price: float, order_id: str, reason: str
) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO trades (symbol, side, qty, price, order_id, reason)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (symbol, side, qty, price, order_id, reason),
        )
        return cur.lastrowid


def update_trade_status(trade_id: int, status: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE trades SET status = ? WHERE id = ?", (status, trade_id)
        )


def get_recent_trades(limit: int = 50) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM trades ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def insert_snapshot(equity: float, cash: float, positions_count: int) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO portfolio_snapshots (equity, cash, positions_count)"
            " VALUES (?, ?, ?)",
            (equity, cash, positions_count),
        )


def get_snapshots(limit: int = 30) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM portfolio_snapshots ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
