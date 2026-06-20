#!/usr/bin/env python3
"""
nse_paper_trades.py
=====================

Forward/paper-trading persistence. Lets you "open" a strategy built in the
dashboard and track its REAL day-by-day P&L going forward using live data
— not a backtest (no historical data involved), not a simulation (no
synthetic price paths) — just an honest journal of "I opened this, here's
what it's actually worth right now, here's what happened when I closed it."

This is intentionally simple: one JSON file, no database. A retail trader
opens a handful of positions at a time, not thousands — a flat file is
the right amount of infrastructure here.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

TRADES_FILE = Path(__file__).parent / "paper_trades.json"


def load_trades() -> list[dict]:
    if not TRADES_FILE.exists():
        return []
    try:
        return json.loads(TRADES_FILE.read_text())
    except Exception as e:
        print(f"[!] Couldn't read paper_trades.json ({e}) — starting fresh, old file left untouched for inspection")
        return []


def save_trades(trades: list[dict]) -> None:
    TRADES_FILE.write_text(json.dumps(trades, indent=2))


def open_trade(symbol: str, name: str, legs: list[dict], lot_size: int) -> dict:
    trades = load_trades()
    entry_net_premium = sum(
        (leg["premium"] if leg["action"] == "SELL" else -leg["premium"]) * leg["qty_lots"]
        for leg in legs
    ) * lot_size
    trade = {
        "id": uuid.uuid4().hex[:8],
        "symbol": symbol,
        "name": name,
        "legs": legs,
        "lot_size": lot_size,
        "opened_at": time.time(),
        "status": "open",
        "entry_net_premium": round(entry_net_premium, 2),
        "closed_at": None,
        "exit_pnl": None,
        "exit_reason": None,
    }
    trades.append(trade)
    save_trades(trades)
    return trade


def close_trade(trade_id: str, exit_pnl: float, reason: str = "manual") -> dict | None:
    trades = load_trades()
    for t in trades:
        if t["id"] == trade_id and t["status"] == "open":
            t["status"] = "closed"
            t["closed_at"] = time.time()
            t["exit_pnl"] = round(exit_pnl, 2)
            t["exit_reason"] = reason
            save_trades(trades)
            return t
    return None


def delete_trade(trade_id: str) -> bool:
    """For removing a mistakenly-opened trade — distinct from closing,
    doesn't count toward win/loss stats."""
    trades = load_trades()
    new_trades = [t for t in trades if t["id"] != trade_id]
    if len(new_trades) == len(trades):
        return False
    save_trades(new_trades)
    return True


def get_trades(status: str = "all") -> list[dict]:
    trades = load_trades()
    if status == "all":
        return trades
    return [t for t in trades if t["status"] == status]


def summary_stats(trades: list[dict]) -> dict:
    closed = [t for t in trades if t["status"] == "closed"]
    if not closed:
        return {"count": 0, "win_rate": None, "total_pnl": 0.0, "avg_win": None, "avg_loss": None}
    wins = [t["exit_pnl"] for t in closed if t["exit_pnl"] > 0]
    losses = [t["exit_pnl"] for t in closed if t["exit_pnl"] <= 0]
    return {
        "count": len(closed),
        "win_rate": round(len(wins) / len(closed) * 100, 1),
        "total_pnl": round(sum(t["exit_pnl"] for t in closed), 2),
        "avg_win": round(sum(wins) / len(wins), 2) if wins else None,
        "avg_loss": round(sum(losses) / len(losses), 2) if losses else None,
    }
