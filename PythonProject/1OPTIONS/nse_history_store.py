#!/usr/bin/env python3
"""
nse_history_store.py
======================

Simple, dependency-free persistence for session snapshots (spot, PCR, ATM
IV, max pain, support/resistance walls) so trends survive a browser reload
or a server restart — unlike the dashboard's in-memory-only session
tracking, which resets every time you close the tab.

Storage format: one JSON-lines file per (symbol, date), e.g.
    history/NIFTY_2026-06-20.jsonl
Each line is one snapshot: {"t": unix_ts, "spot":..., "pcr":..., "atm_iv":...,
"max_pain":..., "support":..., "resistance":...}

Why JSONL + daily files instead of a database: zero dependencies, trivially
inspectable/grep-able, natural rotation (old files can just be deleted),
and appends are O(1) — appropriate for this volume (a few thousand points
per symbol per day at most, even at a 5s refresh cadence).
"""

from __future__ import annotations

import json
import time
from datetime import datetime, date
from pathlib import Path

HISTORY_DIR = Path(__file__).parent / "history"
MAX_DAYS_RETAINED = 30  # prune files older than this on each write, keeps disk bounded


def _file_for(symbol: str, day: date) -> Path:
    HISTORY_DIR.mkdir(exist_ok=True)
    return HISTORY_DIR / f"{symbol.upper()}_{day.isoformat()}.jsonl"


def append_snapshot(symbol: str, snapshot: dict) -> None:
    """Append one snapshot for today. Silently no-ops on write failure —
    history tracking should never be allowed to break the live dashboard."""
    try:
        today = date.today()
        path = _file_for(symbol, today)
        record = {"t": time.time(), **snapshot}
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        _prune_old_files(symbol)
    except Exception as e:  # noqa: BLE001 - history is best-effort, never fatal
        print(f"[!] History write failed for {symbol}: {e}")


def _prune_old_files(symbol: str) -> None:
    try:
        cutoff = time.time() - MAX_DAYS_RETAINED * 86400
        for f in HISTORY_DIR.glob(f"{symbol.upper()}_*.jsonl"):
            if f.stat().st_mtime < cutoff:
                f.unlink()
    except Exception:
        pass  # pruning is housekeeping, not critical


def read_history(symbol: str, days: int = 1) -> list[dict]:
    """Returns snapshots from the last `days` calendar days (including
    today), oldest first. Returns [] if nothing's been recorded yet."""
    if not HISTORY_DIR.exists():
        return []
    records = []
    files = sorted(HISTORY_DIR.glob(f"{symbol.upper()}_*.jsonl"))[-days:]
    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue  # skip a corrupted line rather than failing the whole read
        except Exception as e:
            print(f"[!] History read failed for {path}: {e}")
    records.sort(key=lambda r: r.get("t", 0))
    return records


def downsample(records: list[dict], max_points: int = 500) -> list[dict]:
    """Thin out a long history series for charting — keeps it visually
    smooth without shipping thousands of points to the browser."""
    if len(records) <= max_points:
        return records
    step = len(records) / max_points
    return [records[int(i * step)] for i in range(max_points)]
