#!/usr/bin/env python3
"""
nse_drafts.py
==============

Saved Builder drafts — named snapshots of a leg configuration you can
reload later. Distinct from paper trading (nse_paper_trades.py): a draft
is just "what I was building," not a committed position with live P&L
tracking or win/loss stats. No lifecycle (open/close), just save/load/delete.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

DRAFTS_FILE = Path(__file__).parent / "drafts.json"


def load_drafts() -> list[dict]:
    if not DRAFTS_FILE.exists():
        return []
    try:
        return json.loads(DRAFTS_FILE.read_text())
    except Exception as e:
        print(f"[!] Couldn't read drafts.json ({e}) — starting fresh, old file left untouched for inspection")
        return []


def save_drafts(drafts: list[dict]) -> None:
    DRAFTS_FILE.write_text(json.dumps(drafts, indent=2))


def save_draft(name: str, symbol: str, legs: list[dict]) -> dict:
    drafts = load_drafts()
    draft = {
        "id": uuid.uuid4().hex[:8],
        "name": name,
        "symbol": symbol,
        "legs": legs,
        "saved_at": time.time(),
    }
    drafts.append(draft)
    save_drafts(drafts)
    return draft


def delete_draft(draft_id: str) -> bool:
    drafts = load_drafts()
    new_drafts = [d for d in drafts if d["id"] != draft_id]
    if len(new_drafts) == len(drafts):
        return False
    save_drafts(new_drafts)
    return True


def get_drafts() -> list[dict]:
    drafts = load_drafts()
    drafts.sort(key=lambda d: d["saved_at"], reverse=True)
    return drafts
