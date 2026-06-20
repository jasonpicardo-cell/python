#!/usr/bin/env python3
"""
nse_alerts.py
==============

Telegram alerting for meaningful changes in option-chain positioning: PCR
crossing into extreme territory, the support/resistance OI wall shifting
strikes, max pain moving, or India VIX spiking. Fires server-side, so it
works even if the dashboard tab isn't open.

WHAT THIS DOES NOT DO
----------------------
- WhatsApp: there's no simple, free, self-serve API for sending WhatsApp
  messages the way Telegram's Bot API works. WhatsApp Business API requires
  a registered business, a approved message template, and typically a
  paid provider (Twilio, Gupshup, etc.) — that's a real account-setup step
  on your end, not something I can wire up unilaterally. If you want this,
  the cleanest path is usually a Twilio WhatsApp sandbox/number; happy to
  wire up the integration once you have credentials, same pattern as below.
- This does NOT replicate the Composite Signal / Decision Levels / 6
  Reversal Signals logic from the dashboard (that's all client-side JS).
  Porting that here is possible if you want server-side alerts on those
  specifically — flagged as a clear follow-up, not done in this pass.

SETUP
-----
1. Message @BotFather on Telegram, run /newbot, copy the token it gives you.
2. Message your new bot anything once (so it can message you back), then
   visit https://api.telegram.org/bot<TOKEN>/getUpdates and find your
   "chat":{"id": ...} — that's your chat_id.
3. Create alert_config.json next to this file (see ALERT_CONFIG_TEMPLATE
   below for the shape), or set env vars NSE_TELEGRAM_TOKEN / NSE_TELEGRAM_CHAT_ID.
4. Alerts are OFF by default until configured — nothing fires silently.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

import requests

CONFIG_PATH = Path(__file__).parent / "alert_config.json"

ALERT_CONFIG_TEMPLATE = {
    "enabled": False,
    "telegram_token": "",
    "telegram_chat_id": "",
    "pcr_extreme_high": 1.3,
    "pcr_extreme_low": 0.7,
    "vix_spike_pct": 10.0,
    "min_seconds_between_alerts": 120,
}

# in-memory state: last-alerted values per symbol, so we only alert on
# CHANGES crossing a threshold, not every single refresh
_last_state: dict[str, dict] = {}
_last_alert_time: dict[str, float] = {}


def load_config() -> dict:
    cfg = dict(ALERT_CONFIG_TEMPLATE)
    if CONFIG_PATH.exists():
        try:
            cfg.update(json.loads(CONFIG_PATH.read_text()))
        except Exception as e:
            print(f"[!] Couldn't parse alert_config.json: {e}")
    # env vars override file config, handy for not committing a token to disk
    if os.environ.get("NSE_TELEGRAM_TOKEN"):
        cfg["telegram_token"] = os.environ["NSE_TELEGRAM_TOKEN"]
    if os.environ.get("NSE_TELEGRAM_CHAT_ID"):
        cfg["telegram_chat_id"] = os.environ["NSE_TELEGRAM_CHAT_ID"]
    return cfg


def ensure_config_file_exists() -> None:
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(json.dumps(ALERT_CONFIG_TEMPLATE, indent=2))
        print(f"[i] Wrote a template alert_config.json to {CONFIG_PATH} — fill in your Telegram token/chat_id and set enabled:true to turn on alerts.")


def send_telegram_message(token: str, chat_id: str, text: str) -> bool:
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
        if resp.status_code != 200:
            print(f"[!] Telegram send failed: HTTP {resp.status_code} {resp.text[:200]}")
            return False
        return True
    except Exception as e:
        print(f"[!] Telegram send error: {e}")
        return False


def check_and_alert(symbol: str, snapshot: dict, cfg: Optional[dict] = None) -> list[str]:
    """Compares the new snapshot against the last-seen one for this symbol
    and fires Telegram alerts for meaningful changes. Returns the list of
    alert messages sent (for logging/testing) — empty list if nothing fired
    or alerts are disabled."""
    cfg = cfg or load_config()
    if not cfg.get("enabled") or not cfg.get("telegram_token") or not cfg.get("telegram_chat_id"):
        return []

    now = time.time()
    min_gap = cfg.get("min_seconds_between_alerts", 120)
    if now - _last_alert_time.get(symbol, 0) < min_gap:
        return []

    prev = _last_state.get(symbol)
    _last_state[symbol] = dict(snapshot)
    if prev is None:
        return []  # nothing to compare against yet

    messages = []

    pcr = snapshot.get("pcr")
    prev_pcr = prev.get("pcr")
    if pcr is not None and prev_pcr is not None:
        if pcr >= cfg["pcr_extreme_high"] and prev_pcr < cfg["pcr_extreme_high"]:
            messages.append(f"⚠️ <b>{symbol}</b> PCR crossed into extreme-bullish-positioning territory: {pcr}")
        elif pcr <= cfg["pcr_extreme_low"] and prev_pcr > cfg["pcr_extreme_low"]:
            messages.append(f"⚠️ <b>{symbol}</b> PCR crossed into extreme-bearish-positioning territory: {pcr}")

    if snapshot.get("support") != prev.get("support"):
        messages.append(f"🟢 <b>{symbol}</b> support wall shifted: {prev.get('support')} → {snapshot.get('support')}")
    if snapshot.get("resistance") != prev.get("resistance"):
        messages.append(f"🔴 <b>{symbol}</b> resistance wall shifted: {prev.get('resistance')} → {snapshot.get('resistance')}")

    max_pain, prev_max_pain = snapshot.get("max_pain"), prev.get("max_pain")
    if max_pain is not None and prev_max_pain is not None and max_pain != prev_max_pain:
        messages.append(f"🎯 <b>{symbol}</b> max pain shifted: {prev_max_pain} → {max_pain}")

    vix, prev_vix = snapshot.get("india_vix"), prev.get("india_vix")
    if vix is not None and prev_vix and prev_vix > 0:
        change_pct = (vix - prev_vix) / prev_vix * 100
        if abs(change_pct) >= cfg.get("vix_spike_pct", 10.0):
            direction = "spiked" if change_pct > 0 else "dropped"
            messages.append(f"⚡ India VIX {direction} {abs(change_pct):.1f}%: {prev_vix} → {vix}")

    if messages:
        full_text = "\n".join(messages)
        if send_telegram_message(cfg["telegram_token"], cfg["telegram_chat_id"], full_text):
            _last_alert_time[symbol] = now
    return messages


if __name__ == "__main__":
    ensure_config_file_exists()
    cfg = load_config()
    print("Current config:", {**cfg, "telegram_token": "***" if cfg["telegram_token"] else ""})
    if cfg["enabled"] and cfg["telegram_token"] and cfg["telegram_chat_id"]:
        ok = send_telegram_message(cfg["telegram_token"], cfg["telegram_chat_id"], "✅ NSE dashboard alert test message")
        print("Test message sent:" , ok)
    else:
        print("Alerts not configured/enabled — edit alert_config.json to set this up.")
