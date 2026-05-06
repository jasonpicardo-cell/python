"""
Notifier — Telegram & WhatsApp alerts for the NSE Institutional Scanner
════════════════════════════════════════════════════════════════════════

TELEGRAM SETUP (free, takes 2 minutes)
───────────────────────────────────────
1. Open Telegram → search @BotFather → /newbot
2. Copy the token it gives you
3. Open your new bot → send any message
4. Visit: https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
5. Copy the "id" from result.message.chat
6. Paste both into config.json:
      "telegram_token":   "7812345678:AAF...",
      "telegram_chat_id": "123456789"

WHATSAPP SETUP (via Twilio sandbox — free trial)
──────────────────────────────────────────────────
1. Sign up at twilio.com (free) → Console → Messaging → WhatsApp sandbox
2. Send "join <word>-<word>" from your phone to +1 415 523 8886
3. Copy Account SID and Auth Token from twilio.com/console
4. Paste into config.json:
      "whatsapp_sid":   "ACxxx...",
      "whatsapp_token": "your_auth_token",
      "whatsapp_from":  "whatsapp:+14155238886",
      "whatsapp_to":    "whatsapp:+91XXXXXXXXXX"

notify_min_strength (default 3) controls which signals trigger alerts.
"""

import json, logging, datetime
import requests

log = logging.getLogger("IFS.notifier")

# ═══════════════════════════════════════════════════════════════════
# TELEGRAM
# ═══════════════════════════════════════════════════════════════════

def send_telegram(token: str, chat_id: str, text: str) -> bool:
    """Send a message via Telegram Bot API."""
    url  = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id":    chat_id,
        "text":       text,
        "parse_mode": "HTML",
    }
    try:
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
        log.info("✅ Telegram message sent")
        return True
    except Exception as e:
        log.error(f"❌ Telegram failed: {e}")
        return False


def format_telegram_message(df, scan_time: str) -> str:
    """Build a nicely formatted Telegram HTML message."""
    total  = len(df)
    strong = len(df[df["strength"] >= 3])

    # Top 10 by strength then vol_ratio
    top = df.head(10)

    sig_counts = {
        "Accumulation": int(df["accumulation"].sum()),
        "Block Trade":  int(df["block_trade"].sum()),
        "Breakout":     int(df["breakout"].sum()),
        "Absorption":   int(df["absorption"].sum()),
    }

    lines = [
        f"🏦 <b>NSE Institutional Footprint</b>",
        f"🕐 {scan_time}",
        "",
        f"📊 <b>{total}</b> signals detected",
        f"🔥 <b>{strong}</b> high-conviction (★★★+)",
        "",
    ]

    # Sector summary
    if "sector" in df.columns:
        sec_summary = df.groupby("sector")["ticker"].count().sort_values(ascending=False).head(5)
        lines.append("📂 <b>Top sectors:</b>")
        for sec, cnt in sec_summary.items():
            lines.append(f"  • {sec}: {cnt}")
        lines.append("")

    # Signal breakdown
    lines.append("📈 <b>Signal breakdown:</b>")
    for sig, cnt in sig_counts.items():
        lines.append(f"  • {sig}: {cnt}")
    lines.append("")

    # Top picks
    lines.append("⭐ <b>Top picks (by conviction):</b>")
    stars = {1:"★☆☆☆", 2:"★★☆☆", 3:"★★★☆", 4:"★★★★"}
    for _, r in top.iterrows():
        chg = f"+{r['chg_pct']:.1f}%" if r['chg_pct'] > 0 else f"{r['chg_pct']:.1f}%"
        arrow = "🟢" if r['chg_pct'] > 0 else "🔴"
        sector_tag = f" [{r['sector']}]" if "sector" in r else ""
        lines.append(
            f"{arrow} <b>{r['ticker']}</b>{sector_tag} "
            f"₹{r['price']:,.0f} {chg} | "
            f"{r['vol_ratio']:.1f}× vol | {stars[r['strength']]} | {r['signals']}"
        )

    lines += [
        "",
        "─" * 30,
        "⚠️ For information only. Not financial advice.",
    ]

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# WHATSAPP (via Twilio)
# ═══════════════════════════════════════════════════════════════════

def send_whatsapp(sid: str, token: str, from_: str, to: str, text: str) -> bool:
    """Send a WhatsApp message via Twilio's API."""
    url     = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
    payload = {"From": from_, "To": to, "Body": text}
    try:
        r = requests.post(url, data=payload, auth=(sid, token), timeout=10)
        r.raise_for_status()
        log.info("✅ WhatsApp message sent")
        return True
    except Exception as e:
        log.error(f"❌ WhatsApp failed: {e}")
        return False


def format_whatsapp_message(df, scan_time: str) -> str:
    """Plain-text version of the alert for WhatsApp."""
    total  = len(df)
    strong = len(df[df["strength"] >= 3])
    top    = df.head(8)
    stars  = {1:"★☆☆☆", 2:"★★☆☆", 3:"★★★☆", 4:"★★★★"}

    lines = [
        f"🏦 NSE INSTITUTIONAL FOOTPRINT",
        f"🕐 {scan_time}",
        f"",
        f"📊 {total} signals | 🔥 {strong} high-conviction",
        f"",
        f"⭐ TOP PICKS:",
    ]
    for _, r in top.iterrows():
        chg = f"+{r['chg_pct']:.1f}%" if r['chg_pct'] > 0 else f"{r['chg_pct']:.1f}%"
        lines.append(
            f"{'🟢' if r['chg_pct']>0 else '🔴'} {r['ticker']} "
            f"₹{r['price']:,.0f} ({chg}) {r['vol_ratio']:.1f}x | "
            f"{stars[r['strength']]} {r['signals']}"
        )

    lines += ["", "Not financial advice."]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# MAIN ENTRY — called by the scanner
# ═══════════════════════════════════════════════════════════════════

def send_alerts(df, cfg: dict):
    """
    Send Telegram and/or WhatsApp alerts for signals above
    the configured minimum strength.
    """
    min_s = cfg.get("notify_min_strength", 3)
    df_alert = df[df["strength"] >= min_s].copy()

    if df_alert.empty:
        log.info(f"No signals ≥ strength {min_s} — no alerts sent.")
        return

    scan_time = datetime.datetime.now().strftime("%d %b %Y %H:%M IST")
    sent_any  = False

    # ── Telegram ─────────────────────────────────────────────────
    tg_token = cfg.get("telegram_token","").strip()
    tg_chat  = cfg.get("telegram_chat_id","").strip()
    if tg_token and tg_chat:
        msg = format_telegram_message(df_alert, scan_time)
        send_telegram(tg_token, tg_chat, msg)
        sent_any = True
    else:
        log.info("Telegram not configured — skipping (see notifier.py for setup)")

    # ── WhatsApp ──────────────────────────────────────────────────
    wa_sid   = cfg.get("whatsapp_sid","").strip()
    wa_token = cfg.get("whatsapp_token","").strip()
    wa_from  = cfg.get("whatsapp_from","").strip()
    wa_to    = cfg.get("whatsapp_to","").strip()
    if wa_sid and wa_token and wa_to:
        msg = format_whatsapp_message(df_alert, scan_time)
        send_whatsapp(wa_sid, wa_token, wa_from, wa_to, msg)
        sent_any = True
    else:
        log.info("WhatsApp not configured — skipping (see notifier.py for setup)")

    if not sent_any:
        log.warning("No notification channel configured. "
                    "Edit config.json to add Telegram/WhatsApp credentials.")


# ── Quick test ────────────────────────────────────────────────────
if __name__ == "__main__":
    import json, os, sys
    cfg_path = os.path.join(os.path.dirname(__file__), "config.json")
    if not os.path.exists(cfg_path):
        print("❌ config.json not found — run the scanner first to generate it.")
        sys.exit(1)
    with open(cfg_path) as f:
        cfg = json.load(f)

    # Send a test message
    scan_time = datetime.datetime.now().strftime("%d %b %Y %H:%M IST")
    test_msg  = (f"✅ <b>NSE Scanner — Test Alert</b>\n"
                 f"🕐 {scan_time}\n\n"
                 f"If you see this, notifications are working!")

    tg_token = cfg.get("telegram_token","").strip()
    tg_chat  = cfg.get("telegram_chat_id","").strip()
    if tg_token and tg_chat:
        ok = send_telegram(tg_token, tg_chat, test_msg)
        print("Telegram:", "✅ sent" if ok else "❌ failed")
    else:
        print("Telegram: not configured")

    wa_sid = cfg.get("whatsapp_sid","").strip()
    if wa_sid:
        ok = send_whatsapp(cfg["whatsapp_sid"], cfg["whatsapp_token"],
                           cfg["whatsapp_from"], cfg["whatsapp_to"],
                           "✅ NSE Scanner — WhatsApp test alert working!")
        print("WhatsApp:", "✅ sent" if ok else "❌ failed")
    else:
        print("WhatsApp: not configured")
