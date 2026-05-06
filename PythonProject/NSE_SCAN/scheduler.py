"""
NSE Scanner Scheduler
═════════════════════
Runs the institutional footprint scan automatically at 3:35 PM IST
every weekday (Mon–Fri), skipping NSE market holidays.

Usage:
    python scheduler.py              # start the daily scheduler
    python scheduler.py --now        # run one scan immediately then schedule
    python scheduler.py --time 15:40 # custom trigger time (24h, IST)
    python scheduler.py --test       # fire in 10 seconds (for testing)

Keep this running in the background:
    Linux/Mac : nohup python scheduler.py > scanner.log 2>&1 &
    Windows   : start /B pythonw scheduler.py
    Or use a terminal multiplexer like tmux / screen.
"""

import os, sys, time, logging, argparse, datetime, subprocess, json, threading
import signal as _signal

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)s  %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("scheduler.log", encoding="utf-8"),
    ]
)
log = logging.getLogger("Scheduler")

# ── NSE market holidays 2025-2026 ────────────────────────────────
# Source: NSE official calendar — update yearly
NSE_HOLIDAYS = {
    # 2025
    datetime.date(2025,  1, 26),  # Republic Day
    datetime.date(2025,  2, 19),  # Chhatrapati Shivaji Maharaj Jayanti
    datetime.date(2025,  3, 14),  # Holi
    datetime.date(2025,  4,  1),  # Gudi Padwa / Annual Closing
    datetime.date(2025,  4, 10),  # Mahavir Jayanti (if applicable)
    datetime.date(2025,  4, 14),  # Dr. Ambedkar Jayanti / Good Friday
    datetime.date(2025,  4, 18),  # Good Friday
    datetime.date(2025,  5,  1),  # Maharashtra Day
    datetime.date(2025,  8, 15),  # Independence Day
    datetime.date(2025,  8, 27),  # Ganesh Chaturthi
    datetime.date(2025, 10,  2),  # Gandhi Jayanti
    datetime.date(2025, 10,  2),  # Dussehra
    datetime.date(2025, 10, 24),  # Diwali Laxmi Puja
    datetime.date(2025, 11,  5),  # Diwali Balipratipada
    datetime.date(2025, 11, 15),  # Gurunanak Jayanti
    datetime.date(2025, 12, 25),  # Christmas
    # 2026
    datetime.date(2026,  1, 26),  # Republic Day
    datetime.date(2026,  3,  3),  # Holi
    datetime.date(2026,  3, 25),  # Gudi Padwa
    datetime.date(2026,  4,  3),  # Good Friday
    datetime.date(2026,  4, 14),  # Dr. Ambedkar Jayanti
    datetime.date(2026,  5,  1),  # Maharashtra Day
    datetime.date(2026,  8, 15),  # Independence Day
    datetime.date(2026, 10,  2),  # Gandhi Jayanti
    datetime.date(2026, 11, 14),  # Diwali Laxmi Puja (approx)
    datetime.date(2026, 12, 25),  # Christmas
}

SCANNER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "nse_scanner_pro.py")

# ═══════════════════════════════════════════════════════════════════
# MARKET DAY CHECK
# ═══════════════════════════════════════════════════════════════════
def is_market_day(date: datetime.date = None) -> bool:
    """Return True if the given date is an NSE trading day."""
    d = date or datetime.date.today()
    if d.weekday() >= 5:          # Saturday=5, Sunday=6
        return False
    if d in NSE_HOLIDAYS:
        return False
    return True


def next_market_day(from_date: datetime.date = None) -> datetime.date:
    """Return the next (or current) NSE trading day."""
    d = from_date or datetime.date.today()
    while not is_market_day(d):
        d += datetime.timedelta(days=1)
    return d

# ═══════════════════════════════════════════════════════════════════
# SCAN EXECUTION
# ═══════════════════════════════════════════════════════════════════
def run_scan(extra_args: list[str] = None):
    """
    Execute the scanner as a subprocess so it runs in its own process
    and any crash doesn't kill the scheduler.
    """
    cmd = [sys.executable, SCANNER_SCRIPT]
    if extra_args:
        cmd.extend(extra_args)

    log.info(f"🚀 Starting scan: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=False, timeout=3600)
        if result.returncode == 0:
            log.info("✅ Scan completed successfully")
        else:
            log.warning(f"⚠️  Scan exited with code {result.returncode}")
    except subprocess.TimeoutExpired:
        log.error("❌ Scan timed out after 60 minutes")
    except Exception as e:
        log.error(f"❌ Scan failed: {e}")

# ═══════════════════════════════════════════════════════════════════
# SCHEDULER CORE
# ═══════════════════════════════════════════════════════════════════
IST_OFFSET = datetime.timezone(datetime.timedelta(hours=5, minutes=30))

def now_ist() -> datetime.datetime:
    return datetime.datetime.now(IST_OFFSET)

def next_trigger(trigger_time: datetime.time) -> datetime.datetime:
    """
    Calculate the next datetime (IST) when the scan should run.
    Skips weekends and NSE holidays.
    """
    now = now_ist()
    candidate_date = now.date()

    # If today's trigger time has already passed, move to next day
    trigger_today = datetime.datetime.combine(
        candidate_date, trigger_time,
        tzinfo=IST_OFFSET
    )
    if now >= trigger_today:
        candidate_date += datetime.timedelta(days=1)

    # Skip to next market day
    candidate_date = next_market_day(candidate_date)

    return datetime.datetime.combine(candidate_date, trigger_time, tzinfo=IST_OFFSET)


def seconds_until(target: datetime.datetime) -> float:
    return max(0.0, (target - now_ist()).total_seconds())


def fmt_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0: return f"{h}h {m}m"
    if m > 0: return f"{m}m {s}s"
    return f"{s}s"


class SchedulerDaemon:
    def __init__(self, trigger_time: datetime.time, extra_args: list[str] = None):
        self.trigger_time = trigger_time
        self.extra_args   = extra_args or []
        self._stop        = threading.Event()
        self._scan_count  = 0

        # Graceful shutdown on SIGINT / SIGTERM
        _signal.signal(_signal.SIGINT,  self._shutdown)
        _signal.signal(_signal.SIGTERM, self._shutdown)

    def _shutdown(self, *_):
        log.info("🛑 Shutdown signal received — stopping scheduler…")
        self._stop.set()

    def run(self):
        log.info("═" * 55)
        log.info("  NSE Institutional Footprint Scheduler  v1.0")
        log.info(f"  Trigger time : {self.trigger_time.strftime('%H:%M')} IST (Mon–Fri)")
        log.info(f"  Scanner      : {SCANNER_SCRIPT}")
        log.info("═" * 55)

        while not self._stop.is_set():
            trigger = next_trigger(self.trigger_time)
            wait    = seconds_until(trigger)

            log.info(f"⏳ Next scan: {trigger.strftime('%a %d %b %Y %H:%M IST')} "
                     f"(in {fmt_duration(wait)})")

            # Sleep in short chunks so we can respond to shutdown signals quickly
            deadline = time.monotonic() + wait
            while time.monotonic() < deadline and not self._stop.is_set():
                remaining = deadline - time.monotonic()
                time.sleep(min(60, remaining))    # wake up at least every minute

            if self._stop.is_set():
                break

            # Double-check it's still a market day (holiday list might differ)
            if not is_market_day():
                log.info(f"📅 {datetime.date.today()} is not a market day — skipping")
                continue

            self._scan_count += 1
            log.info(f"🔔 Scan #{self._scan_count} triggered at "
                     f"{now_ist().strftime('%H:%M:%S IST')}")
            run_scan(self.extra_args)

        log.info("Scheduler stopped. Goodbye.")

# ═══════════════════════════════════════════════════════════════════
# STATUS / NEXT RUN PREVIEW
# ═══════════════════════════════════════════════════════════════════
def show_status(trigger_time: datetime.time):
    """Print the next 5 scheduled scan times."""
    print()
    print("  SCHEDULED SCAN TIMES (next 5 market days)")
    print("  " + "─"*44)
    d = datetime.date.today()
    shown = 0
    while shown < 5:
        if is_market_day(d):
            dt = datetime.datetime.combine(d, trigger_time)
            print(f"  {dt.strftime('%a %d %b %Y')}  →  {dt.strftime('%H:%M')} IST")
            shown += 1
        d += datetime.timedelta(days=1)
    print()

# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="NSE Scanner — Daily Scheduler")
    p.add_argument("--time",   default="15:35",
                   help="Scan trigger time IST (24h format, default: 15:35)")
    p.add_argument("--now",    action="store_true",
                   help="Run one scan immediately, then start scheduling")
    p.add_argument("--test",   action="store_true",
                   help="Fire a scan in 10 seconds (for testing)")
    p.add_argument("--status", action="store_true",
                   help="Show next 5 scheduled scan times and exit")
    # Pass-through args to the scanner
    p.add_argument("--top",          type=int,   help="Limit to top N stocks")
    p.add_argument("--min-strength", type=int,   help="Min signal strength to alert")
    p.add_argument("--tf",           type=str,   help="Timeframe override")
    p.add_argument("--no-html",      action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    # Parse trigger time
    try:
        h, m = map(int, args.time.split(":"))
        trigger_time = datetime.time(h, m)
    except ValueError:
        print(f"❌ Invalid time format: {args.time}  (use HH:MM e.g. 15:35)")
        sys.exit(1)

    # Build scanner args to pass through
    scanner_args = []
    if args.top:          scanner_args += ["--top", str(args.top)]
    if args.min_strength: scanner_args += ["--min-strength", str(args.min_strength)]
    if args.tf:           scanner_args += ["--tf", args.tf]
    if args.no_html:      scanner_args += ["--no-html"]

    # Status only
    if args.status:
        show_status(trigger_time)
        sys.exit(0)

    # Test mode — fire in 10 s
    if args.test:
        log.info("🧪 Test mode — firing scan in 10 seconds…")
        time.sleep(10)
        run_scan(scanner_args)
        sys.exit(0)

    # Immediate scan before scheduling
    if args.now:
        log.info("▶️  Running immediate scan before scheduling…")
        run_scan(scanner_args)

    show_status(trigger_time)
    daemon = SchedulerDaemon(trigger_time, scanner_args)
    daemon.run()


if __name__ == "__main__":
    main()
