"""
NSE Institutional Footprint Scanner PRO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Enhanced with:
  • Sector-wise grouping & summary
  • Notifier integration (Telegram + WhatsApp)
  • Rich HTML report with sector breakdown
  • config.json for persistent settings

pip install yfinance pandas numpy requests tqdm tabulate


# Run manually
python nse_scanner_pro.py                         # all NSE stocks, daily
python nse_scanner_pro.py --min-strength 3        # high conviction only
python nse_scanner_pro.py --sector "Banking"      # filter to one sector
python nse_scanner_pro.py --top 500 --tf 60m      # intraday Nifty-500
"""


import os, sys, time, json, logging, argparse, datetime, warnings, io, platform
import requests, pandas as pd, numpy as np

warnings.filterwarnings("ignore")

# ── Raise macOS/Linux file descriptor limit early ─────────────────
# yfinance opens one socket per ticker per thread — default macOS
# limit of 256 is exhausted quickly on 2000+ stock scans.
try:
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = min(hard, 4096)
    if soft < target:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        logging.getLogger("IFS").debug(f"FD limit raised: {soft} → {target}")
except Exception:
    pass  # Windows doesn't have resource module — silently skip
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("IFS")

# ── Optional imports ──────────────────────────────────────────────
try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, it, **kw): self._it = it; print(kw.get("desc",""), "started…")
        def __iter__(self):
            for i,v in enumerate(self._it):
                if i%50==0: print(f"  …{i} batches done")
                yield v

try:
    from tabulate import tabulate as _tab
    def tabulate(df, **kw): return _tab(df, **kw)
except ImportError:
    def tabulate(df, **kw): return df.to_string(index=False)

try:
    import yfinance as yf
except ImportError:
    log.error("yfinance missing — run: pip install yfinance"); sys.exit(1)

# ═══════════════════════════════════════════════════════════════════
# SECTOR MASTER — NSE stocks mapped to their indices/sectors
# ═══════════════════════════════════════════════════════════════════
SECTOR_MAP = {
    # ── Information Technology ────────────────────────────────────
    "IT & Software": [
        "TCS","INFY","WIPRO","HCLTECH","TECHM","MPHASIS","PERSISTENT",
        "COFORGE","KPITTECH","TATAELXSI","HEXAWARE","NIITTECH","ZENSAR",
        "MASTEK","BIRLASOFT","SONATSOFTW","LTIM","OFSS","CYIENT","ROUTE",
        "INTELLECT","ECLERX","SAKSOFT","NEWGEN","RAMCO","NUCLEUS",
    ],
    # ── Banking ───────────────────────────────────────────────────
    "Banking": [
        "HDFCBANK","ICICIBANK","KOTAKBANK","SBIN","AXISBANK","INDUSINDBK",
        "BANKBARODA","CANBK","PNB","UNIONBANK","FEDERALBNK","IDFCFIRSTB",
        "BANDHANBNK","RBLBANK","AUBANK","KARURVYSYA","CITYUNIONBANK",
        "EQUITASBNK","UJJIVANSFB","ESAFSFB","SURYODAY","DCBBANK","LAKSHVILAS",
        "TMVFINANCE","YESBANK","JKBANK","SOUTHBANK","KARNATAKA",
    ],
    # ── NBFCs & Financial Services ────────────────────────────────
    "Financial Services": [
        "BAJFINANCE","BAJAJFINSV","HDFCLIFE","SBICARD","CHOLAFIN","MUTHOOTFIN",
        "MANAPPURAM","SHRIRAMFIN","SUNDARMFIN","LTF","RECLTD","PFC",
        "IIFLWAM","ANGELONE","MOTILALOFS","ICICIPRULI","HDFCAMC","MFSL",
        "NIACL","GICRE","STARHEALTH","SBILIFE","CAMS","CDSL","BSE",
        "MCX","ICICIGI","ABCAPITAL","M&MFIN","TATACOMM",
    ],
    # ── Pharma & Healthcare ───────────────────────────────────────
    "Pharma & Healthcare": [
        "SUNPHARMA","DRREDDY","CIPLA","DIVISLAB","TORNTPHARM","AUROPHARMA",
        "LUPIN","BIOCON","IPCA","ALKEM","ZYDUSLIFE","GRANULES","GLAND",
        "NATCOPHARM","AJANTPHARM","APOLLOHOSP","FORTIS","MAXHEALTH",
        "NARAYANHC","METROPOLIS","THYROCARE","DRLAL","VIJAYA","KRSNAA",
        "POLYMED","LONZA","ERIS","LAURUS","STRIDES","PIRAMAL",
    ],
    # ── Energy & Oil ─────────────────────────────────────────────
    "Energy & Oil": [
        "RELIANCE","ONGC","BPCL","IOC","COALINDIA","POWERGRID","NTPC",
        "ADANIENT","TATAPOWER","CESC","TORNTPOWER","ADANIGREEN","ADANITRANS",
        "PGCIL","NHPC","SJVN","GAIL","IGL","MGL","PETRONET","OIL","MRPL",
        "HINDPETRO","CHENNPETRO","GSPL","ABAN","BHEL","THERMAX",
    ],
    # ── Auto & Auto Ancillaries ───────────────────────────────────
    "Automobile": [
        "MARUTI","TATAMOTORS","M&M","BAJAJ-AUTO","HEROMOTOCO","EICHERMOT",
        "ASHOKLEY","TVSMOTOR","MOTHERSON","BHARAT_FORGE","TIINDIA","BOSCHLTD",
        "EXIDEIND","AMARON","MINDA","SONA","CRAFTSMAN","SUPRAJIT","GABRIEL",
        "LUMAX","SANDHAR","UNO MINDA","HAPPYFORGE","ENDURANCE",
    ],
    # ── Metals & Mining ───────────────────────────────────────────
    "Metals & Mining": [
        "TATASTEEL","JSWSTEEL","HINDALCO","VEDL","SAIL","NMDC","COALINDIA",
        "HINDCOPPER","NATIONALUM","WELSPUNLIV","JSPL","RATNAMANI","APL",
        "MOIL","MIDHANI","GMRINFRA","APLAPOLLO","WELCORP",
    ],
    # ── FMCG & Consumer ───────────────────────────────────────────
    "FMCG & Consumer": [
        "HINDUNILVR","ITC","NESTLEIND","BRITANNIA","DABUR","MARICO","COLPAL",
        "GODREJCP","TATACONSUM","EMAMILTD","JYOTHYLAB","VBL","UBL",
        "RADICO","MCDOWELL-N","BALRAMCHIN","BCONCHEM","AGROPHOS",
        "ZYDUSWELL","HNGSNGBRWR","UNITDSPR",
    ],
    # ── Infrastructure & Real Estate ──────────────────────────────
    "Infra & Realty": [
        "LT","ULTRACEMCO","SHREECEM","AMBUJACEM","ACC","RAMCOCEM",
        "DLF","GODREJPROP","OBEROIRLTY","PHOENIXLTD","PRESTIGE","SOBHA",
        "BRIGADE","KOLTEPATIL","MAHLIFE","SUNTECK","MACROTECH",
        "ADANIPORTS","GMRINFRA","IRB","SADBHAV","PNC","KNR","HG",
    ],
    # ── Capital Goods & Engineering ───────────────────────────────
    "Capital Goods": [
        "SIEMENS","ABB","CUMMINSIND","HAVELLS","POLYCAB","KOEL","BHEL",
        "BEL","HAL","COCHINSHIP","GRINDWELL","CARBORUNIV","GRAPHITE",
        "INGERRAND","THERMAX","KABB","VOLTAMP","ISGEC","KENNAMETAL",
        "SCHAEFFLER","SKFINDIA","TIMKEN","ELGIEQUIP","KIRLOSBROS",
    ],
    # ── Consumer Durables & Retail ────────────────────────────────
    "Consumer Durables": [
        "TITAN","ASIANPAINT","BERGEPAINT","PIDILITIND","KAJARIACER","CERA",
        "DIXON","VGUARD","CROMPTON","ORIENTELEC","BLUESTAR","VOLTAS",
        "WHIRLPOOL","BAJAJCON","PAGEIND","TRENT","DMART","NYKAA","SHOPERSTOP",
        "AVENUESSC","ZOMATO","SWIGGY",
    ],
    # ── Telecom & Media ───────────────────────────────────────────
    "Telecom & Media": [
        "BHARTIARTL","VIL","TATACOMM","STLTECH","HFCL","RAILTEL",
        "SUNTV","ZEEL","PVRINOX","INOXWIND",
    ],
    # ── PSU & Defence ─────────────────────────────────────────────
    "PSU & Defence": [
        "BEL","HAL","COCHINSHIP","RVNL","IRFC","IRCTC","IRCON",
        "HSCL","NBCC","RECLTD","PFC","SJVN","NHPC","MOIL","NMDC",
        "GAIL","BPCL","IOC","ONGC","COALINDIA","SAIL",
    ],
    # ── New Age & Tech ────────────────────────────────────────────
    "New Age Tech": [
        "ZOMATO","NYKAA","PAYTM","POLICYBZR","CARTRADE","DELHIVERY",
        "NAUKRI","JUSTDIAL","INDIAMART","MAPMYINDIA",
    ],
}

# Build reverse lookup: ticker → sector
_TICKER_TO_SECTOR: dict[str,str] = {}
for sector, tickers in SECTOR_MAP.items():
    for t in tickers:
        _TICKER_TO_SECTOR[t] = sector

def get_sector(ticker: str) -> str:
    return _TICKER_TO_SECTOR.get(ticker.replace(".NS",""), "Other")

# ═══════════════════════════════════════════════════════════════════
# CONFIG — load / save user settings from config.json
# ═══════════════════════════════════════════════════════════════════
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.json")

DEFAULT_CONFIG = {
    "vol_ma_len":       20,
    "accum_vol_mult":   1.8,
    "block_vol_mult":   3.0,
    "bo_lookback":      20,
    "bo_vol_mult":      2.0,
    "abs_body_pct":     0.5,
    "abs_vol_mult":     2.5,
    "abs_bars":         3,
    "period":           "3mo",
    "interval":         "1d",
    "batch_size":       10,
    "delay_between_batches": 2,
    "min_price":        10,
    "min_avg_vol":      50000,
    "output_csv":       "institutional_signals.csv",
    "output_html":      "institutional_signals.html",
    "telegram_token":   "",
    "telegram_chat_id": "",
    "whatsapp_sid":     "",
    "whatsapp_token":   "",
    "whatsapp_from":    "whatsapp:+14155238886",
    "whatsapp_to":      "",
    "notify_min_strength": 3,
}

def load_config() -> dict:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            user = json.load(f)
        cfg = {**DEFAULT_CONFIG, **user}
    else:
        cfg = DEFAULT_CONFIG.copy()
        save_config(cfg)
    return cfg

def save_config(cfg: dict):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)

# ═══════════════════════════════════════════════════════════════════
# NSE STOCK LIST
# ═══════════════════════════════════════════════════════════════════
NSE_EQUITY_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
NSE_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"),
    "Referer": "https://www.nseindia.com/",
    "Accept-Language": "en-US,en;q=0.9",
}

def fetch_nse_stock_list(top_n=None) -> list[str]:
    log.info("Fetching NSE equity list…")
    try:
        s = requests.Session()
        s.get("https://www.nseindia.com", headers=NSE_HEADERS, timeout=10)
        r = s.get(NSE_EQUITY_URL, headers=NSE_HEADERS, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        col = [c for c in df.columns if "SYMBOL" in c.upper()][0]
        symbols = df[col].str.strip().tolist()
        log.info(f"  ✅ {len(symbols)} NSE stocks found")
    except Exception as e:
        log.warning(f"  NSE fetch failed ({e}) — using Nifty-500 fallback")
        symbols = _nifty500_fallback()
    yf_syms = [f"{s}.NS" for s in symbols if isinstance(s,str) and s.strip()]
    return yf_syms[:top_n] if top_n else yf_syms

def load_watchlist(path: str) -> list[str]:
    with open(path) as f:
        raw = [l.strip().upper() for l in f if l.strip() and not l.startswith("#")]
    return [s if s.endswith(".NS") else f"{s}.NS" for s in raw]

# ═══════════════════════════════════════════════════════════════════
# DATA DOWNLOAD  — safe, rate-limit-aware, with retry
# ═══════════════════════════════════════════════════════════════════
def _download_single(sym: str, period: str, interval: str,
                     retries: int = 2) -> pd.DataFrame | None:
    """
    Download one ticker via yf.Ticker — avoids the multi-thread
    DNS/FD exhaustion that yf.download(threads=True) causes on macOS.
    """
    for attempt in range(retries + 1):
        try:
            df = yf.Ticker(sym).history(period=period, interval=interval,
                                        auto_adjust=True, actions=False)
            if df is not None and not df.empty and len(df) >= 25:
                return df
            return None
        except Exception:
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    return None


def download_batch(symbols: list[str], period: str, interval: str) -> dict:
    """
    Download a batch of symbols one at a time (threads=False equivalent).
    Sequential downloads are ~20% slower but use ~1/50th the file descriptors
    and never exhaust the macOS default socket/thread limits.
    """
    result = {}
    for sym in symbols:
        df = _download_single(sym, period, interval)
        if df is not None:
            result[sym] = df
    return result

# ═══════════════════════════════════════════════════════════════════
# SIGNAL DETECTION
# ═══════════════════════════════════════════════════════════════════
def detect_signals(df: pd.DataFrame, cfg: dict) -> dict | None:
    if not {"Open","High","Low","Close","Volume"}.issubset(df.columns):
        return None
    min_bars = max(cfg["vol_ma_len"], cfg["bo_lookback"]) + cfg["abs_bars"] + 5
    if len(df) < min_bars:
        return None

    d = df[["Open","High","Low","Close","Volume"]].copy()
    d.columns = ["open","high","low","close","volume"]

    last_price = float(d["close"].iloc[-1])
    avg_vol    = float(d["volume"].rolling(cfg["vol_ma_len"]).mean().iloc[-1])
    if last_price < cfg["min_price"] or avg_vol < cfg["min_avg_vol"]:
        return None

    d["avg_vol"]   = d["volume"].rolling(cfg["vol_ma_len"]).mean()
    d["vol_ratio"] = d["volume"] / d["avg_vol"]

    # Accumulation
    upper40  = d["low"] + (d["high"] - d["low"]) * 0.6
    sig_accum = bool(
        (d["volume"].iloc[-1] > d["avg_vol"].iloc[-1] * cfg["accum_vol_mult"]) and
        (d["close"].iloc[-1] > d["open"].iloc[-1]) and
        (d["close"].iloc[-1] >= upper40.iloc[-1])
    )

    # Block trade
    sig_block = bool(d["volume"].iloc[-1] >= d["avg_vol"].iloc[-1] * cfg["block_vol_mult"])

    # Breakout
    hi_n = d["high"].shift(1).rolling(cfg["bo_lookback"]).max()
    lo_n = d["low"].shift(1).rolling(cfg["bo_lookback"]).min()
    bo_bull = bool((d["close"].iloc[-1] > hi_n.iloc[-1]) and
                   (d["volume"].iloc[-1] > d["avg_vol"].iloc[-1] * cfg["bo_vol_mult"]))
    bo_bear = bool((d["close"].iloc[-1] < lo_n.iloc[-1]) and
                   (d["volume"].iloc[-1] > d["avg_vol"].iloc[-1] * cfg["bo_vol_mult"]))
    sig_bo  = bo_bull or bo_bear
    bo_dir  = "↑" if bo_bull else ("↓" if bo_bear else "")

    # Absorption
    body_pct  = (d["close"] - d["open"]).abs() / d["close"] * 100
    is_tight  = (body_pct <= cfg["abs_body_pct"]) & (d["volume"] > d["avg_vol"] * cfg["abs_vol_mult"])
    cum       = (~is_tight).cumsum()
    consec    = is_tight.astype(int).groupby(cum).cumsum()
    sig_abs   = bool(consec.iloc[-1] >= cfg["abs_bars"])

    strength  = sum([sig_accum, sig_block, sig_bo, sig_abs])
    if strength == 0:
        return None

    parts = []
    if sig_accum: parts.append("ACCUM")
    if sig_block: parts.append("BLOCK")
    if sig_bo:    parts.append(f"BO{bo_dir}")
    if sig_abs:   parts.append("ABSORB")

    prev_close = float(d["close"].iloc[-2]) if len(d) > 1 else last_price
    chg_pct    = (last_price - prev_close) / prev_close * 100

    return {
        "price":        round(last_price, 2),
        "chg_pct":      round(chg_pct, 2),
        "volume":       int(d["volume"].iloc[-1]),
        "avg_volume":   int(avg_vol),
        "vol_ratio":    round(float(d["vol_ratio"].iloc[-1]), 2),
        "strength":     strength,
        "accumulation": sig_accum,
        "block_trade":  sig_block,
        "breakout":     sig_bo,
        "bo_dir":       bo_dir,
        "absorption":   sig_abs,
        "signals":      " + ".join(parts),
    }

# ═══════════════════════════════════════════════════════════════════
# MAIN SCAN LOOP
# ═══════════════════════════════════════════════════════════════════
def run_scan(symbols: list[str], cfg: dict) -> pd.DataFrame:
    bs      = cfg["batch_size"]
    batches = [symbols[i:i+bs] for i in range(0, len(symbols), bs)]
    log.info(f"Scanning {len(symbols)} stocks in {len(batches)} batches "
             f"(interval={cfg['interval']}, period={cfg['period']})")
    results   = []
    processed = 0
    for batch in tqdm(batches, desc="Scanning", total=len(batches)):
        data = download_batch(batch, cfg["period"], cfg["interval"])
        processed += len(batch)
        for sym, df in data.items():
            res = detect_signals(df, cfg)
            if res:
                ticker = sym.replace(".NS","")
                results.append({"ticker": ticker,
                                 "sector": get_sector(ticker), **res})
        # Print a heartbeat every 10 batches so it doesn't look frozen
        if (batches.index(batch) + 1) % 10 == 0:
            log.info(f"  → {processed}/{len(symbols)} stocks processed, "
                     f"{len(results)} signals so far")
        time.sleep(cfg["delay_between_batches"])

    if not results:
        log.warning("No institutional signals detected.")
        return pd.DataFrame()

    df_out = pd.DataFrame(results)
    df_out.sort_values(["strength","vol_ratio"], ascending=False, inplace=True)
    df_out.reset_index(drop=True, inplace=True)
    return df_out

# ═══════════════════════════════════════════════════════════════════
# OUTPUT — terminal
# ═══════════════════════════════════════════════════════════════════
STARS = {1:"★☆☆☆", 2:"★★☆☆", 3:"★★★☆", 4:"★★★★"}

def print_results(df: pd.DataFrame, min_strength: int = 1):
    df = df[df["strength"] >= min_strength].copy()
    if df.empty:
        print(f"\n⚠️  No signals with strength ≥ {min_strength}"); return

    scan_dt = datetime.datetime.now().strftime("%d %b %Y %H:%M IST")
    print(f"\n{'═'*76}")
    print(f"  🏦  INSTITUTIONAL FOOTPRINT SCAN  —  {scan_dt}")
    print(f"  {len(df)} signals  |  "
          f"{len(df[df['strength']>=3])} high-conviction (★★★+)")
    print(f"{'═'*76}")

    # ── Sector summary ────────────────────────────────────────────
    sec_grp = df.groupby("sector")["ticker"].count().sort_values(ascending=False)
    print("\n  SECTOR ACTIVITY")
    print(f"  {'─'*50}")
    for sec, cnt in sec_grp.items():
        bar = "█" * min(cnt * 2, 30)
        print(f"  {sec:<28}  {bar}  {cnt}")
    print()

    # ── Per-sector tables ─────────────────────────────────────────
    cols    = ["ticker","price","chg_pct","vol_ratio","strength","signals"]
    rename  = {"ticker":"Ticker","price":"Price ₹","chg_pct":"Chg %",
               "vol_ratio":"Vol Ratio","strength":"★","signals":"Signals"}

    for sector in df["sector"].unique():
        sub = df[df["sector"]==sector][cols].rename(columns=rename).copy()
        sub["★"]     = sub["★"].map(STARS)
        sub["Chg %"] = sub["Chg %"].apply(lambda x: f"+{x:.2f}%" if x>0 else f"{x:.2f}%")
        print(f"  ── {sector.upper()} ({'─'*(46-len(sector))})")
        print(tabulate(sub, headers="keys", tablefmt="simple",
                       showindex=False, numalign="right"))
        print()

# ═══════════════════════════════════════════════════════════════════
# OUTPUT — CSV
# ═══════════════════════════════════════════════════════════════════
def save_csv(df: pd.DataFrame, path: str):
    with open(path, "w", encoding="utf-8", newline="") as f:
        df.to_csv(f, index=False)
    log.info(f"CSV saved → {path}")

# ═══════════════════════════════════════════════════════════════════
# OUTPUT — HTML report (sector-grouped, dark theme)
# ═══════════════════════════════════════════════════════════════════
def save_html(df: pd.DataFrame, path: str):
    scan_time = datetime.datetime.now().strftime("%d %b %Y %H:%M IST")
    total  = len(df)
    strong = len(df[df["strength"]>=3])

    # Sector summary cards
    sec_data  = df.groupby("sector").agg(
        count=("ticker","count"),
        avg_strength=("strength","mean")
    ).sort_values("count", ascending=False)

    sec_cards = ""
    for sec, row in sec_data.iterrows():
        sec_cards += (f'<div class="sec-card">'
                      f'<div class="sec-name">{sec}</div>'
                      f'<div class="sec-cnt">{int(row["count"])}</div>'
                      f'<div class="sec-sub">signals</div></div>')

    # Stock rows grouped by sector
    body = ""
    for sector in df["sector"].unique():
        sub = df[df["sector"]==sector]
        body += (f'<tr class="sec-hdr"><td colspan="7">'
                 f'<i>📂 {sector}</i></td></tr>')
        for _, r in sub.iterrows():
            dots = "●"*r["strength"] + "○"*(4-r["strength"])
            chg_cls = "pos" if r["chg_pct"] > 0 else "neg"
            chg_str = f"+{r['chg_pct']:.2f}%" if r["chg_pct"]>0 else f"{r['chg_pct']:.2f}%"
            tags = ""
            if r["accumulation"]: tags += '<span class="tag accum">ACCUM</span>'
            if r["block_trade"]:  tags += '<span class="tag block">BLOCK</span>'
            if r["breakout"]:     tags += f'<span class="tag bo">BO{r["bo_dir"]}</span>'
            if r["absorption"]:   tags += '<span class="tag abs">ABSORB</span>'
            body += (f"<tr>"
                     f'<td class="sym">{r["ticker"]}</td>'
                     f'<td class="sec-pill">{r["sector"]}</td>'
                     f'<td>₹{r["price"]:,.2f}</td>'
                     f'<td class="{chg_cls}">{chg_str}</td>'
                     f'<td>{r["vol_ratio"]:.1f}×</td>'
                     f'<td class="dots">{dots}</td>'
                     f'<td>{tags}</td></tr>')

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>NSE Institutional Footprint — {scan_time}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
      background:#0d1117;color:#c9d1d9;padding:28px;line-height:1.5}}
h1{{font-size:20px;font-weight:600;color:#f0f6fc;margin-bottom:4px}}
.sub{{font-size:13px;color:#484f58;margin-bottom:20px}}
.kpis{{display:flex;gap:10px;margin-bottom:20px;flex-wrap:wrap}}
.kpi{{background:#161b22;border:1px solid #21262d;border-radius:10px;
      padding:14px 20px;min-width:110px}}
.kpi-v{{font-size:28px;font-weight:700;color:#f0f6fc}}
.kpi-l{{font-size:11px;color:#484f58;text-transform:uppercase;
        letter-spacing:.05em;margin-top:2px}}
.secs{{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:24px}}
.sec-card{{background:#161b22;border:1px solid #21262d;border-radius:8px;
           padding:10px 14px;text-align:center;min-width:100px}}
.sec-name{{font-size:11px;color:#8b949e;margin-bottom:4px}}
.sec-cnt{{font-size:22px;font-weight:700;color:#58a6ff}}
.sec-sub{{font-size:10px;color:#484f58}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th{{background:#161b22;color:#8b949e;font-weight:500;padding:10px 12px;
    text-align:left;border-bottom:1px solid #21262d;position:sticky;top:0;z-index:1}}
td{{padding:8px 12px;border-bottom:1px solid #161b22}}
tr:hover td{{background:#161b22}}
.sec-hdr td{{background:#1c2128;color:#58a6ff;font-size:12px;
             padding:8px 12px;border-top:1px solid #21262d}}
.sym{{font-weight:700;color:#58a6ff;font-size:14px}}
.sec-pill{{font-size:11px;color:#8b949e}}
.pos{{color:#3fb950}}.neg{{color:#f85149}}
.dots{{letter-spacing:2px;color:#d29922;font-size:14px}}
.tag{{display:inline-block;padding:2px 7px;border-radius:4px;
      font-size:11px;font-weight:600;margin-right:3px}}
.accum{{background:#1a4a2e;color:#3fb950}}
.block{{background:#4a1f0a;color:#f0883e}}
.bo{{background:#0d2d6b;color:#58a6ff}}
.abs{{background:#2e1b4a;color:#bc8cff}}
</style>
</head>
<body>
<h1>🏦 NSE Institutional Footprint Scanner</h1>
<p class="sub">Scan time: {scan_time} &nbsp;|&nbsp;
{total} signals detected &nbsp;|&nbsp; {strong} high-conviction (★★★+)</p>
<div class="kpis">
  <div class="kpi"><div class="kpi-v">{total}</div><div class="kpi-l">Total Signals</div></div>
  <div class="kpi"><div class="kpi-v">{strong}</div><div class="kpi-l">High Conviction</div></div>
  <div class="kpi"><div class="kpi-v">{int(df['accumulation'].sum())}</div><div class="kpi-l">Accumulation</div></div>
  <div class="kpi"><div class="kpi-v">{int(df['block_trade'].sum())}</div><div class="kpi-l">Block Trades</div></div>
  <div class="kpi"><div class="kpi-v">{int(df['breakout'].sum())}</div><div class="kpi-l">Breakouts</div></div>
  <div class="kpi"><div class="kpi-v">{int(df['absorption'].sum())}</div><div class="kpi-l">Absorption</div></div>
</div>
<p style="font-size:12px;color:#8b949e;margin-bottom:10px">Sector activity:</p>
<div class="secs">{sec_cards}</div>
<table>
<thead><tr><th>Ticker</th><th>Sector</th><th>Price</th>
<th>Change</th><th>Vol Ratio</th><th>Strength</th><th>Signals</th></tr></thead>
<tbody>{body}</tbody>
</table>
</body></html>"""

    with open(path,"w") as f:
        f.write(html)
    log.info(f"HTML report saved → {path}")

# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="NSE Institutional Footprint Scanner PRO")
    p.add_argument("--tf",           default=None)
    p.add_argument("--top",          type=int)
    p.add_argument("--min-strength", type=int, default=1)
    p.add_argument("--watchlist",    type=str)
    p.add_argument("--period",       default=None)
    p.add_argument("--no-html",      action="store_true")
    p.add_argument("--no-notify",    action="store_true")
    p.add_argument("--sector",       type=str, help="Filter output to one sector")
    return p.parse_args()

def main():
    args = parse_args()
    cfg  = load_config()

    if args.tf:     cfg["interval"] = args.tf
    if args.period: cfg["period"]   = args.period
    if cfg["interval"] in ("5m","15m","30m","60m"):
        cfg["period"] = "5d"

    if args.watchlist:
        symbols = load_watchlist(args.watchlist)
    else:
        symbols = fetch_nse_stock_list(top_n=args.top)

    if not symbols:
        log.error("No symbols to scan."); sys.exit(1)

    t0 = time.time()
    df = run_scan(symbols, cfg)
    log.info(f"Scan complete in {time.time()-t0:.1f}s")

    if df.empty:
        sys.exit(0)

    # Optional sector filter for display
    if args.sector:
        df = df[df["sector"].str.lower().str.contains(args.sector.lower())]

    print_results(df, min_strength=args.min_strength)
    save_csv(df, cfg["output_csv"])

    if not args.no_html:
        save_html(df, cfg["output_html"])

    # Alerts
    if not args.no_notify:
        try:
            from notifier import send_alerts
            send_alerts(df, cfg)
        except ImportError:
            pass  # notifier not present — silent skip
        except Exception as e:
            log.warning(f"Notification failed: {e}")

def _nifty500_fallback():
    return [
        "RELIANCE","TCS","HDFCBANK","INFY","HINDUNILVR","ICICIBANK","KOTAKBANK",
        "SBIN","BHARTIARTL","ASIANPAINT","ITC","LT","AXISBANK","DMART","SUNPHARMA",
        "TITAN","ULTRACEMCO","WIPRO","NESTLEIND","HCLTECH","MARUTI","M&M",
        "BAJFINANCE","TECHM","POWERGRID","NTPC","ONGC","JSWSTEEL","TATASTEEL",
        "COALINDIA","ADANIENT","ADANIPORTS","GRASIM","HDFCLIFE","BAJAJFINSV",
        "DIVISLAB","DRREDDY","CIPLA","EICHERMOT","HEROMOTOCO","BPCL","IOC",
        "TATAMOTORS","BRITANNIA","TATACONSUM","SBICARD","PIDILITIND","AMBUJACEM",
        "SHREECEM","INDUSINDBK","HINDALCO","VEDL","DLF","GODREJCP","MUTHOOTFIN",
        "HAVELLS","BERGEPAINT","COLPAL","MARICO","DABUR","MOTHERSON","LTIM",
        "MPHASIS","PERSISTENT","COFORGE","LTF","CHOLAFIN","BAJAJ-AUTO","BOSCHLTD",
        "SIEMENS","ABB","DIXON","TRENT","PAGEIND","POLYCAB","ZOMATO","NYKAA",
        "PAYTM","IRCTC","RVNL","IRFC","BEL","HAL","BHEL","NMDC","SAIL",
        "RECLTD","PFC","CANBK","BANKBARODA","PNB","UNIONBANK","FEDERALBNK",
        "IDFCFIRSTB","BANDHANBNK","RBLBANK","AUBANK","KARURVYSYA","KPITTECH",
        "TATAELXSI","HEXAWARE","BIRLASOFT","SONATSOFTW","INGERRAND","CUMMINSIND",
        "THERMAX","KOEL","BHARAT_FORGE","MAHINDCIE","TIINDIA","SUNDARMFIN",
        "SHRIRAMFIN","MANAPPURAM","ANGELONE","ICICIPRULI","HDFCAMC",
        "MAXHEALTH","FORTIS","APOLLOHOSP","NARAYANHC","METROPOLIS",
    ]

if __name__ == "__main__":
    main()