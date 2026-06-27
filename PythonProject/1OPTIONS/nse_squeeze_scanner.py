"""nse_squeeze_scanner.py — BB + Keltner Channel Squeeze (John Carter)
Squeeze: BB inside KC → extreme volatility compression → big move imminent.
BB: SMA20 ± 2*std20
KC: EMA20 ± 1.5*ATR14
"""
from __future__ import annotations
import time
from datetime import date
from nse_pivot_scanner import _find_csv, _read_rows, load_symbol_list
from nse_indicators import sma, stddev, ema, atr

_sq_cache: dict = {"ts":0.0,"data":None,"key":""}
_DEFAULT_TTL = 300

def scan_stock(symbol:str) -> dict|None:
    p=_find_csv(symbol)
    if not p: return None
    rows,err=_read_rows(p,n=40)
    if err or not rows: return None
    today=date.today().isoformat()
    rows=[r for r in rows if r["date"]<today]
    if len(rows)<22: return None
    H=[r["H"] for r in rows]; L=[r["L"] for r in rows]; C=[r["C"] for r in rows]
    price=C[-1]
    e20=next((v for v in reversed(ema(C,20)) if v is not None),None)
    s20=next((v for v in reversed(sma(C,20)) if v is not None),None)
    sd=next((v for v in reversed(stddev(C,20)) if v is not None),None)
    at14=next((v for v in reversed(atr(H,L,C,14)) if v is not None),None)
    if None in (e20,s20,sd,at14): return None
    bb_upper=s20+2*sd; bb_lower=s20-2*sd
    kc_upper=e20+1.5*at14; kc_lower=e20-1.5*at14
    squeeze=bb_upper<kc_upper and bb_lower>kc_lower
    bb_width=round((bb_upper-bb_lower)/s20*100,2)
    kc_width=round((kc_upper-kc_lower)/e20*100,2) if e20 else 0
    # Momentum: close relative to midpoint of high/low range
    val=price-((max(H[-20:])+min(L[-20:]))/2+s20)/2
    if   squeeze and val>0:  signal="Squeeze — Bullish Build"
    elif squeeze and val<0:  signal="Squeeze — Bearish Build"
    elif squeeze:            signal="Squeeze"
    elif not squeeze and bb_width<kc_width*1.05: signal="Squeeze Released"
    else:                    signal="No Squeeze"
    return {"symbol":symbol,"price":round(price,2),"price_date":rows[-1]["date"],
            "squeeze":squeeze,"bb_width":bb_width,"kc_width":kc_width,
            "bb_upper":round(bb_upper,2),"bb_lower":round(bb_lower,2),
            "kc_upper":round(kc_upper,2),"kc_lower":round(kc_lower,2),
            "momentum_val":round(val,2),"signal":signal}

def run_squeeze_scanner(source="niftyfno",force=False,ttl=_DEFAULT_TTL)->dict:
    key=source; c=_sq_cache
    if not force and c["data"] and c["key"]==key and (time.time()-c["ts"])<ttl: return c["data"]
    symbols,sym_label=load_symbol_list(source)
    if not symbols: return {"stocks":[],"errors":[],"skipped":[],"count":0,"total":0,"source":source,"sym_label":sym_label,"as_of":time.strftime("%H:%M:%S"),"error":"Empty"}
    stocks,errors,skipped=[],[],[]
    for sym in symbols:
        try: r=scan_stock(sym); (stocks if r else skipped).append(r or sym)
        except Exception as e: errors.append({"symbol":sym,"reason":str(e)})
    stocks=[s for s in stocks if isinstance(s,dict)]
    data={"stocks":stocks,"errors":errors,"skipped":skipped,"count":len(stocks),"total":len(symbols),"source":source,"sym_label":sym_label,"as_of":time.strftime("%H:%M:%S")}
    c["ts"]=time.time(); c["data"]=data; c["key"]=key
    return data
