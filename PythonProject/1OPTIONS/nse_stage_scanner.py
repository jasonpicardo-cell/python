"""nse_stage_scanner.py — Weinstein Stage Analysis (150D SMA / 30-week)
Stage 1: Basing    — price near flat 150D MA
Stage 2: Advancing — price above rising 150D MA  ← only stage to be long
Stage 3: Topping   — price flat/below declining from highs
Stage 4: Declining — price below falling 150D MA  ← avoid/short
"""
from __future__ import annotations
import time
from datetime import date
from nse_pivot_scanner import _find_csv, _read_rows, load_symbol_list
from nse_indicators import sma

_stage_cache: dict = {"ts":0.0,"data":None,"key":""}
_DEFAULT_TTL = 300

def scan_stock(symbol:str) -> dict|None:
    p=_find_csv(symbol)
    if not p: return None
    rows,err=_read_rows(p,n=170)
    if err or not rows: return None
    today=date.today().isoformat()
    rows=[r for r in rows if r["date"]<today]
    if len(rows)<155: return None
    closes=[r["C"] for r in rows]; price=closes[-1]
    s150=sma(closes,150); cur=next((v for v in reversed(s150) if v is not None),None)
    if cur is None: return None
    # Slope: change over last 10 bars
    prev_vals=[v for v in s150[-15:] if v is not None]
    if len(prev_vals)<5: return None
    slope_pct=(prev_vals[-1]-prev_vals[0])/prev_vals[0]*100
    above=price>cur
    dist=round((price-cur)/cur*100,2)
    if above:
        if slope_pct>0.5:       stage,label=2,"Stage 2 — Advancing"
        elif slope_pct>-0.3:    stage,label=3,"Stage 3 — Topping"
        else:                   stage,label=4,"Stage 4 — Declining (Late)"
    else:
        if slope_pct<-0.5:      stage,label=4,"Stage 4 — Declining"
        elif slope_pct<0.3:     stage,label=1,"Stage 1 — Basing"
        else:                   stage,label=2,"Stage 2 — Early (Below MA)"
    return {"symbol":symbol,"price":round(price,2),"price_date":rows[-1]["date"],
            "sma150":round(cur,2),"slope_pct":round(slope_pct,3),
            "dist_pct":dist,"above_ma":above,"stage":stage,"signal":label}

def run_stage_scanner(source="niftyfno",force=False,ttl=_DEFAULT_TTL)->dict:
    key=source; c=_stage_cache
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
