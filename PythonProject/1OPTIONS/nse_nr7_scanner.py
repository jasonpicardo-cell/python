"""nse_nr7_scanner.py — NR7 / NR4 Volatility Squeeze Scanner"""
from __future__ import annotations
import time
from datetime import date
from nse_pivot_scanner import _find_csv, _read_rows, load_symbol_list

_nr7_cache: dict = {"ts":0.0,"data":None,"key":""}
_DEFAULT_TTL = 300

def scan_stock(symbol:str) -> dict|None:
    p=_find_csv(symbol)
    if not p: return None
    rows,err=_read_rows(p,n=12)
    if err or not rows: return None
    today=date.today().isoformat()
    rows=[r for r in rows if r["date"]<today]
    if len(rows)<8: return None
    cur=rows[-1]; cr=cur["H"]-cur["L"]
    prev=[r["H"]-r["L"] for r in rows[-8:-1]]
    if not prev: return None
    is_nr7=cr<min(prev)
    is_nr4=cr<min(prev[-4:]) if len(prev)>=4 else False
    if not (is_nr7 or is_nr4): return None
    # Direction bias: close near high=bullish, near low=bearish
    rng=cur["H"]-cur["L"] or 1
    pos=(cur["C"]-cur["L"])/rng
    bias="Bullish" if pos>=0.6 else ("Bearish" if pos<=0.4 else "Neutral")
    ntype="NR7" if is_nr7 else "NR4"
    return {"symbol":symbol,"price":round(cur["C"],2),"price_date":cur["date"],
            "range":round(cr,2),"min_prev_range":round(min(prev),2),
            "type":ntype,"bias":bias,"pct_pos":round(pos*100,1)}

def run_nr7_scanner(source="niftyfno",force=False,ttl=_DEFAULT_TTL)->dict:
    key=source; c=_nr7_cache
    if not force and c["data"] and c["key"]==key and (time.time()-c["ts"])<ttl: return c["data"]
    symbols,sym_label=load_symbol_list(source)
    if not symbols: return {"stocks":[],"errors":[],"skipped":[],"count":0,"total":0,"source":source,"sym_label":sym_label,"as_of":time.strftime("%H:%M:%S"),"error":"Empty"}
    stocks,errors,skipped=[],[],[]
    for sym in symbols:
        try:
            r=scan_stock(sym); (stocks if r else skipped).append(r or sym)
        except Exception as e: errors.append({"symbol":sym,"reason":str(e)})
    stocks=[s for s in stocks if isinstance(s,dict)]
    data={"stocks":stocks,"errors":errors,"skipped":skipped,"count":len(stocks),"total":len(symbols),"source":source,"sym_label":sym_label,"as_of":time.strftime("%H:%M:%S")}
    c["ts"]=time.time(); c["data"]=data; c["key"]=key
    return data
