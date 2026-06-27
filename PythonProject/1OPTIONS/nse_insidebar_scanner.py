"""nse_insidebar_scanner.py — Inside Bar Scanner"""
from __future__ import annotations
import time
from datetime import date
from nse_pivot_scanner import _find_csv, _read_rows, load_symbol_list

_ib_cache: dict = {"ts":0.0,"data":None,"key":""}
_DEFAULT_TTL = 300

def scan_stock(symbol:str) -> dict|None:
    p=_find_csv(symbol)
    if not p: return None
    rows,err=_read_rows(p,n=7)
    if err or not rows: return None
    today=date.today().isoformat()
    rows=[r for r in rows if r["date"]<today]
    if len(rows)<3: return None
    c=rows[-1]; p1=rows[-2]; p2=rows[-3] if len(rows)>=3 else None
    inside=c["H"]<p1["H"] and c["L"]>p1["L"]
    if not inside: return None
    double=p2 is not None and p1["H"]<p2["H"] and p1["L"]>p2["L"]
    # NR7 check
    nr7_check=False
    if len(rows)>=8:
        cr=c["H"]-c["L"]
        prev=[r["H"]-r["L"] for r in rows[-8:-1]]
        nr7_check=bool(prev) and cr<min(prev)
    rng=c["H"]-c["L"] or 1
    pos=(c["C"]-c["L"])/rng
    bias="Bullish" if pos>=0.6 else ("Bearish" if pos<=0.4 else "Neutral")
    ib_type="Double IB+NR7" if (double and nr7_check) else ("Double IB" if double else ("IB+NR7" if nr7_check else "Inside Bar"))
    return {"symbol":symbol,"price":round(c["C"],2),"price_date":c["date"],
            "mother_high":round(p1["H"],2),"mother_low":round(p1["L"],2),
            "type":ib_type,"bias":bias,"double":double,"nr7":nr7_check}

def run_insidebar_scanner(source="niftyfno",force=False,ttl=_DEFAULT_TTL)->dict:
    key=source; c=_ib_cache
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
