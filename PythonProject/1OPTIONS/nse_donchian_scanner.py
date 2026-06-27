"""nse_donchian_scanner.py — Donchian Channel Breakout Scanner (20D / 55D)"""
from __future__ import annotations
import time
from datetime import date
from nse_pivot_scanner import _find_csv, _read_rows, load_symbol_list

_dc_cache: dict = {"ts":0.0,"data":None,"key":""}
_DEFAULT_TTL = 300

def scan_stock(symbol:str) -> dict|None:
    p=_find_csv(symbol)
    if not p: return None
    rows,err=_read_rows(p,n=65)
    if err or not rows: return None
    today=date.today().isoformat()
    rows=[r for r in rows if r["date"]<today]
    if len(rows)<22: return None
    price=rows[-1]["C"]
    # Exclude today from the channel calculation
    hist=rows[:-1]
    h20=max(r["H"] for r in hist[-20:]); l20=min(r["L"] for r in hist[-20:])
    h55=(max(r["H"] for r in hist[-55:]) if len(hist)>=55 else None)
    l55=(min(r["L"] for r in hist[-55:]) if len(hist)>=55 else None)
    mid20=round((h20+l20)/2,2)
    if   price>h20:               signal="20D Breakout"
    elif price<l20:               signal="20D Breakdown"
    elif h55 and price>h55:       signal="55D Breakout"
    elif l55 and price<l55:       signal="55D Breakdown"
    elif price>mid20:             signal="Upper Half"
    else:                         signal="Lower Half"
    return {"symbol":symbol,"price":round(price,2),"price_date":rows[-1]["date"],
            "upper_20":round(h20,2),"lower_20":round(l20,2),"mid_20":mid20,
            "upper_55":round(h55,2) if h55 else None,"lower_55":round(l55,2) if l55 else None,
            "signal":signal}

def run_donchian_scanner(source="niftyfno",force=False,ttl=_DEFAULT_TTL)->dict:
    key=source; c=_dc_cache
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
