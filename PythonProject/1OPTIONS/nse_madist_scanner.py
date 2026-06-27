"""nse_madist_scanner.py — MA Distance Scanner (20/50/200 SMA)"""
from __future__ import annotations
import time
from datetime import date
from nse_pivot_scanner import _find_csv, _read_rows, load_symbol_list
from nse_indicators import sma

_mad_cache: dict = {"ts":0.0,"data":None,"key":""}
_DEFAULT_TTL = 300

def scan_stock(symbol:str) -> dict|None:
    p=_find_csv(symbol)
    if not p: return None
    rows,err=_read_rows(p,n=215)
    if err or not rows: return None
    today=date.today().isoformat()
    rows=[r for r in rows if r["date"]<today]
    if len(rows)<21: return None
    closes=[r["C"] for r in rows]; price=closes[-1]
    def _dist(period):
        v=next((x for x in reversed(sma(closes,period)) if x is not None),None)
        if v is None or v==0: return None,None
        return round(v,2), round((price-v)/v*100,2)
    s20,d20=_dist(20); s50,d50=_dist(50) if len(closes)>=50 else (None,None)
    s200,d200=_dist(200) if len(closes)>=200 else (None,None)
    if s20 is None: return None
    # Signal based on 200D distance (if available) else 50D
    primary_dist=d200 if d200 is not None else (d50 if d50 is not None else d20)
    if   primary_dist is not None and primary_dist>=20:  signal="Extended Above"
    elif primary_dist is not None and primary_dist>=5:   signal="Above MA"
    elif primary_dist is not None and primary_dist>=-5:  signal="At MA Zone"
    elif primary_dist is not None and primary_dist>=-20: signal="Below MA"
    else:                                                signal="Extended Below"
    return {"symbol":symbol,"price":round(price,2),"price_date":rows[-1]["date"],
            "sma20":s20,"dist20":d20,"sma50":s50,"dist50":d50,"sma200":s200,"dist200":d200,"signal":signal}

def run_madist_scanner(source="niftyfno",force=False,ttl=_DEFAULT_TTL)->dict:
    key=source; c=_mad_cache
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
