"""nse_roundnum_scanner.py — Round Number Proximity Scanner"""
from __future__ import annotations
import time
from datetime import date
from nse_pivot_scanner import _find_csv, _read_rows, load_symbol_list

_rn_cache: dict = {"ts":0.0,"data":None,"key":""}
_DEFAULT_TTL = 300
ROUND_LEVELS=[50,100,150,200,250,300,400,500,600,750,800,1000,1200,1500,2000,2500,3000,4000,5000,7500,10000]

def scan_stock(symbol:str,threshold_pct:float=1.0) -> dict|None:
    p=_find_csv(symbol)
    if not p: return None
    rows,err=_read_rows(p,n=3)
    if err or not rows: return None
    today=date.today().isoformat()
    rows=[r for r in rows if r["date"]<today]
    if not rows: return None
    price=rows[-1]["C"]
    near=[(lvl,abs(price-lvl)/lvl*100) for lvl in ROUND_LEVELS if abs(price-lvl)/lvl*100<=threshold_pct]
    if not near: return None
    lvl,dist=min(near,key=lambda x:x[1])
    above=price>lvl
    return {"symbol":symbol,"price":round(price,2),"price_date":rows[-1]["date"],
            "level":lvl,"dist_pct":round(dist,3),"above":above,
            "signal":f"{'Above' if above else 'Below'} {lvl}"}

def run_roundnum_scanner(source="niftyfno",force=False,ttl=_DEFAULT_TTL)->dict:
    key=source; c=_rn_cache
    if not force and c["data"] and c["key"]==key and (time.time()-c["ts"])<ttl: return c["data"]
    symbols,sym_label=load_symbol_list(source)
    if not symbols: return {"stocks":[],"errors":[],"skipped":[],"count":0,"total":0,"source":source,"sym_label":sym_label,"as_of":time.strftime("%H:%M:%S"),"error":"Empty"}
    stocks,errors,skipped=[],[],[]
    for sym in symbols:
        try: r=scan_stock(sym); (stocks if r else skipped).append(r or sym)
        except Exception as e: errors.append({"symbol":sym,"reason":str(e)})
    stocks=[s for s in stocks if isinstance(s,dict)]
    stocks.sort(key=lambda s:s.get("dist_pct",99))
    data={"stocks":stocks,"errors":errors,"skipped":skipped,"count":len(stocks),"total":len(symbols),"source":source,"sym_label":sym_label,"as_of":time.strftime("%H:%M:%S")}
    c["ts"]=time.time(); c["data"]=data; c["key"]=key
    return data
