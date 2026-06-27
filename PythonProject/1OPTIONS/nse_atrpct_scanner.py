"""nse_atrpct_scanner.py — ATR% Volatility Rank Scanner"""
from __future__ import annotations
import time
from datetime import date
from nse_pivot_scanner import _find_csv, _read_rows, load_symbol_list
from nse_indicators import atr

_atrpct_cache: dict = {"ts":0.0,"data":None,"key":""}
_DEFAULT_TTL = 300

def scan_stock(symbol:str) -> dict|None:
    p=_find_csv(symbol)
    if not p: return None
    rows,err=_read_rows(p,n=35)
    if err or not rows: return None
    today=date.today().isoformat()
    rows=[r for r in rows if r["date"]<today]
    if len(rows)<20: return None
    H=[r["H"] for r in rows]; L=[r["L"] for r in rows]; C=[r["C"] for r in rows]
    price=C[-1]
    def _atr_pct(h,l,c,n):
        v=atr(h,l,c,n); last=next((x for x in reversed(v) if x is not None),None)
        return round(last/price*100,3) if last and price else None
    a5=_atr_pct(H,L,C,5); a10=_atr_pct(H,L,C,10); a20=_atr_pct(H,L,C,20)
    if a20 is None: return None
    expanding=a5 is not None and a20 is not None and a5>a20
    if   a20>=3.0:  vol_state="High Vol"
    elif a20>=1.5:  vol_state="Normal"
    elif a20>=0.8:  vol_state="Low Vol"
    else:           vol_state="Compressed"
    return {"symbol":symbol,"price":round(price,2),"price_date":rows[-1]["date"],
            "atr5_pct":a5,"atr10_pct":a10,"atr20_pct":a20,
            "expanding":expanding,"vol_state":vol_state}

def run_atrpct_scanner(source="niftyfno",force=False,ttl=_DEFAULT_TTL)->dict:
    key=source; c=_atrpct_cache
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
