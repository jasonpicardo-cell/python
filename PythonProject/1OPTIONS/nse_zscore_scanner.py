"""nse_zscore_scanner.py — Mean Reversion Z-Score Scanner (SMA20)"""
from __future__ import annotations
import time
from datetime import date
from nse_pivot_scanner import _find_csv, _read_rows, load_symbol_list
from nse_indicators import sma, stddev

_z_cache: dict = {"ts":0.0,"data":None,"key":""}
_DEFAULT_TTL = 300

def scan_stock(symbol:str) -> dict|None:
    p=_find_csv(symbol)
    if not p: return None
    rows,err=_read_rows(p,n=30)
    if err or not rows: return None
    today=date.today().isoformat()
    rows=[r for r in rows if r["date"]<today]
    if len(rows)<21: return None
    closes=[r["C"] for r in rows]
    price=closes[-1]
    s20=next((v for v in reversed(sma(closes,20)) if v is not None),None)
    std20=next((v for v in reversed(stddev(closes,20)) if v is not None),None)
    if s20 is None or std20 is None or std20==0: return None
    z=round((price-s20)/std20,2)
    if   z>=2.5:  signal="Extreme Overbought"
    elif z>=1.5:  signal="Overbought"
    elif z>=-1.5: signal="Mean Reversion Range"
    elif z>=-2.5: signal="Oversold"
    else:         signal="Extreme Oversold"
    return {"symbol":symbol,"price":round(price,2),"price_date":rows[-1]["date"],
            "zscore":z,"sma20":round(s20,2),"std20":round(std20,2),"signal":signal}

def run_zscore_scanner(source="niftyfno",force=False,ttl=_DEFAULT_TTL)->dict:
    key=source; c=_z_cache
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
