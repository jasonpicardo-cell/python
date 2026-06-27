"""nse_momrank_scanner.py — Price Momentum Percentile Rank (20D return)
Rank each stock's 20D return within the universe (0–100 percentile).
Top decile (>90) = strongest momentum. Bottom decile (<10) = weakest.
"""
from __future__ import annotations
import time
from datetime import date
from nse_pivot_scanner import _find_csv, _read_rows, load_symbol_list

_mr_cache: dict = {"ts":0.0,"data":None,"key":""}
_DEFAULT_TTL = 300
_PERIOD = 20

def _ret(symbol):
    p=_find_csv(symbol)
    if not p: return None, None
    rows,err=_read_rows(p,n=_PERIOD+5)
    if err or not rows: return None, None
    today=date.today().isoformat()
    rows=[r for r in rows if r["date"]<today]
    if len(rows)<_PERIOD+1: return None, None
    c0=rows[-(_PERIOD+1)]["C"]; cn=rows[-1]["C"]
    if not c0: return None, None
    return round((cn-c0)/c0*100,2), rows[-1]

def run_momrank_scanner(source="niftyfno",force=False,ttl=_DEFAULT_TTL)->dict:
    key=source; c=_mr_cache
    if not force and c["data"] and c["key"]==key and (time.time()-c["ts"])<ttl: return c["data"]
    symbols,sym_label=load_symbol_list(source)
    if not symbols: return {"stocks":[],"errors":[],"skipped":[],"count":0,"total":0,"source":source,"sym_label":sym_label,"as_of":time.strftime("%H:%M:%S"),"error":"Empty"}
    # Gather all returns first
    raw=[]; skipped=[]
    for sym in symbols:
        try:
            ret,last_row=_ret(sym)
            if ret is not None and last_row is not None:
                raw.append((sym,ret,last_row))
            else: skipped.append(sym)
        except Exception: skipped.append(sym)
    # Rank
    raw.sort(key=lambda x:x[1])
    n=len(raw)
    stocks=[]
    for idx,(sym,ret,row) in enumerate(raw):
        pct_rank=round(idx/(n-1)*100,1) if n>1 else 50.0
        if   pct_rank>=90: signal="Top Decile"
        elif pct_rank>=75: signal="Top Quartile"
        elif pct_rank>=50: signal="Above Median"
        elif pct_rank>=25: signal="Below Median"
        elif pct_rank>=10: signal="Bottom Quartile"
        else:              signal="Bottom Decile"
        stocks.append({"symbol":sym,"price":round(row["C"],2),"price_date":row["date"],
                       "ret20":ret,"pct_rank":pct_rank,"signal":signal})
    stocks.sort(key=lambda s:-s["pct_rank"])
    data={"stocks":stocks,"errors":[],"skipped":skipped,"count":len(stocks),"total":len(symbols),
          "source":source,"sym_label":sym_label,"as_of":time.strftime("%H:%M:%S")}
    c["ts"]=time.time(); c["data"]=data; c["key"]=key
    return data
