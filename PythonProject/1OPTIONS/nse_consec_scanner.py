"""nse_consec_scanner.py — Consecutive Days (Bull/Bear Streak)"""
from __future__ import annotations
import time
from datetime import date
from nse_pivot_scanner import _find_csv, _read_rows, load_symbol_list

_consec_cache: dict = {"ts":0.0,"data":None,"key":""}
_DEFAULT_TTL = 300

def scan_stock(symbol:str) -> dict|None:
    p=_find_csv(symbol)
    if not p: return None
    rows,err=_read_rows(p,n=20)
    if err or not rows: return None
    today=date.today().isoformat()
    rows=[r for r in rows if r["date"]<today]
    if len(rows)<3: return None
    closes=[r["C"] for r in rows]
    direction="bull" if closes[-1]>closes[-2] else ("bear" if closes[-1]<closes[-2] else None)
    if direction is None: return None
    count=1
    for i in range(len(closes)-2,0,-1):
        if direction=="bull" and closes[i]>closes[i-1]: count+=1
        elif direction=="bear" and closes[i]<closes[i-1]: count+=1
        else: break
    if count<2: return None  # only return 2+ streaks
    signal=f"{count}D {'Bull' if direction=='bull' else 'Bear'}"
    exhaustion=count>=5
    return {"symbol":symbol,"price":round(closes[-1],2),"price_date":rows[-1]["date"],
            "count":count,"direction":direction,"signal":signal,"exhaustion":exhaustion}

def run_consec_scanner(source="niftyfno",force=False,ttl=_DEFAULT_TTL)->dict:
    key=source; c=_consec_cache
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
