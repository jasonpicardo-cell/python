"""nse_swing_scanner.py — Swing High / Swing Low Quality Counter
Counts HH/HL (uptrend) and LH/LL (downtrend) from last 5 swings.
Score +1 per bullish swing, -1 per bearish. Max ±5.
"""
from __future__ import annotations
import time
from datetime import date
from nse_pivot_scanner import _find_csv, _read_rows, load_symbol_list

_swing_cache: dict = {"ts":0.0,"data":None,"key":""}
_DEFAULT_TTL = 300
_N = 3   # fractal bars on each side

def _pivots(rows):
    hs=[]; ls=[]
    for i in range(_N, len(rows)-_N):
        h=rows[i]["H"]; l=rows[i]["L"]
        if all(rows[i-k]["H"]<=h for k in range(1,_N+1)) and all(rows[i+k]["H"]<=h for k in range(1,_N+1)):
            hs.append((i,h))
        if all(rows[i-k]["L"]>=l for k in range(1,_N+1)) and all(rows[i+k]["L"]>=l for k in range(1,_N+1)):
            ls.append((i,l))
    return hs, ls

def scan_stock(symbol:str) -> dict|None:
    p=_find_csv(symbol)
    if not p: return None
    rows,err=_read_rows(p,n=80)
    if err or not rows: return None
    today=date.today().isoformat()
    rows=[r for r in rows if r["date"]<today]
    if len(rows)<20: return None
    hs,ls=_pivots(rows)
    if len(hs)<3 or len(ls)<3: return None
    # Last 5 swing highs and lows
    last_hs=hs[-5:]; last_ls=ls[-5:]
    score=0; hh=hl=lh=ll=0
    for i in range(1,len(last_hs)):
        if last_hs[i][1]>last_hs[i-1][1]: hh+=1; score+=1
        else: lh+=1; score-=1
    for i in range(1,len(last_ls)):
        if last_ls[i][1]>last_ls[i-1][1]: hl+=1; score+=1
        else: ll+=1; score-=1
    if   score>=4: signal="Strong Uptrend"
    elif score>=2: signal="Uptrend"
    elif score<=-4: signal="Strong Downtrend"
    elif score<=-2: signal="Downtrend"
    else:           signal="Choppy"
    return {"symbol":symbol,"price":round(rows[-1]["C"],2),"price_date":rows[-1]["date"],
            "score":score,"hh":hh,"hl":hl,"lh":lh,"ll":ll,"signal":signal}

def run_swing_scanner(source="niftyfno",force=False,ttl=_DEFAULT_TTL)->dict:
    key=source; c=_swing_cache
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
