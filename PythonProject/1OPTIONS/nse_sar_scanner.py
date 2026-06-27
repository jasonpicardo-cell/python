"""nse_sar_scanner.py — Parabolic SAR Scanner (AF 0.02, step 0.02, max 0.2)"""
from __future__ import annotations
import time
from datetime import date
from nse_pivot_scanner import _find_csv, _read_rows, load_symbol_list

_sar_cache: dict = {"ts":0.0,"data":None,"key":""}
_DEFAULT_TTL = 300

def _parabolic_sar(highs,lows,closes,af0=0.02,af_step=0.02,af_max=0.2):
    n=len(closes)
    if n<3: return [],[]
    sar=[0.0]*n; ep=[0.0]*n; af=[af0]*n; trend=[1]*n
    trend[0]=1 if closes[1]>closes[0] else -1
    sar[0]=lows[0] if trend[0]==1 else highs[0]
    ep[0]=highs[0] if trend[0]==1 else lows[0]
    for i in range(1,n):
        pt=trend[i-1]; ps=sar[i-1]; pe=ep[i-1]; pa=af[i-1]
        ns=ps+pa*(pe-ps)
        if pt==1:
            ns=min(ns,lows[i-1],lows[i-2] if i>=2 else lows[i-1])
            if lows[i]<ns:
                trend[i]=-1; sar[i]=pe; ep[i]=lows[i]; af[i]=af0
            else:
                trend[i]=1; sar[i]=ns
                ep[i]=max(highs[i],pe); af[i]=min(pa+(af_step if highs[i]>pe else 0),af_max)
        else:
            ns=max(ns,highs[i-1],highs[i-2] if i>=2 else highs[i-1])
            if highs[i]>ns:
                trend[i]=1; sar[i]=pe; ep[i]=highs[i]; af[i]=af0
            else:
                trend[i]=-1; sar[i]=ns
                ep[i]=min(lows[i],pe); af[i]=min(pa+(af_step if lows[i]<pe else 0),af_max)
    return trend,sar

def scan_stock(symbol:str) -> dict|None:
    p=_find_csv(symbol)
    if not p: return None
    rows,err=_read_rows(p,n=50)
    if err or not rows: return None
    today=date.today().isoformat()
    rows=[r for r in rows if r["date"]<today]
    if len(rows)<10: return None
    H=[r["H"] for r in rows]; L=[r["L"] for r in rows]; C=[r["C"] for r in rows]
    trend,sar_vals=_parabolic_sar(H,L,C)
    if not trend: return None
    cur_t=trend[-1]; cur_sar=sar_vals[-1]; price=C[-1]
    fresh=len(trend)>=2 and trend[-1]!=trend[-2]
    signal=("SAR Bullish" if cur_t==1 else "SAR Bearish")+((" — Fresh Flip" if fresh else ""))
    return {"symbol":symbol,"price":round(price,2),"price_date":rows[-1]["date"],
            "sar":round(cur_sar,2),"direction":cur_t,"fresh":fresh,"signal":signal}

def run_sar_scanner(source="niftyfno",force=False,ttl=_DEFAULT_TTL)->dict:
    key=source; c=_sar_cache
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
