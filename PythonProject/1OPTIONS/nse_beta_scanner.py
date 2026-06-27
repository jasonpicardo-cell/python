"""nse_beta_scanner.py — Beta vs Nifty (60D rolling)"""
from __future__ import annotations
import time
from datetime import date
from nse_pivot_scanner import _find_csv, _read_rows, load_symbol_list

_beta_cache: dict = {"ts":0.0,"data":None,"key":""}
_DEFAULT_TTL = 300
_PERIOD = 60

def _load_nifty_rets() -> dict[str,float]|None:
    from nse_pivot_scanner import _find_csv, _read_rows
    for name in ("NIFTY","NIFTY50","NIFTY_50","^NSEI"):
        p=_find_csv(name)
        if p:
            rows,err=_read_rows(p,n=_PERIOD+5)
            if not err and rows:
                today=date.today().isoformat()
                rows=[r for r in rows if r["date"]<today]
                if len(rows)>=_PERIOD+1:
                    return {rows[i]["date"]:(rows[i]["C"]-rows[i-1]["C"])/rows[i-1]["C"]
                            for i in range(1,len(rows)) if rows[i-1]["C"]}
    return None

def _beta(stock_rets,nifty_rets,period=_PERIOD):
    xs=stock_rets[-period:]; ys=nifty_rets[-period:]; n=len(xs)
    if n<20: return None
    mx=sum(xs)/n; my=sum(ys)/n
    cov=sum((xs[i]-mx)*(ys[i]-my) for i in range(n))/(n-1)
    var=sum((y-my)**2 for y in ys)/(n-1)
    return round(cov/var,3) if var>0 else None

def scan_stock(symbol:str,nifty_rets:dict)->dict|None:
    p=_find_csv(symbol)
    if not p: return None
    rows,err=_read_rows(p,n=_PERIOD+5)
    if err or not rows: return None
    today=date.today().isoformat()
    rows=[r for r in rows if r["date"]<today]
    if len(rows)<_PERIOD+1: return None
    stock_d={rows[i]["date"]:(rows[i]["C"]-rows[i-1]["C"])/rows[i-1]["C"]
             for i in range(1,len(rows)) if rows[i-1]["C"]}
    common=sorted(set(stock_d)&set(nifty_rets))[-_PERIOD:]
    if len(common)<20: return None
    xs=[stock_d[d] for d in common]; ys=[nifty_rets[d] for d in common]
    b=_beta(xs,ys)
    if b is None: return None
    if   b>=2.0: signal="Very High Beta"
    elif b>=1.5: signal="High Beta"
    elif b>=1.0: signal="Market"
    elif b>=0.5: signal="Low Beta"
    elif b>=0:   signal="Very Low Beta"
    else:        signal="Negative Beta"
    return {"symbol":symbol,"price":round(rows[-1]["C"],2),"price_date":rows[-1]["date"],
            "beta":b,"signal":signal}

def run_beta_scanner(source="niftyfno",force=False,ttl=_DEFAULT_TTL)->dict:
    key=source; c=_beta_cache
    if not force and c["data"] and c["key"]==key and (time.time()-c["ts"])<ttl: return c["data"]
    nifty_rets=_load_nifty_rets()
    if not nifty_rets:
        return {"stocks":[],"errors":[],"skipped":[],"count":0,"total":0,"source":source,
                "as_of":time.strftime("%H:%M:%S"),"error":"Nifty CSV not found"}
    symbols,sym_label=load_symbol_list(source)
    if not symbols: return {"stocks":[],"errors":[],"skipped":[],"count":0,"total":0,"source":source,"sym_label":sym_label,"as_of":time.strftime("%H:%M:%S"),"error":"Empty"}
    stocks,errors,skipped=[],[],[]
    for sym in symbols:
        try: r=scan_stock(sym,nifty_rets); (stocks if r else skipped).append(r or sym)
        except Exception as e: errors.append({"symbol":sym,"reason":str(e)})
    stocks=[s for s in stocks if isinstance(s,dict)]
    data={"stocks":stocks,"errors":errors,"skipped":skipped,"count":len(stocks),"total":len(symbols),"source":source,"sym_label":sym_label,"as_of":time.strftime("%H:%M:%S")}
    c["ts"]=time.time(); c["data"]=data; c["key"]=key
    return data
