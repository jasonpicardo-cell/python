"""nse_hv_scanner.py — Historical Volatility Scanner (annualised, 20D)"""
from __future__ import annotations
import math, time
from datetime import date
from nse_pivot_scanner import _find_csv, _read_rows, load_symbol_list

_hv_cache: dict = {"ts":0.0,"data":None,"key":""}
_DEFAULT_TTL = 300

def _hv(closes:list,period:int)->float|None:
    if len(closes)<period+1: return None
    rets=[math.log(closes[i]/closes[i-1]) for i in range(len(closes)-period,len(closes)) if closes[i-1]>0]
    if len(rets)<period: return None
    m=sum(rets)/len(rets); var=sum((r-m)**2 for r in rets)/(len(rets)-1)
    return round(math.sqrt(var*252)*100,2)

def scan_stock(symbol:str) -> dict|None:
    p=_find_csv(symbol)
    if not p: return None
    rows,err=_read_rows(p,n=30)
    if err or not rows: return None
    today=date.today().isoformat()
    rows=[r for r in rows if r["date"]<today]
    if len(rows)<22: return None
    closes=[r["C"] for r in rows]
    hv5=_hv(closes,5); hv20=_hv(closes,20)
    if hv20 is None: return None
    contracting=hv5 is not None and hv5<hv20
    if   hv20>=60:  signal="Very High Vol"
    elif hv20>=40:  signal="High Vol"
    elif hv20>=20:  signal="Normal"
    elif hv20>=10:  signal="Low Vol"
    else:           signal="Very Low Vol"
    return {"symbol":symbol,"price":round(closes[-1],2),"price_date":rows[-1]["date"],
            "hv5":hv5,"hv20":hv20,"contracting":contracting,"signal":signal}

def run_hv_scanner(source="niftyfno",force=False,ttl=_DEFAULT_TTL)->dict:
    key=source; c=_hv_cache
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
