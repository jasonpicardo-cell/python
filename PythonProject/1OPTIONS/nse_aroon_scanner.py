"""nse_aroon_scanner.py — Aroon Oscillator Scanner (period 25)"""
from __future__ import annotations
import time
from datetime import date
from nse_pivot_scanner import _find_csv, _read_rows, load_symbol_list

_aroon_cache: dict = {"ts":0.0,"data":None,"key":""}
_DEFAULT_TTL = 300

def scan_stock(symbol:str,period:int=25) -> dict|None:
    p=_find_csv(symbol)
    if not p: return None
    rows,err=_read_rows(p,n=period+5)
    if err or not rows: return None
    today=date.today().isoformat()
    rows=[r for r in rows if r["date"]<today]
    if len(rows)<period+1: return None
    w=rows[-(period+1):]
    highs=[r["H"] for r in w]; lows=[r["L"] for r in w]
    bars_hi=period-highs.index(max(highs))
    bars_lo=period-lows.index(min(lows))
    aroon_up=round((period-bars_hi)/period*100,1)
    aroon_dn=round((period-bars_lo)/period*100,1)
    osc=round(aroon_up-aroon_dn,1)
    if   aroon_up>=90 and aroon_dn<=10: signal="Strong Uptrend"
    elif aroon_dn>=90 and aroon_up<=10: signal="Strong Downtrend"
    elif osc>50:                         signal="Bullish"
    elif osc<-50:                        signal="Bearish"
    elif osc>=0:                         signal="Weak Bullish"
    else:                               signal="Weak Bearish"
    return {"symbol":symbol,"price":round(rows[-1]["C"],2),"price_date":rows[-1]["date"],
            "aroon_up":aroon_up,"aroon_dn":aroon_dn,"oscillator":osc,"signal":signal}

def run_aroon_scanner(source="niftyfno",force=False,ttl=_DEFAULT_TTL)->dict:
    key=source; c=_aroon_cache
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
