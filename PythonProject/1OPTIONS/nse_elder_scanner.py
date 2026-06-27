"""nse_elder_scanner.py — Elder Impulse System (EMA13 slope + MACD histogram)
Green  = EMA13 rising AND histogram positive/rising   (buy)
Red    = EMA13 falling AND histogram negative/falling  (sell)
Blue   = mixed signals (stand aside)
"""
from __future__ import annotations
import time
from datetime import date
from nse_pivot_scanner import _find_csv, _read_rows, load_symbol_list
from nse_indicators import ema

_elder_cache: dict = {"ts":0.0,"data":None,"key":""}
_DEFAULT_TTL = 300

def scan_stock(symbol:str) -> dict|None:
    p=_find_csv(symbol)
    if not p: return None
    rows,err=_read_rows(p,n=60)
    if err or not rows: return None
    today=date.today().isoformat()
    rows=[r for r in rows if r["date"]<today]
    if len(rows)<35: return None
    closes=[r["C"] for r in rows]
    e13=ema(closes,13); e26=ema(closes,26); e12=ema(closes,12)
    cur_e13=e13[-1]; prev_e13=next((v for v in reversed(e13[:-1]) if v is not None),None)
    if cur_e13 is None or prev_e13 is None: return None
    ema_rising=cur_e13>prev_e13
    # MACD histogram
    macd_s=[a-b for a,b in zip(e12,e26) if a is not None and b is not None]
    if len(macd_s)<9: return None
    sig_s=ema(macd_s,9)
    hist=[m-s for m,s in zip(macd_s,sig_s) if s is not None]
    if len(hist)<2: return None
    hist_rising=hist[-1]>hist[-2]; hist_pos=hist[-1]>0
    if   ema_rising and hist_pos and hist_rising:    impulse="Green"; signal="Buy Impulse"
    elif (not ema_rising) and (not hist_pos) and (not hist_rising): impulse="Red"; signal="Sell Impulse"
    else:                                            impulse="Blue"; signal="No Impulse"
    return {"symbol":symbol,"price":round(closes[-1],2),"price_date":rows[-1]["date"],
            "ema13":round(cur_e13,2),"ema_rising":ema_rising,
            "hist":round(hist[-1],4),"hist_rising":hist_rising,
            "impulse":impulse,"signal":signal}

def run_elder_scanner(source="niftyfno",force=False,ttl=_DEFAULT_TTL)->dict:
    key=source; c=_elder_cache
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
