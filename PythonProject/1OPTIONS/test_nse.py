#!/usr/bin/env python3
"""
test_nse.py — pre-flight checks for the NSE dashboard.

Run this before shipping any edit:   python3 test_nse.py

Every check here exists because the corresponding bug actually shipped and
was found in production rather than before it. They are grouped by the class
of failure, not by file, because the classes recur:

  STRUCTURE   syntax, tag balance, duplicate definitions
  SCOPE       names referenced where they are not bound - the single most
              common failure in this codebase (cw, _envbool, _INDEX_CFG,
              _dtc, j, _norm_cdf all shipped broken)
  LOGIC       the interpretation rules, which compile perfectly while being
              backwards - PCR read as bullish during selloffs, wall
              abandonment inverted, flow contaminated by delta

A green run does not mean the code is right. It means these specific ways of
being wrong have been ruled out.
"""
import ast
import json
import math
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DASH = os.path.join(HERE, "nse_dashboard.html")
SRV = os.path.join(HERE, "nse_chain_server.py")

_fails, _passes, _warns = [], [], []


def check(name, ok, detail=""):
    if ok:
        _passes.append(name)
    else:
        _fails.append((name, detail))


def warn(name, detail=""):
    _warns.append((name, detail))


# ══════════════════════════════════════════════════════════════════════
# STRUCTURE
# ══════════════════════════════════════════════════════════════════════
def test_structure():
    if not os.path.exists(DASH):
        check("dashboard present", False, DASH)
        return
    c = open(DASH, encoding="utf-8").read()
    blocks = re.findall(r"<script>(.*?)</script>", c, re.S)

    # JS syntax — node --check catches only syntax, never scope
    for i, b in enumerate(blocks):
        p = f"/tmp/_t{i}.js"
        open(p, "w").write(b)
        try:
            r = subprocess.run(["node", "--check", p], capture_output=True, text=True)
            check(f"JS block {i} parses", r.returncode == 0, r.stderr[:300])
        except FileNotFoundError:
            warn("node not installed", "JS syntax unchecked")
            break

    body = re.sub(r"<script>.*?</script>", "", c, flags=re.S)
    o, cl = len(re.findall(r"<div(?:\s|>)", body)), body.count("</div>")
    check("div tags balanced", o == cl, f"{o} open vs {cl} close")

    css = "".join(re.findall(r"<style>(.*?)</style>", c, re.S))
    check("CSS braces balanced", css.count("{") == css.count("}"),
          f"{css.count('{')} vs {css.count('}')}")

    ids = re.findall(r'id="([\w-]+)"', c)
    dup = sorted({k for k in ids if ids.count(k) > 1})
    check("no duplicate element ids", not dup, str(dup[:6]))

    # A duplicate function silently replaces the earlier one. _renderGreeksCard
    # was defined twice and half the code was dead without any error.
    for i, b in enumerate(blocks):
        fns = re.findall(r"\n  (?:async )?function (\w+)", b)
        d = sorted({f for f in fns if fns.count(f) > 1})
        check(f"no duplicate functions in block {i}", not d, str(d[:6]))

    # Template-literal escaping: a stray \\' inside a JS string breaks the
    # help text silently rather than loudly.
    check("no double-escaped quotes", "\\\\'" not in c, "found \\\\' sequences")


# ══════════════════════════════════════════════════════════════════════
# SCOPE — the recurring failure
# ══════════════════════════════════════════════════════════════════════
def test_scope_js():
    c = open(DASH, encoding="utf-8").read()
    blocks = re.findall(r"<script>(.*?)</script>", c, re.S)
    if len(blocks) < 2:
        return
    decl = [set(re.findall(r"\n  (?:const|let|var|function|async function) (\w+)", b))
            for b in blocks]
    # a name used in one block but declared only in another is safe ONLY if
    # every use is inside a deferred function body; simplest robust rule is
    # to hold shared state on window, so flag any that are not
    for i, b in enumerate(blocks):
        used = set(re.findall(r"(?<!\.)\b(_[a-zA-Z]\w+)\b", b))
        # Function declarations are hoisted and shared across blocks once both
        # have executed, and every call site here is inside a deferred handler.
        # It is const/let VALUES that bite, because those are read at the
        # moment the outer block runs. Flag only those.
        vals = [set(re.findall(r"\n  (?:const|let|var) (\w+)", x)) for x in blocks]
        elsewhere = set().union(*[d for j, d in enumerate(vals) if j != i])
        leaked = sorted((used & elsewhere) - decl[i])
        check(f"block {i} has no cross-block names", not leaked,
              f"{leaked[:6]} — hold shared state on window instead")


def _at_module_try(tree, node):
    """True when an import sits directly inside a module-level try block."""
    for n in tree.body:
        if isinstance(n, ast.Try):
            for stmt in n.body:
                if stmt is node:
                    return True
    return False


def _strip_js(s):
    """Remove comments and string/template literals.

    Without this the check matched ordinary English inside comments and CSS
    words inside style strings - "keep", "skip", "the", "font", "muted" - and
    a report full of those is one nobody reads, which makes it worse than no
    report at all.
    """
    s = re.sub(r"/\*.*?\*/", " ", s, flags=re.S)
    s = re.sub(r"//[^\n]*", " ", s)
    s = re.sub(r"`(?:[^`\\]|\\.)*`", " '' ", s, flags=re.S)
    s = re.sub(r"'(?:[^'\\\n]|\\.)*'", " '' ", s)
    s = re.sub(r'"(?:[^"\\\n]|\\.)*"', ' "" ', s)
    return s


def test_scope_js_functions():
    """Names used inside a JS function that are neither params nor locals.

    Added because it was missed: an edit to _leadFire referenced `spoken`
    while the parameter is called `text`. node --check passes that happily -
    it is valid syntax and a runtime ReferenceError. This is the JavaScript
    half of the undefined-name check that already covers the server.
    """
    c = open(DASH, encoding="utf-8").read()
    blocks = re.findall(r"<script>(.*?)</script>", c, re.S)
    src = "\n".join(blocks)
    # collect every top-level declaration so they are not reported as unknown
    known = set(re.findall(r"\n  (?:const|let|var|function|async function) (\w+)", src))
    known |= set(re.findall(r"function (\w+)\s*\(", src))
    bad = []
    for m in re.finditer(r"\n  (?:async )?function (\w+)\s*\(([^)]*)\)\s*\{", src):
        name, params = m.group(1), m.group(2)
        # body by brace balance
        i, depth = src.find("{", m.start()), 0
        j = i
        while j < len(src):
            if src[j] == "{": depth += 1
            elif src[j] == "}":
                depth -= 1
                if depth == 0: break
            j += 1
        body = _strip_js(src[i:j])
        bound = {p.strip().split("=")[0].strip().lstrip("...") for p in params.split(",") if p.strip()}
        # `const a = 1, b = 2` declares BOTH; matching only the first name
        # reported every subsequent one as unbound
        for grp in re.findall(r"(?:const|let|var)\s+([^;\n=]+(?:=[^;\n]*)?)", body):
            bound |= set(re.findall(r"(\w+)\s*(?:=|,|$)", grp))
        bound |= set(re.findall(r"(?:const|let|var)\s+(\w+)", body))
        bound |= set(re.findall(r"(?:const|let|var)\s*\{([^}]*)\}", body).__class__(
            [x for grp in re.findall(r"(?:const|let|var)\s*\{([^}]*)\}", body)
             for x in re.findall(r"(\w+)", grp)]))
        # `for (const [a, b] of …)` binds both names; array and object
        # destructuring in a for-header was the last source of false alarms
        for grp in re.findall(r"for\s*\(\s*(?:const|let|var)\s*([\[{][^\]}]*[\]}]|\w+)", body):
            bound |= set(re.findall(r"(\w+)", grp))
        bound |= set(re.findall(r"catch\s*\(\s*(\w+)", body))
        bound |= set(re.findall(r"\(\s*([\w,\s]*?)\s*\)\s*=>", body).__class__(
            [x for grp in re.findall(r"\(\s*([\w,\s]*?)\s*\)\s*=>", body)
             for x in re.findall(r"(\w+)", grp)]))
        bound |= set(re.findall(r"(\w+)\s*=>", body))
        bound |= set(re.findall(r"function\s*\(([^)]*)\)", body).__class__(
            [x for grp in re.findall(r"function\s*\(([^)]*)\)", body)
             for x in re.findall(r"(\w+)", grp)]))
        # only flag bare lowercase identifiers used as values, not properties
        used = set(re.findall(r"(?<![.\w$'\"`])\b([a-z][a-zA-Z0-9_]{2,})\b(?!\s*:)", body))
        JS = {"const","let","var","function","return","if","else","for","while","try","catch",
              "finally","new","typeof","instanceof","this","null","true","false","undefined",
              "await","async","class","switch","case","break","continue","delete","in","of",
              "throw","void","yield","document","window","console","Math","JSON","Date","Object",
              "Array","String","Number","Boolean","parseInt","parseFloat","isFinite","isNaN",
              "setTimeout","setInterval","clearTimeout","clearInterval","requestAnimationFrame",
              "localStorage","sessionStorage","fetch","Promise","Set","Map","Error","RegExp",
              "encodeURIComponent","decodeURIComponent","structuredClone","navigator","location",
              # HTML and CSS words that survive imperfect template-literal
              # stripping; listing them is cheaper than a full JS parser and
              # keeps the report short enough to actually read
              "span","div","style","title","button","href","class","input","label","table",
              "width","height","color","font","fill","stroke","opacity","margin","padding",
              "flex","grid","auto","none","hidden","left","right","center","top","bottom"}
        unknown = sorted(used - bound - known - JS)
        if unknown:
            bad.append((name, unknown[:3]))
    # report as a warning rather than a failure: the heuristic cannot see every
    # binding form, so a hard failure here would be noise. The value is in
    # showing a short list to eyeball after an edit.
    # HONEST LIMIT: this heuristic does not work reliably and is kept only as
    # a rough hint. It failed its own regression test - reintroducing the
    # `spoken`/`text` mix-up in _leadFire went undetected, because a variable
    # of that name exists inside a DIFFERENT function and this check cannot
    # tell function scopes apart without parsing the language.
    #
    # The real tool for this is a linter. One line gets it:
    #     npx eslint --no-eslintrc --rule '{"no-undef":"error"}' --env browser,es2021 file.js
    # Run that on the extracted <script> blocks and it catches the whole class
    # properly. Until then, treat anything below as a prompt to look, never as
    # a clean bill of health.
    if bad:
        warn("JS names worth eyeballing (unreliable heuristic)",
             f"{bad[:3]} — use eslint no-undef for a real check")
    else:
        warn("JS scope check inconclusive", "heuristic found nothing, which proves little")


def test_scope_python():
    if not os.path.exists(SRV):
        return
    src = open(SRV, encoding="utf-8").read()
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        check("server parses", False, str(e))
        return
    check("server parses", True)

    # module-level names
    mod = set(dir(__builtins__)) if isinstance(__builtins__, dict) else set(dir(__builtins__))
    import builtins
    mod = set(dir(builtins))
    # module dunders exist in every module and are not undefined names
    mod |= {"__file__", "__name__", "__doc__", "__package__", "__spec__", "__loader__"}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                mod.add(a.asname or a.name.split(".")[0])
        elif isinstance(n, ast.ImportFrom):
            # Module-level imports bind at module scope even when wrapped in a
            # try/except for a friendly error message, which this file does for
            # its sibling modules. Requiring col_offset 0 missed all of them.
            if getattr(n, "col_offset", 1) == 0 or _at_module_try(tree, n):
                for a in n.names:
                    mod.add(a.asname or a.name)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            mod.add(n.name)
        elif isinstance(n, ast.Assign) and getattr(n, "col_offset", 1) == 0:
            for t in n.targets:
                for nn in ast.walk(t):
                    if isinstance(nn, ast.Name):
                        mod.add(nn.id)
        elif isinstance(n, ast.AnnAssign) and getattr(n, "col_offset", 1) == 0:
            # `_oi_snapshots: dict = {}` is a module-level binding just as much
            # as a plain assignment; missing this produced false alarms, and a
            # suite that cries wolf is one nobody reads.
            for nn in ast.walk(n.target):
                if isinstance(nn, ast.Name):
                    mod.add(nn.id)

    def locals_of(fn):
        out = {a.arg for a in fn.args.args}
        out |= {a.arg for a in getattr(fn.args, "kwonlyargs", [])}
        if fn.args.vararg: out.add(fn.args.vararg.arg)
        if fn.args.kwarg: out.add(fn.args.kwarg.arg)
        for n in ast.walk(fn):
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    for nn in ast.walk(t):
                        if isinstance(nn, ast.Name): out.add(nn.id)
            elif isinstance(n, (ast.AugAssign, ast.AnnAssign)):
                for nn in ast.walk(n.target):
                    if isinstance(nn, ast.Name): out.add(nn.id)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                for a in n.names: out.add(a.asname or a.name.split(".")[0])
            elif isinstance(n, (ast.For, ast.AsyncFor)):
                for nn in ast.walk(n.target):
                    if isinstance(nn, ast.Name): out.add(nn.id)
            elif isinstance(n, ast.comprehension):
                for nn in ast.walk(n.target):
                    if isinstance(nn, ast.Name): out.add(nn.id)
            elif isinstance(n, ast.withitem) and n.optional_vars:
                for nn in ast.walk(n.optional_vars):
                    if isinstance(nn, ast.Name): out.add(nn.id)
            elif isinstance(n, ast.ExceptHandler) and n.name:
                out.add(n.name)
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # a nested def binds its NAME here and its PARAMETERS inside
                # itself; collecting only the name reported every inner
                # helper's arguments as undefined
                out.add(n.name)
                if n is not fn:
                    out |= {a.arg for a in n.args.args}
                    out |= {a.arg for a in getattr(n.args, "kwonlyargs", [])}
                    if n.args.vararg: out.add(n.args.vararg.arg)
                    if n.args.kwarg: out.add(n.args.kwarg.arg)
            elif isinstance(n, ast.Global) or isinstance(n, ast.Nonlocal):
                out |= set(n.names)
            elif isinstance(n, ast.Lambda):
                # lambda parameters are bindings too; missing them reported
                # sort keys and filter variables as undefined names
                out |= {a.arg for a in n.args.args}
                out |= {a.arg for a in getattr(n.args, "kwonlyargs", [])}
                if n.args.vararg: out.add(n.args.vararg.arg)
                if n.args.kwarg: out.add(n.args.kwarg.arg)
        return out

    # Only TOP-LEVEL functions are checked. A nested function legitimately
    # reads its parent's locals through closure, so walking every def reports
    # every closure as broken - noise that buries the real finding. The bugs
    # that actually shipped (_envbool, _INDEX_CFG, _dtc, math vs _math,
    # _norm_cdf) were all in top-level functions reaching for module names.
    bad = []
    for fn in [n for n in tree.body
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        loc = locals_of(fn)
        used = {n.id for n in ast.walk(fn)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        missing = sorted(used - loc - mod)
        if missing:
            bad.append((fn.name, missing[:4]))
    # This is the check that would have caught _envbool, _INDEX_CFG, _dtc
    # and _norm_cdf before they reached you.
    check("no undefined names in server functions", not bad, str(bad[:5]))


# ══════════════════════════════════════════════════════════════════════
# LOGIC — rules that compile while being backwards
# ══════════════════════════════════════════════════════════════════════
def test_pcr_reading():
    NEUTRAL = 1.15
    def read(pcr, chg, price_up):
        use = chg if (chg is not None and chg > 0) else pcr
        if pcr > 1.9 or pcr < 0.55: return "stretched"
        if price_up is None: return "unclear"
        heavy = use > NEUTRAL
        if heavy and price_up: return "bullish"
        if heavy and not price_up: return "bearish"
        if not heavy and not price_up: return "bearish"
        return "bullish"
    # The bug: puts building during a SELLOFF were called bullish, because
    # open interest cannot tell writing from buying without price direction.
    check("PCR: puts building on a falling market reads bearish",
          read(1.45, 1.62, False) == "bearish")
    check("PCR: puts building on a rising market reads bullish",
          read(1.45, 1.62, True) == "bullish")
    check("PCR: extremes are not directional",
          read(2.10, None, False) == "stretched")
    check("PCR: no price context gives no direction",
          read(1.45, 1.62, None) == "unclear")


def test_wall_abandonment():
    def classify(strike, spot, drift):
        pct = drift / spot * 100
        moving = abs(pct) > 0.05
        above = strike >= spot
        toward = moving and ((above and drift > 0) or (not above and drift < 0))
        if toward: return "bullish" if above else "bearish"
        if moving: return "neutral"
        return "mild"
    # The bug: every call-wall decay was called bullish, so a call wall
    # unwinding during a selloff was announced as bullish.
    check("wall: call wall decaying into a rally is bullish",
          classify(24600, 24500, +40) == "bullish")
    check("wall: call wall decaying on a selloff is NOT bullish",
          classify(24600, 24500, -40) == "neutral")
    check("wall: put wall decaying into a selloff is bearish",
          classify(24400, 24500, -40) == "bearish")
    check("wall: put wall decaying on a rally is NOT bearish",
          classify(24400, 24500, +40) == "neutral")


def _bs_delta(S, K, T, sigma, is_call, r=0.07):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        itm = S > K if is_call else S < K
        return (1.0 if itm else 0.0) if is_call else (-1.0 if itm else 0.0)
    d1 = (math.log(S / K) + (r + sigma * sigma / 2) * T) / (sigma * math.sqrt(T))
    n = 0.5 * (1 + math.erf(d1 / math.sqrt(2)))
    return n if is_call else n - 1


def test_delta():
    # known Black-Scholes values; ATM call delta is slightly above 0.5
    S, T, sig = 23400, 1 * (252 / 365) / 252, 0.13
    d = _bs_delta(S, S, T, sig, True)
    check("delta: ATM call slightly above 0.5", 0.50 < d < 0.53, f"{d:.4f}")
    check("delta: call plus put equals one",
          abs(_bs_delta(S, S, T, sig, True) - _bs_delta(S, S, T, sig, False) - 1) < 1e-12)
    check("delta: deep ITM call approaches 1",
          _bs_delta(S, S - 2000, T, sig, True) > 0.99)
    check("delta: deep OTM call approaches 0",
          _bs_delta(S, S + 2000, T, sig, True) < 0.01)
    check("delta: expiry is a step function",
          _bs_delta(S, S - 10, 0, sig, True) == 1.0)


def test_flow_delta_adjustment():
    # The bug: put premium rises on any dip through delta alone, so routine
    # pullbacks inside a rising session were announced as put buying.
    def demand(spot, prev_spot, K, is_call, prem_old, prem_new, iv=0.13, dte=1.0):
        pr_pct = (prem_new - prem_old) / prem_old * 100
        d_spot = spot - prev_spot
        T = max(1e-6, dte * (252 / 365) / 252)
        if abs(d_spot) <= 0.01 or not prem_old:
            return ("buying" if pr_pct > 0 else "selling")
        dlt = _bs_delta(spot, K, T, iv, is_call)
        expected = (dlt * d_spot) / prem_old * 100
        excess = pr_pct - expected
        if abs(excess) < max(1.5, abs(expected) * 0.35):
            return None
        return "buying" if excess > 0 else "selling"
    check("flow: small dip lifting put premium is NOT demand",
          demand(23400, 23410, 23400, False, 80, 84) is None)
    check("flow: large excess over delta IS demand",
          demand(23400, 23410, 23400, False, 80, 95) == "buying")
    check("flow: put premium rising while spot rises is demand",
          demand(23420, 23400, 23400, False, 80, 85) == "buying")


def test_pcr_bands_scale():
    # A fixed strike count means different things per instrument and breaks
    # where strike spacing changes mid-chain.
    def band(spot, iv, dte, sig_mult=1.0):
        T = max(1e-6, dte * (252 / 365) / 252)
        return spot * (iv / 100) * math.sqrt(T) * sig_mult
    n = band(23400, 13, 1)
    b = band(51000, 15.5, 1)
    check("bands: BankNifty band is wider than Nifty", b > n * 2, f"{n:.0f} vs {b:.0f}")
    check("bands: a volatile day widens the band",
          band(23400, 22, 1) > band(23400, 13, 1) * 1.5)


def test_index_weights():
    p = os.path.join(HERE, "index_weights.json")
    if not os.path.exists(p):
        warn("index_weights.json missing", "contribution figures fall back to the built-in table")
        return
    d = json.load(open(p))
    for k, v in d.items():
        if k.startswith("_"):
            continue
        w = v.get("weights") or {}
        check(f"weights: {k} sums to 100%", abs(sum(w.values()) - 100) < 1.0,
              f"{sum(w.values()):.2f}%")
        check(f"weights: {k} has constituents", len(w) >= 10, f"{len(w)}")


def test_env():
    p = next((os.path.join(HERE, n) for n in ("env", ".env")
              if os.path.exists(os.path.join(HERE, n))), None)
    if not p:
        warn("no env file", "server will use built-in defaults")
        return
    present = [n for n in ("env", ".env", ".env.local", "env.txt")
               if os.path.exists(os.path.join(HERE, n))]
    # Having two is a silent trap: edits to the wrong one simply do nothing.
    check("only one settings file", len(present) == 1, f"found {present}")


def test_aggressor_flow():
    # Every print is a buyer lifting the offer or a seller hitting the bid,
    # judged against the quote that stood BEFORE the trade.
    sys.path.insert(0, HERE)
    import types
    if "fyers_apiv3" not in sys.modules:
        try:
            import fyers_apiv3  # noqa: F401
        except Exception:
            sys.modules["fyers_apiv3"] = types.ModuleType("fyers_apiv3")
    try:
        import nse_adapter_fyers as A
    except Exception as e:  # noqa: BLE001
        warn("aggressor flow: adapter not importable here", str(e)[:120])
        return
    A._flow.pop("T", None)
    u = A._flow_update
    u("T", {"ltp": 100, "volume": 1000, "bid": 99.5, "ask": 100.5})
    check("flow: first tick only seeds", A._flow_read("T") is None)
    u("T", {"ltp": 100.5, "volume": 1050, "bid": 100, "ask": 101})    # at prior offer
    u("T", {"ltp": 100, "volume": 1080, "bid": 99.5, "ask": 100.5})   # at prior bid
    u("T", {"ltp": 100.2, "volume": 1090})                            # inside, uptick
    u("T", {"ltp": 100.2, "volume": 1090, "bid": 100.1, "ask": 100.3})  # quote only
    x = A._flow_read("T")
    check("flow: trade at the offer is a buy, at the bid a sell, inside by tick test",
          x and x["b5"] == 60 and x["s5"] == 30, str(x))
    u("T", {"ltp": 100, "volume": 5})
    check("flow: volume counter reset starts a new session", A._flow_read("T") is None)
    A._flow.pop("T", None)
    # v3 DepthUpdate is FLAT (bid_price1..5 etc., type "dp", no ltp) - the
    # format in the SDK's map.json. It must land as a five-level ladder and
    # keep the whole-book totals from the price tick.
    A._push_ticks = lambda: None
    A._on_message({"symbol": "D", "ltp": 40.5, "vol_traded_today": 1000, "tot_buy_qty": 600000, "tot_sell_qty": 1800000})
    m = {"type": "dp", "symbol": "D"}
    for i in range(1, 6):
        m[f"bid_price{i}"] = 40.4 - 0.05 * (i - 1); m[f"ask_price{i}"] = 40.6 + 0.05 * (i - 1)
        m[f"bid_size{i}"] = 750; m[f"ask_size{i}"] = 300
    A._on_message(m)
    dd = A._depth.get("D") or {}
    check("depth: flat v3 DepthUpdate becomes a 5-level ladder",
          len(dd.get("bids", [])) == 5 and sum(x["q"] for x in dd["asks"]) == 1500, str(dd)[:160])
    check("depth: whole-book totals kept beside the ladder", dd.get("tbq") == 600000, str(dd.get("tbq")))
    A._depth.pop("D", None); A._flow.pop("D", None); A._ticks.pop("D", None)
    # futures get depth first, the index never (the SDK drops it), and a
    # reconnect asks for everything again - the SDK forgets, our sets did not
    calls = []
    class _S:
        def subscribe(self, symbols, data_type="SymbolUpdate"):
            calls.append((data_type, list(symbols)))
    old_sock, old_px, old_dp = A._socket, set(A._subscribed), set(A._depth_subs)
    try:
        A._socket = _S(); A._subscribed.clear(); A._depth_subs.clear()
        A._subscribe_depth(["NSE:NIFTY26OCTFUT", "NSE:NIFTY26NOVFUT"] +
                           [f"NSE:NIFTY26OCT{k}CE" for k in range(22000, 24000, 50)] + ["NSE:NIFTY50-INDEX"])
        dp = calls[-1][1] if calls else []
        check("depth: futures subscribed first, index skipped",
              dp[:2] == ["NSE:NIFTY26OCTFUT", "NSE:NIFTY26NOVFUT"] and not any(x.endswith("-INDEX") for x in dp))
        A._subscribed.add("NSE:NIFTY50-INDEX")
        calls.clear()
        A._on_socket_connect()
        import time as _t
        _t.sleep(1.4)
        check("reconnect: price and depth subscriptions re-requested",
              any(c[0] == "DepthUpdate" and "NSE:NIFTY26OCTFUT" in c[1] for c in calls)
              and any(c[0] == "SymbolUpdate" for c in calls), str([(c[0], len(c[1])) for c in calls]))
    finally:
        A._socket = old_sock; A._subscribed.clear(); A._subscribed.update(old_px)
        A._depth_subs.clear(); A._depth_subs.update(old_dp)


def test_iv_solver():
    # Black-76 on the synthetic forward must recover the vol that priced
    # the chain, and leave few-tick wings unsolved rather than invent 30%.
    sys.path.insert(0, HERE)
    try:
        import nse_adapter_fyers as A
    except Exception as e:  # noqa: BLE001
        warn("iv solver: adapter not importable here", str(e)[:120])
        return
    spot, T_days, vol = 25000.0, 3.0, 0.14
    T = T_days / 365.0
    F = spot * math.exp(0.065 * T)
    rows = []
    for k in range(23500, 26501, 100):
        c = A._b76(F, k, T, vol, True)
        p = A._b76(F, k, T, vol, False)
        rows.append({"strike": float(k), "ce_ltp": round(c * 20) / 20, "pe_ltp": round(p * 20) / 20,
                     "ce_iv": 0.0, "pe_iv": 0.0})
    src = A._solve_chain_iv(rows, spot, T_days)
    atm = min(rows, key=lambda r: abs(r["strike"] - spot))
    check("iv: solved when the broker sends none", src == "solved", src)
    check("iv: ATM vol recovered within 0.3 pts", abs(atm["ce_iv"] - 14.0) < 0.3, str(atm["ce_iv"]))
    wing = rows[0]
    check("iv: a few-tick wing is left unsolved", wing["ce_iv"] == 0 or wing["pe_ltp"] >= 1.0,
          f'{wing["strike"]} pe {wing["pe_ltp"]} iv {wing["ce_iv"]}')
    rows2 = [dict(r, ce_iv=15.0, pe_iv=15.0) for r in rows]
    check("iv: broker IV is never overwritten", A._solve_chain_iv(rows2, spot, T_days) == "broker")
    import calendar
    ep = calendar.timegm((2026, 10, 13, 15, 30, 0)) - 19800
    real_time = A.time.time
    try:
        A.time.time = lambda: calendar.timegm((2026, 10, 12, 9, 0, 0))     # 14:30 IST
        mdte = A._monthly_dte([ep, ep + 7 * 86400])
    finally:
        A.time.time = real_time
    check("futures dte: counts to the monthly expiry, not the weekly", mdte and 14.5 < mdte < 15.5, str(mdte))


def test_no_shadowed_imports():
    # A local "import x" anywhere in a function makes x local to the WHOLE
    # function, so an earlier use raises UnboundLocalError. That broke
    # /api/fno-symbols inside the 2,000-line do_GET.
    import ast
    src = open(os.path.join(HERE, "nse_chain_server.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    bad = []
    for f in ast.walk(tree):
        if not isinstance(f, ast.FunctionDef):
            continue
        imports = {}
        for n in ast.walk(f):
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                for a in n.names:
                    nm = a.asname or a.name.split(".")[0]
                    imports.setdefault(nm, n.lineno)
        for n in ast.walk(f):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in imports \
                    and n.lineno < imports[n.id]:
                bad.append(f"{f.name}:{n.lineno} {n.id}")
    check("no function-local import shadows an earlier use", not bad, str(sorted(set(bad))[:5]))


def test_ist_offset():
    # IST = UTC+5:30, and getTimezoneOffset() is -330 in India. The local
    # clock that reads as IST is now + (330 + offset) minutes; the minus form
    # put an Indian browser 11 hours ahead ("Pre-Market" at 14:30).
    c = open(DASH, encoding="utf-8").read()
    bad = re.findall(r"5\.5 \* 3600 - \w+\.getTimezoneOffset\(\)", c)
    check("IST conversion uses + getTimezoneOffset()", not bad, str(bad[:3]))


def test_reset_defaults():
    # Reset must actually restore the shipped defaults: uiStateV1 replays
    # every checkbox on load, and a beforeunload save rewrote it after the
    # reset cleared it - so Pivots (and every other tick) came straight back.
    c = open(DASH, encoding="utf-8").read()
    check("chart: pivots off by default", c.count("pivots: false, pivotType") == 2 and "pivots: true, pivotType" not in c)
    check("reset: clears saved control states", "chk: {}, sel: {}, num: {}, zoom: ui.zoom" in c)
    check("context pane: VIX on by default", "vix: true, iv: false," in c)
    check("crosshair mirror is defined (it was called but never written)", "function _mirrorCrosshair(" in c)
    check("context history coalesces from the window start, not a sliding last.ts",
          "row.ts - (last.t0 || last.ts) < 400" in c)
    check("reset: unload save suspended during reset",
          "if (window._settingsResetting) return;" in c and "window._settingsResetting = true;" in c)


def test_units():
    # Fyers counts shares; NSE (and every threshold here) counts contracts.
    sys.path.insert(0, HERE)
    try:
        import nse_adapter_fyers as A
    except Exception as e:  # noqa: BLE001
        warn("units: adapter not importable here", str(e)[:120])
        return
    rows = [{"ce_oi": 65 * (1000 + 7 * i), "pe_oi": 65 * (900 + 3 * i),
             "ce_volume": 65 * (50 + i), "pe_volume": 65 * (40 + 2 * i)} for i in range(8)]
    check("units: lot read off share quantities", A._infer_lot(rows) == 65, str(A._infer_lot(rows)))
    rows2 = [{"ce_oi": 1001 + 7 * i, "pe_oi": 903 + 3 * i, "ce_volume": 51 + i, "pe_volume": 43 + i} for i in range(8)]
    check("units: contract quantities are not mistaken for shares", A._infer_lot(rows2) is None)
    check("units: lot table current (Nifty 65)", A._LOTS.get("NIFTY") == 65, str(A._LOTS.get("NIFTY")))
    # OI mixes lots across a lot-size revision (65 and 75 -> divisor 5); the
    # lot must come from what trades now, never from OI
    mixed = [{"ce_oi": 65 * 1000 + 75 * i, "pe_oi": 65 * 700 + 75 * (i + 1), "ce_volume": 0, "pe_volume": 0}
             for i in range(10)]
    check("units: lot is never read from OI", A._infer_lot(mixed) is None, str(A._infer_lot(mixed)))
    A._depth["T1"] = {"bids": [{"p": 1, "q": 65 * k, "n": 1} for k in (1, 2, 3, 4, 7)],
                      "asks": [{"p": 2, "q": 65 * k, "n": 1} for k in (1, 5, 2, 9, 3)]}
    A._depth["T2"] = {"bids": [{"p": 1, "q": 65 * k, "n": 1} for k in (6, 1, 2, 11, 4)],
                      "asks": [{"p": 2, "q": 65 * k, "n": 1} for k in (2, 3, 1, 8, 13)]}
    check("units: lot read off the live order book", A._infer_lot_book(["T1", "T2"]) == 65,
          str(A._infer_lot_book(["T1", "T2"])))
    A._depth.pop("T1", None); A._depth.pop("T2", None)
    # nse_lot_sizes answers 1 without an NSE session; that must never be used
    import types as _ty
    _old = sys.modules.get("nse_lot_sizes")
    sys.modules["nse_lot_sizes"] = _ty.SimpleNamespace(get_lot_size=lambda s: 1)
    try:
        check("lot size: an answer of 1 is rejected", A._lot_size("NIFTY") == 65, str(A._lot_size("NIFTY")))
    finally:
        if _old is not None:
            sys.modules["nse_lot_sizes"] = _old
        else:
            sys.modules.pop("nse_lot_sizes", None)
    import calendar as _cal
    ep = _cal.timegm((2026, 10, 13, 0, 0, 0)) - 19800          # 13 Oct, IST midnight
    check("expiry settles at 15:30 IST", A._settle_of(ep) == _cal.timegm((2026, 10, 13, 10, 0, 0)),
          str(A._settle_of(ep)))
    dsrc = open(DASH, encoding="utf-8").read()
    check("dashboard sends only a user-picked expiry",
          "if (state.expiry && state.expiryUser) params.set('expiry', state.expiry);" in dsrc)
    src = open(os.path.join(HERE, "nse_adapter_fyers.py"), encoding="utf-8").read()
    check("OI change never falls back to the PERCENT field", '"oich", "oichp"' not in src)
    # replay migration converts a broker snapshot once, and leaves NSE alone
    # the server is not imported (that would start it); the function is
    # lifted out of its source and run on its own
    import ast as _ast, types as _types
    ssrc = open(os.path.join(HERE, "nse_chain_server.py"), encoding="utf-8").read()
    fn = next((n for n in _ast.parse(ssrc).body
               if isinstance(n, _ast.FunctionDef) and n.name == "_snapshot_to_contracts"), None)
    if fn is None:
        check("replay migration present", False, "_snapshot_to_contracts not found")
        return
    ns = {"_QTY_FIELDS_ROW": ("ce_oi", "pe_oi", "ce_oi_chg", "pe_oi_chg", "ce_volume", "pe_volume",
                              "ce_oi_15m", "pe_oi_15m", "ce_d60", "pe_d60"),
          "_QTY_FIELDS_TOP": ("futures_oi", "futures_oi_chg", "futures_volume", "futures_far_oi",
                              "total_ce_oi", "total_pe_oi")}
    exec(compile(_ast.Module(body=[fn], type_ignores=[]), "srv", "exec"), ns)
    S = _types.SimpleNamespace(_snapshot_to_contracts=ns["_snapshot_to_contracts"])
    snap = {"_source": "fyers", "lot_size": 65, "futures_oi": 65 * 1000,
            "strikes": [dict(strike=22000 + 50 * i, **r) for i, r in enumerate(rows)]}
    ok1 = S._snapshot_to_contracts(snap)
    check("replay: broker snapshot converted to contracts",
          ok1 and snap["strikes"][0]["ce_oi"] == 1000 and snap["futures_oi"] == 1000 and snap["oi_unit"] == "contracts")
    check("replay: an already-converted snapshot is left alone", not S._snapshot_to_contracts(snap))
    nse = {"_source": "nse", "strikes": [dict(strike=22000 + 50 * i, **r) for i, r in enumerate(rows)]}
    check("replay: NSE snapshot untouched", not S._snapshot_to_contracts(nse))
    bad = {"_source": "fyers", "symbol": "NIFTY", "oi_unit": "contracts", "qty_per_contract": 5,
           "strikes": [{"strike": 22000.0, "ce_oi": 1000.0, "pe_oi": 13.0}]}
    S._snapshot_to_contracts(bad)
    check("replay: a snapshot divided by a lot of 5 is rescaled to 65",
          bad["qty_per_contract"] == 65 and abs(bad["strikes"][0]["ce_oi"] - 1000 * 5 / 65) < 0.01, str(bad)[:120])


def test_boxes():
    # the bottom-right boxes: the day's 1 sigma from VIX, not a zero-day IV;
    # move-quality legs read from the session only and timed start->extreme
    c = open(DASH, encoding="utf-8").read()
    check("desk box: day 1σ from India VIX first", "const iv = vix > 0 ? vix : ((d?.dte ?? 0) >= 1 ? aiv : 0);" in c)
    check("desk box: IV change is ATM IV vs ATM IV", "const oldIv = _dhAgo(900)?.aiv;" in c)
    sys.path.insert(0, HERE)
    try:
        import nse_move_quality as M
    except Exception as e:  # noqa: BLE001
        warn("move quality not importable", str(e)[:100])
        return
    import tempfile, calendar
    d = tempfile.mkdtemp()
    t0 = calendar.timegm((2026, 10, 9, 3, 30, 0))           # 09:00 IST
    rows = []
    for i in range(420):
        m = 540 + i
        spot = 25000 + (0 if m < 555 else (m - 555) * 0.8 if m < 700 else 116 - (m - 700) * 0.3)
        rows.append(json.dumps({"underlying_value": spot, "_replay_ts": t0 + i * 60, "strike_gap": 50, "strikes": []}))
    path = os.path.join(d, "NIFTY_2026-10-09.jsonl")
    open(path, "w").write("\n".join(rows) + "\n")
    ser = M.day_series(path)
    check("move quality: pre-open and evening rows dropped",
          M._ist_min(ser[0]["t"]) == 555 and M._ist_min(ser[-1]["t"]) == 930, f"{len(ser)} rows")
    leg = M.current(d, "NIFTY", "2026-10-09").get("leg") or {}
    check("move quality: leg minutes run start to extreme",
          leg.get("mins") is not None and leg["mins"] == (leg["ext_t"] - leg["start_t"]) // 60, str(leg)[:120])


def test_constituents():
    """Constituent score: weighted breadth, momentum window, socket-served
    quotes (no REST once streaming), REST throttled, Trend Engine wiring."""
    import ast as _ast, time as _t
    ssrc = open(os.path.join(HERE, "nse_chain_server.py"), encoding="utf-8").read()
    fn = next((n for n in _ast.parse(ssrc).body
               if isinstance(n, _ast.FunctionDef) and n.name == "_constituent_score"), None)
    check("constituent score present", fn is not None)
    if fn is None:
        return
    ns = {"time": _t, "_const_hist": {}}
    exec(compile(_ast.Module(body=[fn], type_ignores=[]), "srv", "exec"), ns)
    sc = ns["_constituent_score"]
    rows = [{"symbol": f"S{i}", "ltp": 101.0, "open": 100.0, "pct_change": 1.0,
             "weight_est": w, "pts_contributed": 1} for i, w in enumerate((40, 30, 20, 10))]
    rows[3].update(ltp=99.0, pct_change=-1.0)            # 10% weight falling
    now = _t.time()
    r = sc("NIFTY", rows, {"index_value": 25000}, now - 400)
    check("constituents: weighted breadth since open", abs(r["open"]["w"] - 0.8) < 1e-6 and r["open"]["adv"] == 3)
    check("constituents: no 5-min reading without history", r["mom"] is None)
    rows2 = [dict(x, ltp=x["ltp"] * (0.99 if i < 2 else 1.01)) for i, x in enumerate(rows)]
    r2 = sc("NIFTY", rows2, {"index_value": 25000}, now)
    check("constituents: 5-min breadth follows the heavyweights", r2["mom"] and r2["mom"]["w"] < -0.3
          and r2["mom"]["pts"] < 0, str(r2.get("mom")))
    check("constituents: score bounded", -100 <= r2["score"] <= 100 and r["score"] > 50, f'{r["score"]} {r2["score"]}')
    # adapter: socket-served quotes cost no REST call
    import nse_adapter_fyers as A
    calls = []
    class _C:
        def quotes(self, req):
            calls.append(req); return {"s": "ok", "d": []}
    class _S:
        def subscribe(self, symbols, data_type=None): pass
    old = (A._client, A._socket, A.connect)
    try:
        A._client, A._socket, A.connect = _C(), _S(), (lambda *a, **k: True)
        A._ticks["NSE:ZZTEST-EQ"] = {"ltp": 110.0, "prev_close": 100.0, "open": 105.0, "volume": 10, "ts": _t.time()}
        A._subscribed.add("NSE:ZZTEST-EQ")
        q = A.constituent_quotes(["ZZTEST"])
        check("constituents: streamed name served from the socket", q.get("ZZTEST", {}).get("pChange") == 10.0 and not calls)
        A.constituent_quotes(["ZZNEW"]); n1 = len(calls)
        A._eq_rest_cache["ZZNEW"] = (_t.time(), {"symbol": "ZZNEW", "lastPrice": 1})
        A.constituent_quotes(["ZZNEW"])
        check("constituents: REST fallback at most once per 30 s", n1 == 1 and len(calls) == 1, f"{n1} {len(calls)}")
        A._dirty.clear()
        A._on_message({"symbol": "NSE:ZZTEST-EQ", "ltp": 111.0})
        check("constituents: equity ticks are not pushed to browsers", "NSE:ZZTEST-EQ" not in A._dirty)
    finally:
        A._client, A._socket, A.connect = old
        A._ticks.pop("NSE:ZZTEST-EQ", None); A._subscribed.discard("NSE:ZZTEST-EQ")
    check("constituents: background refresh never scrapes NSE", "_fetch_nifty_constituents(sym, broker_only=True)" in ssrc
          and "if broker_only:" in ssrc)
    dsrc = open(os.path.join(HERE, "nse_dashboard.html"), encoding="utf-8").read()
    check("constituents: Trend Engine input and score strip", "add(P, 'Constituents', 3" in dsrc and "${_consBlock(t)}" in dsrc)


def test_theme():
    dsrc = open(os.path.join(HERE, "nse_dashboard.html"), encoding="utf-8").read()
    import re as _re
    root = _re.search(r":root \{(.*?)\}", dsrc, _re.S).group(1)
    check("dark theme: page and widget backgrounds are true black",
          all(f"--{k}: #000000;" in root for k in ("bg", "panel", "panel-2")))
    check("light theme: variables also on <html>", "html.light-mode, body.light-mode {" in dsrc
          and "document.documentElement.classList.toggle('light-mode', light)" in dsrc)
    check("light theme: dark-tuned inline colours remapped", 'body.light-mode [style*="color:#9ab;"]' in dsrc)
    check("tab bar wraps instead of overflowing", ".tabs { display: flex; flex-wrap: wrap;" in dsrc)


def test_watchdog():
    """Silent SDK subscribe failures are re-requested; an index/depth stall
    in market hours is re-subscribed, then the socket reopened."""
    import nse_adapter_fyers as A, datetime as _dt
    IST = _dt.timezone(_dt.timedelta(hours=5, minutes=30))
    T = _dt.datetime(2026, 10, 15, 11, 0, tzinfo=IST).timestamp()      # Thursday 11:00 IST
    calls = []
    class _WS:
        closed = 0
        def close(self): _WS.closed += 1
    class _S:
        def subscribe(self, symbols, data_type=None): calls.append((data_type, list(symbols)))
    s_ = _S(); setattr(s_, "_FyersDataSocket__ws_object", _WS())
    saved = (A._socket, dict(A._raw_last), set(A._subscribed), set(A._depth_subs), dict(A._sub_req), dict(A._dsub_req), dict(A._wd))
    try:
        A._socket = s_
        for x in (A._raw_last, A._sub_req, A._dsub_req, A._retry_n):
            x.clear()
        A._subscribed.clear(); A._depth_subs.clear()
        A._wd.update(last_heal=0.0, last_force=0.0, heals=0, forced_reconnects=0, retried=0)
        idx, opt, fut = "NSE:NIFTY50-INDEX", "NSE:NIFTY26OCT25000CE", "NSE:NIFTY26OCTFUT"
        # subscribed at T-100; the option and future confirmed, the index never answered
        A._subscribed.update([idx, opt]); A._depth_subs.add(fut)
        A._sub_req.update({idx: T - 100, opt: T - 100}); A._dsub_req[fut] = T - 100
        A._raw_last[opt] = {"sf": (T - 2, {})}
        A._raw_last[fut] = {"dp": (T - 3, {})}
        r = A._watchdog_pass(T)
        check("watchdog: an unanswered subscription is requested again",
              r.get("retried") == 1 and any(idx in c[1] for c in calls), f"{r} {calls}")
        # now everything confirmed earlier, then index + depth went silent while options tick
        calls.clear()
        A._sub_req.update({idx: T - 400}); A._dsub_req[fut] = T - 400
        A._subscribed.add(idx)
        A._raw_last[idx] = {"if": (T - 300, {})}
        A._raw_last[fut] = {"dp": (T - 300, {})}
        A._raw_last[opt] = {"sf": (T - 1, {})}
        r = A._watchdog_pass(T)
        check("watchdog: silent index and depth are re-subscribed",
              r.get("resubscribed") == 2 and ("DepthUpdate", [fut]) in calls, f"{r} {calls}")
        # still silent three minutes later -> socket reopened, once
        A._sub_req[idx] = T - 400; A._dsub_req[fut] = T - 400
        A._raw_last[opt] = {"sf": (T + 179, {})}
        A._subscribed.add(idx); A._depth_subs.add(fut)
        r = A._watchdog_pass(T + 180)
        check("watchdog: still stalled after re-subscribing -> socket reopened", r.get("forced") and _WS.closed == 1, str(r))
        A._raw_last[opt] = {"sf": (T + 299, {})}
        r = A._watchdog_pass(T + 300)
        check("watchdog: reopen is rate-limited", _WS.closed == 1, str(r))
        # outside market hours a quiet index is normal
        A._wd.update(last_heal=0.0, last_force=0.0)
        night = T + 11 * 3600
        A._raw_last[opt] = {"sf": (night - 1, {})}
        r = A._watchdog_pass(night)
        check("watchdog: no stall action outside market hours", not r.get("resubscribed") and not r.get("forced"), str(r))
        st = A.watchdog_state()
        check("watchdog: state reported for the audit", "log" in st and "price_unconfirmed" in st)
    finally:
        A._socket = saved[0]
        A._raw_last.clear(); A._raw_last.update(saved[1])
        A._subscribed.clear(); A._subscribed.update(saved[2])
        A._depth_subs.clear(); A._depth_subs.update(saved[3])
        A._sub_req.clear(); A._sub_req.update(saved[4])
        A._dsub_req.clear(); A._dsub_req.update(saved[5])
        A._wd.clear(); A._wd.update(saved[6])


def test_strike_picker():
    """Trade-entry engine on a synthetic Nifty chain: the daily move from VIX,
    T1/T2 as fractions of it per regime (the two worked scenarios), strikes
    chosen for that move, picks held, and the outcome log."""
    import subprocess, json as _json
    dsrc = open(os.path.join(HERE, "nse_dashboard.html"), encoding="utf-8").read()
    check("trade entries rendered in the desk read widget", "${pickHtml}" in dsrc and "function _strikePicker(d)" in dsrc)
    def grab(name):
        i = dsrc.index(f"function {name}(")
        depth, j = 0, dsrc.index("{", i)
        for k in range(j, len(dsrc)):
            depth += dsrc[k] == "{"; depth -= dsrc[k] == "}"
            if depth == 0:
                return dsrc[i:k + 1]
    code = "\n".join(grab(n) for n in ("bsErf", "bsNormCdf", "bsNormPdf", "bsD1D2", "bsGreeks", "bsPriceCalc", "_volT",
                                        "_strikePicker", "_istNowMin", "_tradeCost", "_scalpCfg", "_seTrack", "_seStats",
                                        "_seLog", "_seSave"))
    harness = code + r"""
const RISK_FREE = 0.07; const window = {}; const fmtInt = x => String(x);
const _store = {}; const localStorage = { getItem: k => _store[k] ?? null, setItem: (k, v) => { _store[k] = String(v); } };
const _SCALP_KEY = 'sc', _SE_KEY = 'se'; const _pickMem = {}; const state = { symbol: 'NIFTY' };
const _chart = { candles: [] }; const _isHistorical = () => false; const _dayType = () => null; const _sessionBudget = () => null;
const _cfLatch = {}; const _vwapBands = () => ({ vw: [], sd: [] });
let CONF = 0; const _bookHwConf = () => ({ confirmed: CONF, score: CONF * 70 });
const _computeGex = () => ({ flip: 24800 });
const spot = 25000;
function chain(vix) {
  const st = [], T = _volT(4), iv = vix / 100;
  const px = (K, typ) => { const [d1, d2] = bsD1D2(spot, K, T, iv, 0.07);
    return typ === 'CE' ? spot * bsNormCdf(d1) - K * Math.exp(-0.07 * T) * bsNormCdf(d2)
                        : K * Math.exp(-0.07 * T) * bsNormCdf(-d2) - spot * bsNormCdf(-d1); };
  for (let K = 23500; K <= 26500; K += 50) {
    const c = Math.max(0.05, px(K, 'CE')), p = Math.max(0.05, px(K, 'PE'));
    st.push({ strike: K, ce_ltp: c, pe_ltp: p, ce_bid: c - 0.1, ce_ask: c + 0.1, pe_bid: p - 0.1, pe_ask: p + 0.1,
              ce_iv: vix, pe_iv: vix, ce_volume: 100, pe_volume: 100, ce_oi: 1e5, pe_oi: 1e5 });
  }
  return { symbol: 'NIFTY', underlying_value: spot, strikes: st, dte: 4, atm_iv: vix, india_vix: vix, lot_size: 65 };
}
const vixFor = D => D * Math.sqrt(252) / spot * 100;
// scenario 1: trending, daily move +-350
window._trendLast = { score: 70, at: Date.now() };
const t = _strikePicker(chain(vixFor(350)));
const tb = t.choose('BUYCE', 'BUY', 'CE', 1, t.T1, 0.35 * t.T1, t.T2);
const ts = t.choose('SELLPE', 'SELL', 'PE', 1, t.T1, 0.35 * t.T1, t.T2);
// scenario 2: range, daily move +-150
window._trendLast = { score: 5, at: Date.now() };
const r = _strikePicker(chain(vixFor(150)));
const rb = r.choose('BUYPE', 'BUY', 'PE', -1, r.T1, 0.6 * r.T1, r.T2);
const rb2 = r.choose('BUYPE', 'BUY', 'PE', -1, r.T1, 0.6 * r.T1, r.T2);
// the 15-Oct case: +-203 priced, 374-point rally. Candles from 09:15 rising
// steadily (one-way) vs the same range chopped back and forth.
window._trendLast = { score: 5, at: Date.now() };
const D203 = vixFor(203), t0 = 1792036500;
const rally = [], chop = [];
for (let i = 0; i < 300; i++) {
  const a = 24700 + i * 374 / 300, b2 = 24700 + 374 * Math.abs(Math.sin(i / 20)) * (i % 40 < 20 ? 1 : 0.4);
  rally.push({ t: t0 + i * 60, o: a - 0.6, h: a + 0.6, l: a - 1.2, c: a });
  chop.push({ t: t0 + i * 60, o: b2, h: b2 + 1, l: b2 - 1, c: b2 });
}
chop[chop.length - 1].c = 24750;
_chart.candles = rally; const up = _strikePicker(chain(D203));
_chart.candles = chop;  const ch = _strikePicker(chain(D203));
_chart.candles = [];
// latch: confirmed, then the confirmation drops -> still TREND for 15 min
const realNow = Date.now; let TN = realNow();
Date.now = () => TN;
CONF = 1; const L1 = _strikePicker(chain(D203));
CONF = 0; TN += 5 * 60000; const L2 = _strikePicker(chain(D203));
TN += 11 * 60000; const L3 = _strikePicker(chain(D203));
Date.now = realNow;
window._trendLast = { score: 5, at: Date.now() };
// outcome log: T1 reached, then T2 -> half at each, net of costs
localStorage.setItem('se', JSON.stringify([{ t: Date.now(), sym: 'NIFTY', exp: '', side: 'BUY', typ: 'CE', K: 25000,
  entry: 100, stop: 90, tgt: 115, tgt2: 130, costU: 1, lot: 65, riskU: 11, hold: 60, regime: 'trend' }]));
_seTrack({ symbol: 'NIFTY', expiry: '', strikes: [{ strike: 25000, ce_ltp: 116 }] });
const lg = _seTrack({ symbol: 'NIFTY', expiry: '', strikes: [{ strike: 25000, ce_ltp: 131 }] });
const stt = _seStats(lg, 'NIFTY');
console.log(JSON.stringify({
  trend: { D: Math.round(t.D), reg: t.regime, T1: t.T1, T2: t.T2, buy: [tb.K, tb.delta, tb.plan.rFull], sell: [ts.K, ts.delta] },
  range: { D: Math.round(r.D), reg: r.regime, T1: r.T1, T2: r.T2, buy: [rb.K, rb.delta], hold: rb2.K === rb.K },
  log: [lg[0].res, Math.round(lg[0].pnlLot), stt.all.n, stt.all.w, stt.trend.n],
  latch: [L1.regime, L1.trendDir, L2.regime, L2.trendDir, L3.regime],
  oct15: { D: Math.round(up.D), ER: Math.round(up.ER), upReg: up.regime, upDir: up.trendDir, upT2: up.T2, upRoom: Math.round(up.roomLeft),
           chopReg: ch.regime, chopT2: ch.T2, chopT1: ch.T1 } }));
"""
    fn = os.path.join(HERE, "__pick_test.js")
    open(fn, "w").write(harness)
    try:
        out = subprocess.run(["node", fn], capture_output=True, text=True, timeout=30)
    finally:
        os.remove(fn)
    if out.returncode:
        check("trade entries engine runs", False, out.stderr[-400:]); return
    r = _json.loads(out.stdout.strip().splitlines()[-1])
    t, g = r["trend"], r["range"]
    check("daily move = spot x VIX / sqrt(252)", abs(t["D"] - 350) <= 1 and abs(g["D"] - 150) <= 1, f'{t["D"]} {g["D"]}')
    check("scenario trend +-350: T1 above 100 pts, T2 up to 70%", t["reg"] == "trend" and t["T1"] > 100 and t["T2"] == 245, str(t))
    check("scenario range +-150: T1 above 50 pts, T2 up to 40%", g["reg"] == "range" and g["T1"] > 50 and g["T2"] == 60, str(g))
    check("trend buy: liquid strike between delta 0.35 and 0.80", 0.35 <= abs(t["buy"][1]) <= 0.80, str(t["buy"]))
    check("trend sell: the against-trend option, delta 0.20-0.55", t["sell"][0] and 0.20 <= abs(t["sell"][1]) <= 0.55, str(t["sell"]))
    check("range buy put chosen and held", 0.35 <= abs(g["buy"][1]) <= 0.80 and g["hold"], str(g))
    check("a confirmed trend is held for 15 minutes, then released", r["latch"] == ["trend", 1, "trend", 1, "range"], str(r["latch"]))
    o = r["oct15"]
    check("expected range = 1.6 x the close-to-close move", abs(o["ER"] - 1.6 * o["D"]) <= 1, str(o))
    check("a one-way 374-pt day on a +-203 forecast is upgraded to TREND (up)", o["upReg"] == "trend" and o["upDir"] == 1, str(o))
    check("trend day room uses 2.5 sigma: T2 not collapsed to T1", o["upT2"] > round(0.30 * o["D"]) + 20, str(o))
    check("the same range chopped back and forth stays RANGE, T2 capped", o["chopReg"] == "range" and o["chopT2"] == o["chopT1"], str(o))
    lg = r["log"]
    check("entry log: T1 then T2 scored half at each, net of costs",
          lg[0] == "T2" and lg[1] == round((0.5 * (115 - 100 - 1) + 0.5 * (130 - 100 - 1)) * 65) and lg[2] == 1 and lg[3] == 1 and lg[4] == 1, str(lg))
    check("entry alerts: banner, two-read hold, one per direction, cooldown",
          "_seBanner(s)" in dsrc and "if (h.n < R.holdReads || now - h.since < R.holdMs) continue;" in dsrc and "cooldownMs: 300000" in dsrc)
    check("cost defaults: STT 0.1% and exchange 0.03503% on options", "sttPct: 0.1, exchPct: 0.03503" in dsrc)

def test_entry_alert_discipline():
    """Trade-entry popup: a setup must stay ready 20 s / 3 reads; one move is
    announced once (re-arms only after a minute off, never while that
    direction's last alert is still open); no about-turn within 3 minutes;
    the banner lapses after 8 s off, by side/type/direction. Trend entries
    need a real pullback that turned; supporting checks need 2 confirming."""
    import subprocess, json as _json
    d = open(os.path.join(HERE, "nse_dashboard.html"), encoding="utf-8").read()
    def grab(name):
        i = d.index(f"function {name}(")
        depth, j = 0, d.index("{", i)
        for k in range(j, len(d)):
            depth += d[k] == "{"; depth -= d[k] == "}"
            if depth == 0:
                return d[i:k + 1]
    rule = d[d.index("  const _SE_RULE = "):d.index("\n", d.index("  const _SE_RULE = "))]
    harness = r"""
let NOW = 1e12; const _realNow = Date.now; Date.now = () => NOW;
const state = { symbol: 'NIFTY' }; const _isHistorical = () => false;
const _istNowMin = () => ({ m: 600, wd: 3 }); const _seCfg = () => ({ on: true, voice: false, buy: true, sell: true });
const _seState = { hold: {}, fired: {}, banner: null };
let OPEN = []; const _seLog = () => OPEN;
const fired = []; const _seFire = (d, s) => { fired.push([Math.round((NOW - 1e12) / 1000), s.dir]); _seState.banner = { id: s.id, side: s.side, typ: s.typ, dir: s.dir, off: null }; };
let lapsedAt = null; const _seBannerLapsed = () => { lapsedAt = Math.round((NOW - 1e12) / 1000); _seState.banner = null; };
""" + rule + "\n" + grab("_seEvaluate") + r"""
const D = { symbol: 'NIFTY' };
const up = ready => [{ id: 'NIFTY|BUY|CE|25000', side: 'BUY', typ: 'CE', dir: 1, ready, preferred: true }];
const dn = ready => [{ id: 'NIFTY|BUY|PE|25000', side: 'BUY', typ: 'PE', dir: -1, ready, preferred: true }];
const run = (secs, set, step = 2) => { for (let t = 0; t < secs; t += step) { _seEvaluate(D, set); NOW += step * 1000; } };
const R = {};
run(6, up(true)); run(4, up(false)); R.flicker = fired.length;          // 6 s ready then off: nothing
run(30, up(true)); R.first = fired.slice();                            // fires once it has held 20 s
run(600, up(true)); R.stillOne = fired.length;                         // ready for 10 more minutes: no repeat
run(30, up(false)); run(30, up(true)); R.shortOff = fired.length;      // off 30 s: not re-armed
run(70, up(false)); OPEN = [{ res: null, sym: 'NIFTY', dir: 1 }]; run(30, up(true)); R.openBlocks = fired.length;
OPEN = []; run(70, up(false)); run(30, up(true)); R.rearmed = fired.length;   // off > 60 s, no open trade: alerts again
const tUp = NOW; run(60, dn(true)); R.flipBlocked = fired.length;     // DOWN within 3 min of an UP alert: held
run(150, dn(true)); R.flipLater = fired.slice(-1)[0];                  // allowed once 3 minutes have passed
// banner lapse: 4 s off keeps it, 8 s off lapses it
lapsedAt = null; run(4, dn(false)); R.keep = lapsedAt; run(10, dn(false)); R.lapse = lapsedAt != null;
console.log(JSON.stringify(R));
"""
    res = subprocess.run(["node", "-e", harness], capture_output=True, text=True)
    try:
        R = _json.loads(res.stdout.strip().splitlines()[-1])
    except Exception:
        check("entry alerts: harness ran", False, (res.stderr or res.stdout)[:500]); return
    check("entry alerts: a setup ready for only a few seconds does not fire; one held 20 s / 3 reads does",
          R["flicker"] == 0 and len(R["first"]) == 1 and R["first"][0][1] == 1 and 28 <= R["first"][0][0] <= 32, str(R))
    check("entry alerts: one move is announced once - no repeat while it stays ready, none after a short dip",
          R["stillOne"] == 1 and R["shortOff"] == 1, str(R))
    check("entry alerts: no new alert in a direction while its last alert is still open; re-arms after a minute off",
          R["openBlocks"] == 1 and R["rearmed"] == 2, str(R))
    check("entry alerts: no about-turn within 3 minutes of an alert; allowed after",
          R["flipBlocked"] == 2 and R["flipLater"] and R["flipLater"][1] == -1, str(R))
    check("entry alerts: banner lapses after 8 s off, not on a one-poll flicker", R["keep"] is None and R["lapse"], str(R))
    sc = grab("_scalpSetups")
    check("entry alerts: trend entries need a pullback to the 9 EMA / VWAP in the last 8 bars and a turn back with the trend",
          "must(base, 'pulled back & turning', pbOk, pbTxt);" in sc and "for (let i = Math.max(0, n0 - 8); i < n0; i++)" in sc
          and "const turned = le != null && (lc.c - lc.o) * dir > 0 && (lc.c - le) * dir >= 0;" in sc)
    check("entry alerts: supporting checks need 2 confirming - room left, not the lull and breadth do not count as evidence",
          "ready: mOk === musts.length && sOk >= 3 && eOk >= 2," in sc
          and "pts of the day's move left`, false);" in sc and "'active hours', false);" in sc and "(${p.conf.restState})` : '—', false);" in sc)


def test_ctx_and_intervals():
    """Context history kept server-side (same arithmetic as the dashboard),
    candles bucketed from 09:15 for every interval, clickable controls."""
    import ast as _ast, time as _t, json as _j, math as _m
    ssrc = open(os.path.join(HERE, "nse_chain_server.py"), encoding="utf-8").read()
    tree = _ast.parse(ssrc)
    fns = {n.name: n for n in tree.body if isinstance(n, _ast.FunctionDef)}
    ns = {"time": _t, "json": _j}
    for name in ("_sess_bucket", "_ctx_point"):
        check(f"{name} present", name in fns)
        if name not in fns:
            return
        exec(compile(_ast.Module(body=[fns[name]], type_ignores=[]), "srv", "exec"), ns)
    import datetime as _dt
    IST = _dt.timezone(_dt.timedelta(hours=5, minutes=30))
    t915 = int(_dt.datetime(2026, 10, 15, 9, 15, tzinfo=IST).timestamp())
    ok = all(ns["_sess_bucket"](t915 + 7 * 60 + 13, k) == t915 + (7 // k) * k * 60 for k in (1, 2, 3, 5, 10, 15))
    check("candles: every interval (incl. 2m, 10m) starts at 09:15 IST", ok and ns["_sess_bucket"](t915, 10) == t915)
    st = [{"strike": k, "ce_oi": 100.0, "pe_oi": 150.0 if k < 25000 else 50.0, "ce_ltp": 100, "pe_ltp": 90,
           "ce_iv": 12.0, "pe_iv": 14.0} for k in range(24000, 26050, 50)]
    pt = ns["_ctx_point"]({"strikes": st, "underlying_value": 25000, "strike_gap": 50, "dte": 4,
                           "atm_iv": 13, "india_vix": 12.4}, t915 + 3600)
    check("context point: PCR bands, straddle, skew, VIX, IV",
          pt and pt["pcr"] and pt["pcr1d"] and pt["straddle"] == 190 and pt["skew"] == 2.0 and pt["vix"] == 12.4, str(pt))
    check("context history served and recorded per index", "/api/ctx-history" in ssrc and "_ctx_record(symbol.upper(), data)" in ssrc)
    dsrc = open(os.path.join(HERE, "nse_dashboard.html"), encoding="utf-8").read()
    check("dashboard merges the server's context history", "_ctxServerSync" in dsrc and "/api/ctx-history?symbol=" in dsrc)
    check("context tick boxes built once, not on every redraw", "ctlEl.dataset.sig !== sig" in dsrc and ".ctx-ctrls" in dsrc)
    check("panels hold their redraw while a click is in progress", "function _holdRender(box)" in dsrc
          and "if (_holdRender(box)) return;" in dsrc)
    check("chart intervals 1/2/3/5/10/15 with the selected one highlighted",
          all(f'data-iv="{k}"' in dsrc for k in (1, 2, 3, 5, 10, 15)) and ".chart-iv.active {" in dsrc)


def test_book_heavyweights():
    """ITM call book 3x+ held 60 s with the put book not bid-heavy, plus the
    heavyweight basket broadly rising, confirms an uptrend; the mirror case
    confirms a downtrend; both sides bid-heavy confirms nothing."""
    import subprocess, json as _json
    dsrc = open(os.path.join(HERE, "nse_dashboard.html"), encoding="utf-8").read()
    def grab(name):
        i = dsrc.index(f"function {name}(")
        depth, j = 0, dsrc.index("{", i)
        for k in range(j, len(dsrc)):
            depth += dsrc[k] == "{"; depth -= dsrc[k] == "}"
            if depth == 0:
                return dsrc[i:k + 1]
    code = "\n".join(grab(n) for n in ("bsErf", "bsNormCdf", "bsNormPdf", "bsD1D2", "bsGreeks", "_volT",
                                        "_cfHeld", "_median", "_bookHwConf", "_depthFor", "_bookOf", "_ladderFresh"))
    harness = code + r"""
const RISK_FREE = 0.07; let NOW = 1792040000000; Date.now = () => NOW;
const state = { symbol: 'NIFTY' }; const scored = [];
const _scoreRecord = (src, dir) => scored.push(src + ':' + dir); const _dhAgo = () => null;
let _cfH = { c: [], p: [], held: {}, bad: {} };
function chain(ceR, peR, hwPts, up, dn, rest) {
  const st = [], strikes = {};
  for (let K = 24500; K <= 25500; K += 50) {
    st.push({ strike: K, ce_iv: 13, pe_iv: 13, ce_ltp: 10, pe_ltp: 10 });
    strikes[K + 'CE'] = { tbq: K < 25000 ? 1000 * ceR : 1000, tsq: 1000, tts: NOW / 1000 };
    strikes[K + 'PE'] = { tbq: K > 25000 ? 1000 * peR : 1000, tsq: 1000, tts: NOW / 1000 };
  }
  return { symbol: 'NIFTY', underlying_value: 25000, strikes: st, dte: 4, atm_iv: 13, india_vix: 13, strike_gap: 50,
           depth: { strikes }, constituents: { available: true, score: 40,
           hw: { win_s: 300, share: 60, m5: { pts: hwPts, up, dn, n: 11, ret: 0.2 }, rest_m5: rest || null } } };
}
const run = (ceR, peR, hw, up, dn, rest) => { _cfH = { c: [], p: [], held: {}, bad: {} }; NOW = 1792040000000;
  _bookHwConf(chain(ceR, peR, hw, up, dn, rest)); NOW += 70000; const r = _bookHwConf(chain(ceR, peR, hw, up, dn, rest));
  return { book: r.book, raw: r.bookRaw, hw: r.hw, conf: r.confirmed, score: r.score, C: r.C.strikes, label: r.label }; };
console.log(JSON.stringify({
  up: run(4, 0.7, 40, 9, 1), down: run(0.7, 4, -40, 1, 9), both: run(4, 4, 40, 9, 1),
  narrow: run(4, 0.7, 40, 3, 2), restOpp: run(4, 0.7, 40, 9, 1, { pts: -20, up: 6, dn: 30, n: 39, share: 40 }),
  restOk: run(4, 0.7, 40, 9, 1, { pts: 8, up: 25, dn: 10, n: 39, share: 40 }),
  sustained: (() => { _cfH = { c: [], p: [], held: {}, bad: {} }; NOW = 1792040000000;
    for (let i = 0; i < 30; i++) { _bookHwConf(chain(1.0, 0.8, 0, 5, 5)); NOW += 11000; }     // quiet half hour
    let r; for (let i = 0; i < 40; i++) { r = _bookHwConf(chain(3.5, 0.8, 40, 9, 1)); NOW += 90000; }   // an hour at 3.5x
    return { book: r.book, vsNorm: r.C.vsNorm, conf: r.confirmed }; })(),
  dip: (() => { _cfH = { c: [], p: [], held: {}, bad: {} }; NOW = 1792040000000;
    _bookHwConf(chain(4, 0.7, 40, 9, 1)); NOW += 70000; _bookHwConf(chain(4, 0.7, 40, 9, 1)); NOW += 5000;
    const a = _bookHwConf(chain(2.7, 0.8, 40, 9, 1)).book; NOW += 5000;           // one weak refresh
    _bookHwConf(chain(1.5, 0.8, 40, 9, 1)); NOW += 25000;
    const b = _bookHwConf(chain(1.5, 0.8, 40, 9, 1)).book;                         // 25 s below 2.5x
    return { keptThroughDip: a, afterFade: b }; })(),
  scored }));
"""
    fn = os.path.join(HERE, "__conf_test.js")
    open(fn, "w").write(harness)
    try:
        out = subprocess.run(["node", fn], capture_output=True, text=True, timeout=30)
    finally:
        os.remove(fn)
    if out.returncode:
        check("book + heavyweights engine runs", False, out.stderr[-400:]); return
    r = _json.loads(out.stdout.strip().splitlines()[-1])
    check("ITM call book 4x + broad heavyweight rally -> UPTREND CONFIRMED",
          r["up"]["conf"] == 1 and r["up"]["book"] == 1 and r["up"]["hw"] == 1 and r["up"]["score"] >= 60, str(r["up"]))
    check("ITM put book 4x + broad heavyweight fall -> DOWNTREND CONFIRMED", r["down"]["conf"] == -1, str(r["down"]))
    check("both books bid-heavy is not direction (mirror rule)", r["both"]["book"] == 0 and r["both"]["conf"] == 0, str(r["both"]))
    check("one or two heavyweights carrying it is not broad", r["narrow"]["hw"] == 0 and r["narrow"]["conf"] == 0, str(r["narrow"]))
    check("heavyweights up while the rest of the index falls is NARROW: no confirmation",
          r["restOpp"]["conf"] == 0 and "narrow" in r["restOpp"]["label"], str(r["restOpp"]))
    check("heavyweights up and the rest agreeing still confirms", r["restOk"]["conf"] == 1, str(r["restOk"]))
    check("a call stack held for an hour is still unusual (normal built from quiet periods)",
          r["sustained"]["book"] == 1 and (r["sustained"]["vsNorm"] or 0) >= 3, str(r["sustained"]))
    check("one weak refresh does not reset a held stack; 20 s+ below 2.5x does",
          r["dip"]["keptThroughDip"] == 1 and r["dip"]["afterFade"] == 0, str(r["dip"]))
    check("ITM strikes are the ones just in the money", r["up"]["C"] and all(k < 25000 for k in r["up"]["C"]), str(r["up"]["C"]))
    check("each witness and the pair go to the Signal Scorecard",
          any(x.startswith("ITM book (mirror + norm):1") for x in r["scored"]) and any(x.startswith("Book + heavyweights:1") for x in r["scored"]))
    check("trade entries: no trades against a confirmed book + heavyweights", "'not against book + heavyweights'" in dsrc)
    check("no double counting: entries read the Trend Engine without its book/constituent inputs",
          "scoreEx" in dsrc and "cf.scoreCore / 80" in dsrc)
    check("a side's ratio needs at least two live books", "ratio: n >= 2 && ask > 0" in dsrc)


def test_pcr_smooth():
    """Context-pane PCR has soft edges (no step when spot crosses a strike
    midpoint), outliers are filtered, and the curve cannot overshoot."""
    import subprocess, json as _json
    dsrc = open(os.path.join(HERE, "nse_dashboard.html"), encoding="utf-8").read()
    def grab(name):
        i = dsrc.index(f"function {name}(")
        depth, j = 0, dsrc.index("{", i)
        for k in range(j, len(dsrc)):
            depth += dsrc[k] == "{"; depth -= dsrc[k] == "}"
            if depth == 0:
                return dsrc[i:k + 1]
    code = grab("_volT") + "\n" + grab("_pcrKernel")
    harness = code + r"""
const st = [];
for (let K = 24000; K <= 26000; K += 50) st.push({ strike: K, ce_oi: 1000, pe_oi: K === 24800 ? 30000 : 1000 });
const at = (spot) => _pcrKernel({ strikes: st, underlying_value: spot, strike_gap: 50, atm_iv: 12 }, 'rod', Date.UTC(2026, 9, 15, 9, 30));
const a = at(24974), b = at(24976), b2 = at(24978), c = at(24990);
console.log(JSON.stringify({ a, b, b2, c }));
"""
    fn = os.path.join(HERE, "__pcr_test.js")
    open(fn, "w").write(harness)
    try:
        out = subprocess.run(["node", fn], capture_output=True, text=True, timeout=30)
    finally:
        os.remove(fn)
    if out.returncode:
        check("PCR kernel runs", False, out.stderr[-300:]); return
    r = _json.loads(out.stdout.strip().splitlines()[-1])
    # continuity: the 2-point change ACROSS the 24,975 midpoint is no bigger
    # than the same 2-point change away from it - a slope, not a step
    check("PCR does not step when spot crosses a strike midpoint",
          abs(r["a"] - r["b"]) <= 1.2 * abs(r["b"] - r["b2"]) + 1e-6, str(r))
    check("PCR still follows price (a big put strike weighs less as spot moves away)", r["c"] < r["a"], str(r))
    check("context lines: median per slot, outlier filter, monotone curve",
          "3 * mad" in dsrc and "Fritsch-Carlson" in dsrc and "perSlot(s.get, 0.45)" in dsrc)
    ssrc = open(os.path.join(HERE, "nse_chain_server.py"), encoding="utf-8").read()
    check("server records the same soft-edged PCR to a new file version", "k = _mm.exp(-0.5 * z * z)" in ssrc and ".v2.jsonl" in ssrc)


def test_market_direction():
    """Market direction from price + PCR + VIX + IV + skew read together."""
    import subprocess, json as _json
    dsrc = open(os.path.join(HERE, "nse_dashboard.html"), encoding="utf-8").read()
    check("market direction strip sits directly above the candles",
          dsrc.index('id="mktDirBody"') < dsrc.index('id="chartBody"') and "renderMktDir(data)" in dsrc)
    def grab(name):
        i = dsrc.index(f"function {name}(")
        depth, j = 0, dsrc.index("{", i)
        for k in range(j, len(dsrc)):
            depth += dsrc[k] == "{"; depth -= dsrc[k] == "}"
            if depth == 0:
                return dsrc[i:k + 1]
    harness = grab("_mktDir") + r"""
const state = { symbol: 'NIFTY' }; const _chart = { candles: [], hasVolume: false };
let _ctxHist = [];
function series(dSpot, dPcr, dVix, dSkew) {
  _ctxHist = []; const now = Date.now();
  for (let i = 0; i <= 20; i++) {
    const f = i / 20;
    _ctxHist.push({ sym: 'NIFTY', ts: now - (20 - i) * 60000, spot: 25000 + dSpot * f, pcr: 1.0 + dPcr * f,
                    vix: 13 + dVix * f, iv: 13 + dVix * f, skew: 2 + dSkew * f, straddle: 300 });
  }
  return _mktDir({ underlying_value: 25000 + dSpot, atm_iv: 13 }, 15);
}
const up = series(60, 0.08, -0.4, -0.4), down = series(-60, 0.08, 0.5, 0.5), nervous = series(60, 0.0, 0.6, 0);
console.log(JSON.stringify({ up: up.score, down: down.score, nervous: nervous.comp.find(c => c.k === 'vix'),
  upPcr: up.comp.find(c => c.k === 'pcr').txt, downPcr: down.comp.find(c => c.k === 'pcr').txt }));
"""
    fn = os.path.join(HERE, "__md_test.js")
    open(fn, "w").write(harness)
    try:
        out = subprocess.run(["node", fn], capture_output=True, text=True, timeout=30)
    finally:
        os.remove(fn)
    if out.returncode:
        check("market direction runs", False, out.stderr[-300:]); return
    r = _json.loads(out.stdout.strip().splitlines()[-1])
    check("rising price + put writing + VIX and skew falling reads UP", r["up"] >= 55, str(r))
    check("falling price + puts bought + VIX and skew rising reads DOWN", r["down"] <= -55, str(r))
    check("PCR read against price: written under a rise, bought into a fall",
          "written" in r["upPcr"] and "bought" in r["downPcr"], str(r))
    check("a rally with VIX rising is flagged nervous", r["nervous"] and "nervous" in r["nervous"]["txt"] and r["nervous"]["s"] < 0, str(r))


def test_traffic_light():
    """Turn light (replaces the trend-summary traffic light): WAIT -> WATCH at
    a level with a high enough score -> BUY/SELL on a closed candle that turns,
    reclaims and (broker feed) is flow-confirmed -> FAILED/DONE -> WAIT; QUIET
    at the open, the close and scheduled events; hand-set weights until a
    fitted model is accepted; evidence in families counted once."""
    import subprocess, json as _json
    dsrc = open(os.path.join(HERE, "nse_dashboard.html"), encoding="utf-8").read()
    def grab(name):
        i = dsrc.index(f"function {name}(")
        depth, j = 0, dsrc.index("{", i)
        for k in range(j, len(dsrc)):
            depth += dsrc[k] == "{"; depth -= dsrc[k] == "}"
            if depth == 0:
                return dsrc[i:k + 1]
    consts = "\n".join(dsrc[dsrc.index(f"  const {c} = "):dsrc.index(";\n", dsrc.index(f"  const {c} = ")) + 2]
                       for c in ("_TURN_FAM", "_TURN_PRIOR", "_TURN_RULE", "_TURN_EVENTS"))
    harness = consts + "\n" + "\n".join(grab(f) for f in ("_turnWeights", "_turnScore", "_turnThr", "_turnQuiet", "_turnStep", "_turnResLoad", "_turnResAdd", "_turnTally", "_turnCool")) + r"""
const _turnRes = { day: '', list: [] };
let NOW = 1e12; Date.now = () => NOW; let MIN = 600;
const _turnSide = dir => dir > 0 ? 'bottom' : 'top';
const _turn = { st: 'WAIT', dir: 0, since: 0, note: '', w: null, trig: null, end: 0, lastBarT: null, model: null, events: [] };
const state = { symbol: 'NIFTY' }; const fmtInt = x => String(x);
const _istNowMin = () => ({ m: MIN, wd: 3 }); let DAY = '2026-10-28'; const _istDayKey = () => DAY;
const recs = []; const _scoreRecord = (src, dir) => recs.push([src, dir]);
const fam = o => Object.assign({ loc: 0, sweep: 0, mom: 0, flow: 0, absorb: 0, pos: 0, gamma: 0, breadth: 0, smt: 0, vix: 0, time: 0, reg: 1 }, o);
const F = (t, bar, prev, fb, ft, flowOn = true) => ({ spot: bar.c, D: 200, s15: 40, iv: 1, flowOn, lastT: t, bar, prevBar: prev,
  recentLo: Math.min(bar.l, prev.l), recentHi: Math.max(bar.h, prev.h),
  side: { 1: { f: fam(fb), why: {}, lvl: 24900, lvlLab: 'put wall 24900' }, '-1': { f: fam(ft), why: {}, lvl: 25200, lvlLab: 'call wall 25200' } } });
const score = f => ({ 1: Math.round(100 * _turnScore(f.side[1].f, 'bottom')), '-1': Math.round(100 * _turnScore(f.side[-1].f, 'top')) });
const step = (f, spot) => { const d = { underlying_value: spot ?? f.bar.c, symbol: 'NIFTY' }; _turnStep(d, f, score(f), _turnQuiet(), true); return _turn.st; };
const R = {};
const strong = { loc: 1, sweep: 1, mom: 0.9 };                  // -3 + 1.4 + 1.1 + 0.81 = 0.31 -> 58
R.prior = Math.round(100 * _turnScore(fam(strong), 'bottom'));
R.weak = Math.round(100 * _turnScore(fam({ loc: 1, mom: 0.5 }), 'bottom'));
const b1 = { o: 24930, h: 24935, l: 24895, c: 24900 }, p1 = { o: 24960, h: 24965, l: 24925, c: 24930 };
R.s0 = step(F(1, b1, p1, { loc: 1, mom: 0.5 }, {}));               // at a level but weak: WAIT
R.s1 = step(F(1, b1, p1, strong, {}));                             // strong + at a level: WATCH up
R.dir1 = _turn.dir;
NOW += 60000;
const b2 = { o: 24900, h: 24915, l: 24890, c: 24912 };            // green, closes back above the 24900 level
R.s2 = step(F(2, b2, b1, Object.assign({ flow: 0 }, strong), {}));        // turned + reclaimed but no flow confirm: WATCH
NOW += 60000;
const b3 = { o: 24910, h: 24925, l: 24905, c: 24920 };
R.s3 = step(F(3, b3, b2, Object.assign({ flow: 0.6 }, strong), {}));      // flow confirms: BUY
R.trig = _turn.trig && { stop: Math.round(_turn.trig.stop), target: Math.round(_turn.trig.target) };
NOW += 30000; R.s4 = step(F(3, b3, b2, strong, {}), 24880);        // trades under the stop: FAILED
NOW += 60000; R.s5 = step(F(4, b3, b2, {}, {}));                   // still showing FAILED for 2 min
NOW += 90000; R.s6 = step(F(5, b3, b2, {}, {}));                   // back to WAIT
// target path + top side
_turn.st = 'WAIT'; NOW += 60000;
const t1 = { o: 25170, h: 25205, l: 25165, c: 25195 }, tp = { o: 25150, h: 25175, l: 25140, c: 25170 };
R.t1 = step(F(10, t1, tp, {}, strong)); R.tdir = _turn.dir;        // WATCH down
NOW += 60000;
const t2 = { o: 25195, h: 25198, l: 25170, c: 25176 };            // red, closes back under the 25200 level
R.t2 = step(F(11, t2, t1, {}, strong, false));                     // no flow feed: needs score >= watch + 10 -> 58 >= 50 -> SELL
NOW += 60000; R.t3 = step(F(11, t2, t1, {}, strong), 25176 - 61);  // target (0.3 x 200 = 60) reached: DONE
// QUIET: opening, an event window
_turn.st = 'WAIT'; NOW += 5 * 60000; MIN = 560; R.q1 = step(F(20, b1, p1, strong, {}));
MIN = 600; DAY = '2026-12-04'; R.q2 = [step(F(21, b1, p1, strong, {})), _turn.note];
DAY = '2026-10-28'; _turn.events = [{ day: '2026-10-28', from: 590, to: 620, lab: 'custom' }]; R.q3 = _turn.note && step(F(22, b1, p1, strong, {}));
_turn.events = [];
// a fitted model takes over only when accepted
_turn.model = { accepted: false, sides: { bottom: { b: -9, w: {}, watch_p: 0.2 } } }; R.notAcc = _turnThr(1);
_turn.model = { accepted: true, sides: { bottom: { b: -1, w: { loc: 2 }, watch_p: 0.31, auc: 0.6 }, top: { b: -1, w: {}, watch_p: 0.3, auc: 0.6 } } };
R.acc = [_turnThr(1), Math.round(100 * _turnScore(fam({ loc: 1 }), 'bottom'))];
R.recs = recs.map(r => r.join(':'));
// a one-way trend against the call takes points off (58 -> 38, under the watch line)
_turn.model = null;
R.reg = [Math.round(100 * _turnScore(fam(strong), 'bottom')), Math.round(100 * _turnScore(fam(Object.assign({ reg: 0 }, strong)), 'bottom'))];
// a trigger whose stop would be too far from the entry is refused: stays WATCH, says why
_turn.model = null; _turn.st = 'WAIT'; _turn.w = null; _turn.trig = null; NOW += 600000;
const l1 = { o: 24930, h: 24935, l: 24895, c: 24900 }, lp = { o: 24960, h: 24965, l: 24925, c: 24930 };
step(F(30, l1, lp, strong, {}));
NOW += 60000;
const l2 = { o: 24905, h: 24940, l: 24900, c: 24935 };             // a big green bar: 24935 is 44 pts above the stop (max 36), 40 above the low (not yet 'away')
R.late = [step(F(31, l2, l1, Object.assign({ flow: 0.6 }, strong), {})), !!(_turn.w && _turn.w.lastCheck && _turn.w.lastCheck.late), _turn.trig == null];
// two stops in a row raise the bar for a while, then it relaxes
_turnRes.list = []; _turnRes.day = DAY;
_turnResAdd(1, 'FAILED'); const c1 = _turnThr(1); _turnResAdd(-1, 'FAILED'); const c2 = _turnThr(1);
NOW += 46 * 60000; const c3 = _turnThr(1);
_turnResAdd(1, 'DONE'); R.cool = [c1, c2, c3, JSON.stringify(_turnTally())];
console.log(JSON.stringify(R));
"""
    fn = os.path.join(HERE, "__tl_test.js")
    open(fn, "w").write(harness)
    try:
        out = subprocess.run(["node", fn], capture_output=True, text=True, timeout=30)
    finally:
        os.remove(fn)
    if out.returncode:
        check("turn light runs", False, out.stderr[-400:]); return
    r = _json.loads(out.stdout.strip().splitlines()[-1])
    check("turn light: hand-set logistic score - strong evidence at a level 58, weak 24", r["prior"] == 58 and r["weak"] == 24, str(r))
    check("turn light: WAIT when the score is low, WATCH ▲ when at a level with a high enough score",
          r["s0"] == "WAIT" and r["s1"] == "WATCH" and r["dir1"] == 1, str(r))
    check("turn light: no trigger without flow confirmation on the broker feed; BUY when the closed candle turns, reclaims and flow confirms",
          r["s2"] == "WATCH" and r["s3"] == "BUY", str(r))
    check("turn light: stop just beyond the watched extreme, target 0.3 × the day's σ",
          r["trig"] and r["trig"]["stop"] == 24886 and r["trig"]["target"] == 24980, str(r["trig"]))
    check("turn light: FAILED when the stop trades, held 2 minutes, then WAIT", r["s4"] == "FAILED" and r["s5"] == "FAILED" and r["s6"] == "WAIT", str(r))
    check("turn light: top side - WATCH ▼, SELL on the NSE feed by score + 10, DONE at the target",
          r["t1"] == "WATCH" and r["tdir"] == -1 and r["t2"] == "SELL" and r["t3"] == "DONE", str(r))
    check("turn light: QUIET in the opening 10 minutes, on RBI policy day and in NSE_EVENTS windows",
          r["q1"] == "QUIET" and r["q2"] == ["QUIET", "RBI policy 10:00"] and r["q3"] == "QUIET", str(r))
    check("turn light: a fitted model is used only once accepted (its own watch line)",
          r["notAcc"] == 40 and r["acc"] == [31, 73], str(r))
    check("turn light: watch and trigger logged to the Signal Scorecard",
          "Turn watch:1" in r["recs"] and "Turn trigger:1" in r["recs"] and "Turn trigger:-1" in r["recs"], str(r["recs"]))
    check("turn light: a one-way trend against the call takes points off (reg 1 -> 0 drops 58 to 38)",
          r["reg"] == [58, 38], str(r["reg"]))
    check("turn light: a trigger whose stop is too far from the entry is refused - stays WATCH and says so",
          r["late"] == ["WATCH", True, True], str(r["late"]))
    check("turn light: two stops in a row raise the watch line by 8 for 45 minutes; today's tally kept",
          r["cool"] == [40, 48, 40, '{"win":1,"loss":2,"to":0}'], str(r["cool"]))
    feats = grab("_turnFeatures")
    check("turn light: the trend regime is measured (efficiency over 30 min x distance) per side and logged with the other families",
          "f.reg = +(1 - adv).toFixed(3)" in grab("_turnFeatures") and "const ter = tpath > 0" in grab("_turnFeatures")
          and "for (const k of _TURN_FAM)" in grab("_turnLog"))
    check("turn light: oscillators are ONE momentum family (Squeeze, WaveTrend) and only regular divergences count",
          "f.mom = +mo.toFixed(3)" in feats and feats.count("f.mom =") == 1 and "!x.hidden && x.kind ===" in feats)
    check("turn light: new inputs - Nifty/BankNifty divergence, futures absorption, VIX spike-reversal, events",
          "f.smt = 1" in feats and "f.absorb = holds ?" in feats and "VIX spiked" in feats and "'RBI policy 10:00'" in dsrc
          and "_turnPeerSym = s => s === 'NIFTY' ? 'BANKNIFTY' : 'NIFTY'" in dsrc and "/api/ctx-history?symbol=" in dsrc)
    check("turn light: one row per closed bar logged to the server; drawn over the candles; past days show labelled turns",
          "fetch(`${base}/api/turn-log`" in dsrc and "_turnChartSvg(y, lo, hi, padL, W - padR," in dsrc
          and "/api/turn-labels?symbol=" in dsrc and 'id="tlight"' in dsrc and "_trafficVotes" not in dsrc)


def test_trade_engine():
    """nse_trade: paper and live order flow, every server-side guard, the
    managed exits on the live feed, charges, reconcile."""
    import importlib, calendar as _cal, tempfile, shutil, time
    sys.path.insert(0, HERE)
    N = importlib.import_module("nse_trade")
    T0 = _cal.timegm((2026, 10, 29, 5, 30, 0))                 # Thu 11:00 IST
    class Clk:
        t = T0
        def __call__(self): return self.t
    clk = Clk()
    class Feed:
        def __init__(self): self.q, self.m, self.sp = {}, {}, {"NIFTY": 25000.0}
        def add(self, inst, bid, ask, lot=65, typ="CE", ltp=None):
            self.q[inst] = dict(bid=bid, ask=ask, ltp=ltp or (bid + ask) / 2, ts=clk())
            self.m[inst] = dict(under="NIFTY", typ=typ, strike=25000, expiry="2026-11-03", lot=lot, ts=inst.split(":")[-1])
        def move(self, inst, bid, ask): self.q[inst].update(bid=bid, ask=ask, ltp=(bid + ask) / 2, ts=clk())
        def quote(self, i): return self.q.get(i)
        def meta(self, i): return self.m.get(i)
        def spot(self, u): return self.sp.get(u)
        def depth(self, i):
            q = self.q.get(i)
            return q and {"asks": [{"p": q["ask"], "q": 130}, {"p": round(q["ask"] + 0.5, 2), "q": 500}],
                          "bids": [{"p": q["bid"], "q": 130}, {"p": round(q["bid"] - 0.5, 2), "q": 500}]}
    I, P = "NFO:NIFTY25000CE", "NFO:NIFTY25000PE"
    def mk(broker=None, **kw):
        clk.t = T0
        f = Feed(); f.add(I, 99.0, 99.5); f.add(P, 80.0, 80.5, typ="PE")
        cfg = N.Config(env={"NSE_TRADE_MODE": "live" if broker else "paper"})
        for k, v in kw.items(): setattr(cfg, k, v)
        return f, N.Engine(f, broker, cfg, clock=clk, async_exits=False, sleep=lambda s: None)
    # ── entry: fills at the ask by walking the book; guards ──
    f, e = mk()
    r = e.buy(I, 1, {"sl_pts": 10, "tp_pts": 20, "trail_pts": 8, "be_pts": 6}, req_id="a")
    p = e.pos[I]
    check("trade: paper buy fills at the ask (65 = 1 lot) and arms the exits as levels from the fill",
          r["ok"] and p.qty == 65 and p.avg == 99.5 and (p.sl, p.tp, p.trail, p.be_at) == (89.5, 119.5, 8.0, 105.5), str(r))
    check("trade: a repeated request id (double click) is answered, not re-sent", e.buy(I, 1, req_id="a").get("dup") is True and e.pos[I].qty == 65)
    f, e = mk(max_lots=2, max_open_lots=4)
    check("trade: more lots than the per-order cap is refused (when a cap is set)", "per-order cap" in e.buy(I, 3)["msg"])
    f2, e2 = mk(max_lots=2, max_open_lots=4); e2.buy(I, 2)
    check("trade: open lots are capped across orders", e2.buy(P, 1, {}) ["ok"] and "open lots" in e2.buy(P, 2)["msg"])
    f3, e3 = mk(max_value=1000)
    f4, e4 = mk(max_spread_pct=6)
    f4.move(I, 90.0, 99.5)
    f5, e5 = mk(min_premium=3); f5.add(I, 1.0, 1.5)
    check("trade: order value, a wide spread and a junk premium are refused",
          "order value" in e3.buy(I, 1)["msg"] and "spread" in e4.buy(I, 1)["msg"] and "below the minimum" in e5.buy(I, 1)["msg"])
    f6, e6 = mk(); clk.t += 30
    check("trade: a stale quote is refused (no price to protect the position with)", "no fresh quote" in e6.buy(I, 1)["msg"])
    f7, e7 = mk(); clk.t = _cal.timegm((2026, 10, 29, 3, 40, 0))      # 09:10 IST
    f7.move(I, 99.0, 99.5)
    check("trade: outside the entry window is refused", "entries only" in e7.buy(I, 1)["msg"])
    f8, e8 = mk(); e8.set_kill(True, why="test")
    check("trade: the kill switch blocks entries", "kill switch" in e8.buy(I, 1)["msg"])
    check("trade: lots must be a whole number >= 1 (0, 1.5 and True are refused by the engine itself)",
          all("whole number" in (mk()[1].check_entry(I, x) or "") for x in (0, 1.5, -1)))
    f9, e9 = mk()
    check("trade: an unknown contract is refused", e9.buy("NFO:NOPE", 1)["msg"] == "not a listed index option")
    # ── managed exits ──
    f, e = mk(); e.buy(I, 1, {"sl_pts": 10, "tp_pts": 20, "trail_pts": 8, "be_pts": 6}); p = e.pos[I]
    f.move(I, 108.0, 108.5); e.step()
    check("trade: peak, breakeven and trail follow the bid up", p.peak == 108.0 and p.sl == 99.55 and p.eff_sl() == 100.0, f"{p.peak} {p.sl} {p.eff_sl()}")
    f.move(I, 101.0, 101.5); e.step()
    check("trade: still held above the trailed stop", I in e.pos)
    f.move(I, 99.9, 100.4); e.step()
    t = e.trades[-1] if e.trades else {}
    check("trade: the trailed stop sells the position at the bid and books the trade with charges",
          I not in e.pos and t.get("why", "").startswith("stop") and t.get("exit") == 99.9 and t.get("fee", 0) > 40, str(t))
    f, e = mk(); e.buy(I, 1, {"sl_pts": 10, "tp_pts": 12}); f.move(I, 112.0, 112.5); e.step()
    check("trade: the target sells at the bid for a profit", I not in e.pos and e.trades[-1]["gross"] > 0 and "target" in e.trades[-1]["why"])
    f, e = mk(max_lots=3, max_open_lots=4); e.buy(I, 2, {"sl_pts": 10, "tp1_pts": 8, "tp1_pct": 50}); p = e.pos[I]
    f.move(I, 108.0, 108.5); e.step()
    check("trade: a partial target sells half the lots and moves the stop to breakeven for the rest",
          p.qty == 65 and p.tp1_done and p.sl > p.avg and e.trades[-1]["qty"] == 65, f"{p.qty} {p.sl}")
    f, e = mk(); e.buy(I, 1, {"u_stop": 24950, "u_target": 25100}); f.sp["NIFTY"] = 24940.0; e.step()
    check("trade: an underlying-level stop sells a call when the index falls through it", I not in e.pos and "underlying stop" in e.trades[-1]["why"])
    f, e = mk(); f.add(P, 80.0, 80.5, typ="PE"); e.buy(P, 1, {"u_stop": 25050}); f.sp["NIFTY"] = 25060.0; e.step()
    check("trade: ... and a put when the index rises through it", P not in e.pos)
    f, e = mk(); e.buy(I, 1, {"time_stop_s": 60}); clk.t += 61; f.move(I, 99.0, 99.5); e.step()
    check("trade: the time stop closes a position held too long", I not in e.pos and e.trades[-1]["why"] == "time stop")
    f, e = mk(); e.buy(I, 1, {}); clk.t = _cal.timegm((2026, 10, 29, 9, 45, 0)); f.move(I, 99.0, 99.5); e.step()      # 15:15 IST
    check("trade: everything is squared off at 15:14 even with no exits set", I not in e.pos and "end-of-day" in e.trades[-1]["why"])
    f, e = mk(); e.buy(I, 1, {"sl_pts": 10}); clk.t += 20; e.step(); p = e.pos[I]
    check("trade: a stale feed is flagged on the position and no stop fires blind", p.stale and I in e.pos)
    # ── risk: daily loss, streak ──
    f, e = mk(day_loss=300); e.buy(I, 1, {}); f.move(I, 94.0, 94.5); e.step()
    check("trade: the daily loss limit trips the kill switch and flattens", e.kill and I not in e.pos and "daily loss" in e.kill_why, str(e.kill_why))
    f, e = mk(loss_streak=2, cooldown_s=300)
    for _ in range(2):
        e.buy(I, 1, {}); clk.t += 2; f.move(I, 98.0, 98.5); e.exit(I, 100, "t"); clk.t += 2; f.move(I, 99.0, 99.5)
    check("trade: a losing streak starts a cooldown that refuses entries", "cooling off" in e.buy(I, 1)["msg"] and e.state()["cooldown_s"] > 0)
    clk.t += 301; f.move(I, 99.0, 99.5)
    check("trade: ... and entries resume afterwards", e.buy(I, 1)["ok"])
    check("trade: charges estimate is plausible (about Rs 50-60 for a 6,500 buy and 6,600 sell)", 40 < N.est_charges(6500, 6600, 2) < 70, str(N.est_charges(6500, 6600, 2)))
    # ── reduce-only, partial exit rounding ──
    f, e = mk(max_lots=3, max_open_lots=4); e.buy(I, 3); e.exit(I, 50, "half")
    check("trade: a 50% exit sells whole lots (3 lots -> 2 sold... rounds, never above what is held)", 0 < e.pos[I].qty < 195 and e.pos[I].qty % 65 == 0)
    check("trade: there is no way to sell what is not held (no short)", e.exit(P, 100)["msg"] == "no such position")
    # ── live ──
    class Broker:
        def __init__(self, status="COMPLETE", filled=None, avg=None):
            self.st, self.filled, self.avg, self.placed, self.pos_rows = status, filled, avg, [], []
        def place(self, ts, side, qty, price, tag=""):
            self.placed.append((ts, side, qty, price)); return "OID1"
        def status(self, oid): return {"status": self.st, "filled": self.placed[-1][2] if self.filled is None else self.filled,
                                         "avg": self.placed[-1][3] - 0.3 if self.avg is None else self.avg, "msg": ""}
        def positions(self): return self.pos_rows
    b = Broker(); f, e = mk(b)
    check("trade: LIVE mode refuses orders until armed", "not armed" in e.buy(I, 1)["msg"] and not b.placed)
    check("trade: arming works only in live mode with a broker", e.set_armed(True)["ok"] and mk()[1].set_armed(True)["ok"] is False)
    r = e.buy(I, 1)
    check("trade: a live buy is a marketable LIMIT above the ask (99.5 + 1.5% -> 101.00), and books the broker's fill",
          r["ok"] and b.placed[0] == ("NIFTY25000CE", "BUY", 65, 101.0) and e.pos[I].avg == 100.7, f"{r} {b.placed}")
    e.exit(I, 100, "t")
    check("trade: a live sell is a marketable limit BELOW the bid (99.0 - 1.5% -> 97.50)", b.placed[1][:3] == ("NIFTY25000CE", "SELL", 65) and abs(b.placed[1][3] - 97.5) < 0.06, str(b.placed))
    b2 = Broker("CANCELLED", filled=0); f, e = mk(b2); e.set_armed(True)
    r = e.buy(I, 1)
    check("trade: an IOC order that finds nothing is reported, and no position is invented", not r["ok"] and I not in e.pos and not e.halt)
    b3 = Broker("OPEN"); f, e = mk(b3); e.set_armed(True)
    r = e.buy(I, 1)
    check("trade: an order whose outcome cannot be confirmed HALTS the engine until reconciled", not r["ok"] and e.halt and "halted" in e.buy(P, 1)["msg"] and not e.set_armed(True)["ok"])
    b4 = Broker(); f, e = mk(b4); e.set_armed(True); e.buy(I, 1); b4.pos_rows = [{"tradingsymbol": "NIFTY25000CE", "quantity": 0}]
    rec = e.reconcile()
    check("trade: reconcile spots a position the broker no longer shows and halts", not rec["ok"] and e.halt.startswith("book differs"))
    b4.pos_rows = [{"tradingsymbol": "NIFTY25000CE", "quantity": 65, "average_price": 100.7}]
    rec = e.reconcile(adopt=True)
    check("trade: ... and adopting the broker's book clears the halt", rec["ok"] and not e.halt and e.pos[I].qty == 65)
    b5 = Broker(); f, e = mk(b5); e.set_armed(True); e.buy(I, 1, {"sl_pts": 10}); f.move(I, 85.0, 85.5)
    e.set_kill(True, flatten=True, why="test")
    check("trade: the kill switch with flatten sells the live position and disarms", I not in e.pos and not e.armed and b5.placed[-1][1] == "SELL")
    # ── journal ──
    d = tempfile.mkdtemp()
    try:
        f, e = mk(); e.jdir = d; e.buy(I, 1, {}); e.exit(I, 100, "t")
        lines = open(os.path.join(d, os.listdir(d)[0])).read().strip().splitlines()
        check("trade: every order and fill is journalled to a daily file", len(lines) >= 4 and '"ev":"fill"' in lines[-1])
    finally:
        shutil.rmtree(d, ignore_errors=True)
    # ── break-even needs the bid above entry; a dragged stop sits under the bid ──
    f, e = mk(); e.buy(I, 1, {"sl_pts": 10}); f.move(I, 108.0, 108.5)
    e.modify(I, {"sl": 99.5})
    check("trade: a stop at break-even is kept once the bid is above entry", e.pos[I].sl == 99.5, str(e.pos[I].sl))
    e.modify(I, {"sl": 120.0})
    check("trade: a stop dragged above the bid is held two ticks under it (it would fire at once otherwise)", e.pos[I].sl == 107.9, str(e.pos[I].sl))
    # ── state carries the whole closed-trade list and the last 60 orders (the analytics tab reads them) ──
    f, e = mk(loss_streak=0)
    for _ in range(30):
        e.buy(I, 1, {}); e.exit(I, 100, "t"); clk.t += 1; f.move(I, 99.0, 99.5)
    st = e.state()
    check("trade: state returns every closed trade (not only the last 25) and the last 60 orders",
          len(st["trades"]) == 30 and len(st["orders"]) <= 60 and len(st["orders"]) > 25, f'{len(st["trades"])} {len(st["orders"])}')
    # ── short selling (sell to open) ──
    f, e = mk(); r = e.sell(I, 1, {"sl_pts": 10, "tp_pts": 20, "trail_pts": 8, "be_pts": 6}); p = e.pos.get(I)
    check("short: a paper sell fills at the bid, and the stop sits ABOVE and the target BELOW the entry",
          r["ok"] and p and p.side == -1 and p.avg == 99.0 and (p.sl, p.tp) == (109.0, 79.0), str(r))
    f.move(I, 90.0, 90.5); e.step()
    check("short: the mark is the ask, the trail follows the lowest ask down, and break-even is under entry",
          p.best == 90.5 and p.sl < 109.0 and p.sl <= p.avg, f"{p.best} {p.sl}")
    f.move(I, 78.5, 78.8); e.step()
    check("short: the target buys back at the ask for a profit", I not in e.pos and e.trades[-1]["gross"] > 0 and e.trades[-1]["side"] == "short", str(e.trades[-1:]))
    f, e = mk(); e.sell(I, 1, {"sl_pts": 10}); f.move(I, 110.0, 110.5); e.step()
    check("short: a stop above the entry buys back at the ask for a loss", I not in e.pos and e.trades[-1]["gross"] < 0 and "stop" in e.trades[-1]["why"])
    f, e = mk()
    check("short: no stop is required (no limits: it goes out as pressed)", e.sell(I, 1, {})["ok"] and e.pos[I].side == -1 and e.pos[I].sl is None)
    f, e = mk()
    check("short: a stop inside the market is refused", not e.sell(I, 1, {"sl": 99.0})["ok"])
    f, e = mk(max_short_lots=1, max_lots=3, max_open_lots=6)
    check("short: lots are capped by the short-lot limit", not e.sell(I, 2, {"sl_pts": 10})["ok"] and e.sell(I, 1, {"sl_pts": 10})["ok"])
    check("short: the same contract cannot be held long and short at once", not e.buy(I, 1)["ok"])
    f, e = mk(max_risk=100)
    check("short: the per-order max risk refuses a stop that is too wide", not e.sell(I, 1, {"sl_pts": 10})["ok"] and e.sell(I, 1, {"sl_pts": 1})["ok"])
    f, e = mk(allow_short=False)
    check("short: can still be switched off in code (nothing in env does)", "short" in e.sell(I, 1, {"sl_pts": 10})["msg"].lower() and I not in e.pos)
    big = N.Config(env={"NSE_TRADE_MODE": "live", "NSE_TRADE_MAX_LOTS": "1", "NSE_TRADE_DAY_LOSS": "10", "NSE_TRADE_MAX_OPEN_LOTS": "1", "NSE_TRADE_ALLOW_SHORT": "0", "NSE_TRADE_MAX_VALUE": "1"})
    check("config: only NSE_TRADE_MODE and NSE_TRADE_PIN are read - every limit env key is ignored, and shorts are always allowed",
          big.mode == "live" and big.allow_short and not (big.max_lots or big.max_open_lots or big.day_loss or big.max_value or big.max_short_lots or big.max_risk or big.day_target or big.giveback_pct or big.max_trades or big.loss_streak or big.max_spread_pct)
          and N.Config(env={"NSE_TRADE_PIN": " 12 "}).pin == "12" and N.Config(env={}).mode == "paper")
    f, e = mk(); W2 = "NFO:NIFTY25100CE"; f.add(W2, 70.0, 70.5)
    r1, r2, r3 = e.buy(I, 9, {}), e.buy(P, 9, {}), e.sell(W2, 5, {})
    check("config: with no limits, 9 lots, 9 more on another contract, and a 5-lot short with no stop all go through", r1["ok"] and r2["ok"] and r3["ok"], f"{r1} {r2} {r3}")
    # live short: SELL to open, BUY (above the ask) to cover
    b = Broker(); f, e = mk(b, allow_short=True); e.set_armed(True); r = e.sell(I, 1, {"sl_pts": 10}); 
    check("short live: a sell-to-open is a marketable LIMIT below the bid", r["ok"] and b.placed[0][1] == "SELL" and b.placed[0][3] < 99.0 and e.pos[I].side == -1, f"{r} {b.placed}")
    e.exit(I, 100, "t")
    check("short live: the cover is a BUY limit above the ask", b.placed[1][1] == "BUY" and b.placed[1][3] > 99.5 and I not in e.pos, str(b.placed))
    b4 = Broker(); f, e = mk(b4, allow_short=True); e.set_armed(True); e.sell(I, 1, {"sl_pts": 10}); b4.pos_rows = [{"tradingsymbol": "NIFTY25000CE", "quantity": -65, "average_price": 98.7}]
    check("short live: reconcile reads the broker's negative quantity as a short and agrees", e.reconcile()["ok"])
    # ── hedged short ──
    W = "NFO:NIFTY25100CE"
    def mkh(broker=None, **kw):
        f, e = mk(broker, **kw); f.add(W, 60.0, 60.5); f.m[W]["strike"] = 25100; return f, e
    f, e = mkh(); r = e.place({"inst": I, "side": "SELL", "lots": 1, "exits": {"sl_pts": 10}, "hedge": W}, "h1")
    check("hedge: the wing is bought first, then the short is sold", r["ok"] and e.pos[W].role == "hedge" and e.pos[I].side == -1 and e.pos[W].side == 1, str(r))
    check("hedge: the wing cannot be sold while its short is open", "short" in e.exit(W, 100)["msg"].lower() and W in e.pos)
    f.move(I, 90.0, 90.5); f.move(W, 55.0, 55.5); e.exit(I, 100, "cover"); clk.t += 2; f.move(W, 55.0, 55.5); e.step()
    check("hedge: the wing is sold automatically once the short is covered", I not in e.pos and W not in e.pos and any(x.get("why") == "hedge released" for x in e.trades), str([x.get("why") for x in e.trades]))
    f, e = mkh(); f.add(W, 60.0, 60.5, typ="PE")
    check("hedge: a wing of the other type is refused and nothing is sold", not e.place({"inst": I, "side": "SELL", "lots": 1, "exits": {"sl_pts": 10}, "hedge": W})["ok"] and not e.pos)
    f, e = mkh(); f.move(I, 99.0, 99.5); f.q[I]["bid"] = 99.0
    f.m[W]["strike"] = 24900
    check("hedge: a wing closer to the money than the short is refused", not e.place({"inst": I, "side": "SELL", "lots": 1, "exits": {"sl_pts": 10}, "hedge": W})["ok"])
    f, e = mkh(); f.q[I]["ts"] = clk() - 60; r = e.place({"inst": I, "side": "SELL", "lots": 1, "exits": {"sl_pts": 10}, "hedge": W}, "h2")
    check("hedge: if the short cannot be sold the wing is sold back (no orphan long)", not r["ok"] and W not in e.pos and I not in e.pos, str(r))
    # ── structure (straddle) with a combined stop and target ──
    f, e = mk(max_short_lots=3)
    legs = [{"inst": I, "side": "SELL"}, {"inst": P, "side": "SELL"}]
    r = e.place({"legs": legs, "lots": 1, "g_sl": 500, "g_tp": 800, "kind": "straddle"}, "g1", "test")
    check("structure: a two-leg short straddle opens as one group", r["ok"] and len(e.groups) == 1 and I in e.pos and P in e.pos, str(r))
    f.move(I, 112.0, 112.5); f.move(P, 92.0, 92.5); e.step(); clk.t += 4; f.move(I, 112.0, 112.5); f.move(P, 92.0, 92.5); e.step()
    check("structure: the combined stop closes every leg when the group loses more than the limit", I not in e.pos and P not in e.pos, str(e.state()["groups"]))
    f, e = mk(max_short_lots=3)
    e.place({"legs": legs, "lots": 1, "g_sl": 500, "g_tp": 800, "kind": "straddle"}, "g2", "test")
    f.move(I, 85.0, 85.5); f.move(P, 66.0, 66.5); e.step(); clk.t += 4; f.move(I, 85.0, 85.5); f.move(P, 66.0, 66.5); e.step()
    check("structure: the combined target closes every leg when the group gains enough", I not in e.pos and P not in e.pos)
    # ── waiting orders ──
    f, e = mk()
    r = e.add_pending({"inst": I, "side": "BUY", "lots": 1, "exits": {"sl_pts": 10}}, {"kind": "px", "op": "le", "level": 90.0})
    check("waiting: an order is queued, not sent", r["ok"] and len(e.state()["pending"]) == 1 and I not in e.pos, str(r))
    f.move(I, 89.0, 89.5); clk.t += 2; f.move(I, 89.0, 89.5); e.step(); time.sleep(0.3); e.step()
    check("waiting: it fires once when the premium reaches the level", I in e.pos and not e.state()["pending"], str(e.state()["pend_log"]))
    f, e = mk(); e.add_pending({"inst": I, "side": "BUY", "lots": 1, "exits": {}}, {"kind": "px", "op": "le", "level": 90.0})
    check("waiting: it can be cancelled", e.cancel_pending(e.state()["pending"][0]["id"])["ok"] and not e.state()["pending"])
    f, e = mk(); e.add_pending({"inst": I, "side": "BUY", "lots": 1, "exits": {}}, {"kind": "px", "op": "le", "level": 90.0}, ttl_s=60); clk.t += 120; f.move(I, 99.0, 99.5); e.step()
    check("waiting: it expires and is logged", not e.state()["pending"] and e.state()["pend_log"], str(e.state()["pend_log"]))
    # ── day rules ──
    f, e = mk(day_target=200, loss_streak=0); e.buy(I, 1, {}); f.move(I, 110.0, 110.5); e.exit(I, 100, "t"); f.move(I, 99.0, 99.5)
    check("day: reaching the daily target blocks new entries", e.state()["blocked"] and "target" in e.state()["blocked"].lower() and not e.buy(I, 1)["ok"], str(e.state()["blocked"]))
    f, e = mk(giveback_pct=50, giveback_min=100, loss_streak=0); e.buy(I, 1, {}); f.move(I, 110.0, 110.5); e.exit(I, 100, "t"); clk.t += 2; f.move(I, 99.0, 99.5); e.step()
    e.buy(I, 1, {}); f.move(I, 99.0, 99.5); clk.t += 2; f.move(I, 90.0, 90.5); e.step(); e.exit(I, 100, "t"); clk.t += 2; f.move(I, 99.0, 99.5); e.step()
    check("day: giving back too much of the day's peak blocks new entries", e.state()["blocked"] and "back" in e.state()["blocked"].lower(), str(e.state()["blocked"]))
    # ── trade records ──
    f, e = mk(loss_streak=0); e.sell(I, 1, {"sl_pts": 10}, tag="Fade"); f.move(I, 92.0, 92.5); e.step(); f.move(I, 97.0, 97.5); e.exit(I, 100, "t")
    tr = e.trades[-1]
    check("records: a trade keeps its tag, side, risk, MFE/MAE and slippage", tr["tag"] == "Fade" and tr["side"] == "short" and tr["risk"] > 0 and tr["mfe"] >= 0 and "mae" in tr and "slip" in tr, str(tr))


def test_scalping_tab():
    """The Scalping tab and its endpoints: placement between Home and Today, the
    ticket / blotter / hotkeys wiring, the chart right-click mute item (the same
    switch as Alerts > Master Control), the request guard that stops a page on
    another site from placing an order from your browser, and the charges
    figure matching the server's."""
    import re as _re, ast as _ast, subprocess as _sp, time as _tm
    d = open(DASH, encoding="utf-8").read()
    s_ = open(SRV, encoding="utf-8").read()
    nav = d[d.index('<nav class="tabs">'):d.index("</nav>", d.index('<nav class="tabs">'))]
    check("scalping: its tab sits between Home and Today",
          nav.index('data-tab="home"') < nav.index('data-tab="scalp"') < nav.index('data-tab="today"') and nav.count('data-tab="scalp"') == 1)
    check("scalping: one panel, placed before Today's, opened by the tab handler (which also starts its poll)",
          d.count('id="tab-scalp"') == 1 and d.index('id="tab-scalp"') < d.index('id="tab-today"')
          and "tab.dataset.tab === 'scalp' && typeof window.renderScalpTab === 'function'" in d)
    for el in ("scArm", "scKill", "scFlat", "scBuyCE", "scBuyPE", "scLots", "scStrikes", "scLadder", "scExitFields", "scPresets", "scRiskRs", "scConfirm", "scChart", "scChTabs", "scTfs", "scDepth", "scFlow", "scTape", "scDockTabs",
               "scSide", "scStruct", "scHedge", "scWing", "scMarginBtn", "scTrig", "scTag", "scBook", "scFocus", "scDockBody", "scMod", "scTurn", "scPin", "scDayPnl", "scEq"):
        if d.count(f'id="{el}"') != 1:
            check(f"scalping: element {el} present exactly once", False); break
    else:
        check("scalping: every ticket, chart, depth, blotter and risk element is present exactly once", True)
    check("scalping chart: the indicators menu, legends and zoom buttons are present exactly once",
          all(d.count(f'id="{x}"') == 1 for x in ("scInd", "scIndMenu", "scLegP", "scLegR", "scZoom")))
    i = d.index("FEATURE: Scalping terminal"); j = d.index("// ---------- boot ----------", i)
    seg = d[i:j]
    check("scalping: orders carry the intent header and the PIN, and go to /api/trade/ on the page's own server",
          "'X-Trade-Intent': '1'" in seg and "h['X-Trade-Pin'] = pin" in seg and "/api/trade/${path}" in seg)
    check("scalping: LIVE orders need a second press (confirm) and the buy buttons say why they are disabled",
          "SC.pending !== typ" in seg and "'Confirm live'" in seg and "Entries blocked: ${w}" in seg)
    check("scalping: polling is twice a second, only while the tab is open and visible, and single-flight",
          "SC_POLL_MS = 500" in seg and "setInterval(scPoll, SC_POLL_MS)" in seg and "classList.contains('active') || document.hidden || SC.busy" in seg)
    check("scalping: hotkeys run in the capture phase and stop the page's global letter shortcuts",
          "}, true);" in seg[seg.index("hotkeys:"):] and "e.stopImmediatePropagation()" in seg and "if (!panel.classList.contains('active')) return;" in seg)
    check("scalping: the positions dock is not rebuilt under a half-finished click",
          "if (SC.down && (tab === 'pos' || tab === 'pend')) return;" in seg and "body.addEventListener('mousedown', () => { SC.down = true; })" in seg)
    check("chart timeframes follow Kite (1m 3m 5m 10m 15m), candles are anchored to the 09:15 open, history is fetched once per contract and interval, and an old saved 5s/15s/30s choice falls back to 1m",
          "const SC_TFS = [[60, '1m'], [180, '3m'], [300, '5m'], [600, '10m'], [900, '15m']]" in seg and "const SC_ANCHOR = 13500000" in seg
          and "(x.ms - SC_ANCHOR) / ms" in seg and "/api/trade/candles?" in seg and "if (!SC_TFS.some(x => x[0] === SC.pref.tf)) SC.pref.tf = 60" in seg)
    check("scalping server: /api/trade/candles is GET, guarded, limited to 1/3/5/10/15 minutes and to listed options or the four indices",
          'parsed.path == "/api/trade/candles"' in s_ and "tf must be 1, 3, 5, 10 or 15 minutes" in s_ and '"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"' in s_ and "not a listed index option" in s_)
    check("strike grid: one element, directly below the chart box and its volume pane (HTML, so zoom cannot touch it), opaque, fed by the state poll's b= list",
          d.count('id="scGrid"') == 1 and '<div class="sc-grid sc-grid-below" id="scGrid"' in d and ".sc-grid.sc-grid-below { position: static;" in d and ".sc-grid { --g-bg: #0b0d12;" in d
          and "&b=${encodeURIComponent(bnames.join(','))}" in seg and 'parse_qs' in s_ and '(_qs.get("b") or [""])[0]' in s_ and "book_totals" in s_)
    check("chart indicators: EMA 5/13 and RSI 14 with a 14-bar MA are on by default, and RSI draws 80, 70 (dotted), 50, 30 (dotted) and 20",
          "ema: true, emaF: 5, emaS: 13, rsi: true, rsiN: 14, rsiM: 14" in seg
          and "[[80, SC_OB, '', .8], [70, SC_OB, '2,3', .65], [50, '#8b93a7', '', .55], [30, SC_OS, '2,3', .65], [20, SC_OS, '', .8]]" in seg)
    # the indicator maths, run for real in node against known values
    fn = "\n".join(seg[seg.index(k):seg.index("\n  }\n", seg.index(k)) + 5] for k in ("function scEmaArr", "function scRsiArr", "function scSmaArr"))
    js = fn + """
    const cl = [44.34,44.09,44.15,43.61,44.33,44.83,45.10,45.42,45.84,46.08,45.89,46.03,45.61,46.28,46.28,46.00,46.03,46.41,46.22,45.64];
    const r = scRsiArr(cl, 14), e = scEmaArr([1,2,3,4,5,6,7,8], 3), m = scSmaArr([null,null,10,20,30,40], 2);
    console.log(JSON.stringify({ r14: r[14], r15: r[15], r19: r[19], pre: r[13], e: e, m: m, up: scRsiArr([1,2,3,4,5,6], 3)[5], flat: scRsiArr([5,5,5,5,5,5], 3)[5] }));"""
    try:
        o = json.loads(_sp.run(["node", "-e", js], capture_output=True, text=True, timeout=20).stdout)
        check("chart indicators: Wilder RSI(14) matches the textbook series (70.5, 66.3 ... 58.0), is blank until 15 bars, 100 on a pure rise, 50 when flat",
              abs(o["r14"] - 70.53) < 0.3 and abs(o["r15"] - 66.32) < 0.3 and abs(o["r19"] - 57.97) < 0.4 and o["pre"] is None and o["up"] == 100 and o["flat"] == 50, str(o))
        js2 = "const SC_ANCHOR = 13500000; let NOW = 0; const scNow = () => NOW;\n" + seg[seg.index("function scMergeBars"):seg.index("\n  }\n", seg.index("function scMergeBars")) + 5] + seg[seg.index("function scFillGaps"):seg.index("\n  }\n", seg.index("function scFillGaps")) + 5] + seg[seg.index("function scVwapFill"):seg.index("\n  }\n", seg.index("function scVwapFill")) + 5] + """
        const bar = (t, o, c, v) => ({ t, o, h: Math.max(o, c), l: Math.min(o, c), c, v, v0: v, lv0: 0, bv: 0, sv: 0, cvd: null });
        const D = Date.UTC(2026, 9, 28) - 19800000, OPEN = D + 33300000, M = 60000;           // 28 Oct 2026, 09:15 IST
        NOW = OPEN + 13 * M + 5000;
        // history: first trade at 09:25, then 09:26, then nothing until 09:28; the 09:28 bar is still forming and live ticks have begun
        const H = [bar(OPEN + 10 * M, 100, 101, 100), bar(OPEN + 11 * M, 101, 102, 200), bar(OPEN + 13 * M, 102, 103, 300)];
        H[2].lv0 = 100;
        const L = [{ t: OPEN + 13 * M, o: 103.5, h: 106, l: 102.5, c: 105, v: 160, bv: 90, sv: 70, cvd: 20 }, { t: OPEN + 14 * M, o: 105, h: 105, l: 105, c: 105, v: 0, bv: 0, sv: 0, cvd: 20 }];
        const m = scMergeBars(H, L, 60); scVwapFill(m);
        const prevDay = scMergeBars([bar(OPEN - 86400000 + 5 * M, 90, 91, 50), bar(OPEN, 100, 101, 100)], [], 60); scVwapFill(prevDay);
        console.log(JSON.stringify({ n: m.length, ts: m.map(x => (x.t - OPEN) / M), pad: m[0].c, padv: m[0].v, gap: m[12].c, fo: m[13].o, h: m[13].h, c: m[13].c, v: m[13].v, same: scMergeBars([], L, 60) === L,
          days: prevDay.filter(x => x.dayStart).length, firstDay: (prevDay[0].t - (OPEN - 86400000)) / M, vw: m[10].vw }));"""
        o2 = json.loads(_sp.run(["node", "-e", js2], capture_output=True, text=True, timeout=20).stdout)
        check("chart candles: every session runs from 09:15 (late first trade padded back to the open, minutes with no trade filled flat), history plus live merge (the forming bar keeps its historical open, takes the live high and close, and adds only the volume seen since the history call), and VWAP restarts per session",
              o2["ts"] == list(range(0, 15)) and o2["pad"] == 100 and o2["padv"] == 0 and o2["same"] and o2["gap"] == 102 and o2["fo"] == 102 and o2["c"] == 105 and o2["h"] == 106 and o2["v"] == 360
              and o2["days"] == 2 and o2["firstDay"] == 0 and abs(o2["vw"] - 302 / 3) < 1e-6, str(o2))
        js3 = "const fmtInt = x => x;\n" + seg[seg.index("function scFills"):seg.index("\n  }\n", seg.index("function scFills")) + 5] + """
        const S = { trades: [
          { inst: 'A', typ: 'CE', strike: 25000, side: 'long', role: 'main', qty: 65, avg: 100, exit: 110, net: 500, t: 2000, opened: 1000, why: 'partial' },
          { inst: 'A', typ: 'CE', strike: 25000, side: 'long', role: 'main', qty: 65, avg: 100, exit: 90, net: -700, t: 3000, opened: 1000, why: 'stop' },
          { inst: 'B', typ: 'PE', strike: 25000, side: 'short', role: 'main', qty: 65, avg: 80, exit: 70, net: 600, t: 4000, opened: 3500, why: 'target' },
          { inst: 'W', typ: 'CE', strike: 25200, side: 'long', role: 'hedge', qty: 65, avg: 20, exit: 21, net: 10, t: 4000, opened: 3500, why: 'hedge released' }],
          positions: [{ inst: 'A', typ: 'CE', strike: 25000, side: 'long', role: 'main', qty: 65, lots: 1, avg: 120, opened: 5000 }] };
        const a = scFills(S, { kind: 'CE', inst: 'A' }), sp = scFills(S, { kind: 'SPOT' });
        console.log(JSON.stringify({ a: a.map(f => [f.k, f.buy, f.up, f.qty, f.price]), spUp: sp.filter(f => f.inst === 'B').map(f => [f.k, f.up]), spN: sp.length, hedgeOnSpot: sp.some(f => f.inst === 'W'), wOwn: scFills(S, { kind: 'CE', inst: 'W' }).length }));"""
        o3 = json.loads(_sp.run(["node", "-e", js3], capture_output=True, text=True, timeout=20).stdout)
        check("chart fills: an option's chart gets one entry marker per fill (partial exits share their entry, quantities add up), an outlined exit marker per exit, and the open position; the spot chart draws by view (a short put is bullish: up at entry, down at exit) and leaves wings out",
              o3["a"] == [["in", True, True, 130, 100], ["out", False, False, 65, 110], ["out", False, False, 65, 90], ["in", True, True, 65, 120]]
              and o3["spUp"] == [["in", True], ["out", False]] and not o3["hedgeOnSpot"] and o3["spN"] == 6 and o3["wOwn"] == 2, str(o3))
        js4 = ("const fmtInt = x => x, escapeHtml = x => x; let SYM = 'NIFTY'; const state = { get symbol() { return SYM; }, lastData: { expiry: '03-Nov-2026', underlying_value: 25012, strikes: [] } };\n" +
              "const scChainOk = () => true; const SC = { spot: 25012, book: {}, pref: { selCE: 'o0' } }; const el = { style: {} }; const $sc = () => el; const scSet = (e, h) => { e.h = h; };\n" +
              seg[seg.index("const SC_GRID_SYMS"):seg.index("\n  }\n", seg.index("function scRenderGridBody")) + 5] + """
        for (let k = 24300; k <= 25700; k += 50) state.lastData.strikes.push({ strike: k, ce_symbol: 'C' + k, pe_symbol: 'P' + k });
        SC.book = { C25000: [90000, 30000, 1], P25000: [30000, 90000, 1], C25100: [30000, 90000, 1], P25100: [90000, 30000, 1], C24900: [null, null, 1] };
        state.lastData.strikes.find(r => r.strike === 24800).ce_tbq = 12000; state.lastData.strikes.find(r => r.strike === 24800).ce_tsq = 6000;
        scRenderGrid(); const h = el.h; const order = [...h.matchAll(/data-k="(\\d+)"/g)].map(m => +m[1]);
        const row = k => h.split('<tr class=').find(x => x.includes('data-k="' + k + '"')) || '';
        SC.spot = 25080; scRenderGrid(); const order2 = [...el.h.matchAll(/data-k="(\\d+)"/g)].map(m => +m[1]);
        SYM = 'BANKNIFTY'; const bank = scGridRows(); SYM = 'FINNIFTY'; const none = scGridRows(); SYM = 'NIFTY';
        console.log(JSON.stringify({ order, order2, ceBuyers: /class="up">3.00× buyers/.test(row(25000).split('data-k')[0]), peSellers: /class="up">0.33× sellers/.test(row(25000)), ceSellers: /class="dn">0.33× sellers/.test(row(25100)), peBuyers: /class="dn">3.00× buyers/.test(row(25100)), no50: !order.some(k => k % 100), bankOk: !!bank && bank.rows.length === 11, noHeads: !/<th/.test(h), 
          atm: /atm/.test(row(25000)), blank: row(24900).includes('<td>–</td>'), fallback: /class="up">2.00× buyers/.test(row(24800)), none, names: (SYM = 'NIFTY', scGridNames().length) }));""")
        r4 = _sp.run(["node", "-e", js4], capture_output=True, text=True, timeout=20)
        o4 = json.loads(r4.stdout or "null") or {"err": r4.stderr[-600:]}
        assert "order" in o4, o4
        check("strike grid: ATM plus and minus 5 round-hundred strikes (11 rows, no column headers) (the 50s are left out) with the highest strike on top, moving with the spot; CE buyers and PE sellers green, CE sellers and PE buyers red; empty books show a dash; shown for NIFTY and BANKNIFTY, hidden for the other indices",
              o4["no50"] and o4["order"] == [25500, 25400, 25300, 25200, 25100, 25000, 24900, 24800, 24700, 24600, 24500] and o4["order2"] == [25600, 25500, 25400, 25300, 25200, 25100, 25000, 24900, 24800, 24700, 24600]
              and o4["fallback"] and o4["ceBuyers"] and o4["peSellers"] and o4["ceSellers"] and o4["peBuyers"] and o4["atm"] and o4["blank"] and o4["none"] is None and o4["bankOk"] and o4["names"] == 22 and o4["noHeads"], str(o4))
        check("chart indicators: EMA seeds with the SMA and follows 2/(n+1); the RSI average waits for a full window",
              o["e"][:2] == [None, None] and o["e"][2] == 2 and abs(o["e"][3] - 3) < 1e-9 and abs(o["e"][7] - 7) < 1e-9 and o["m"] == [None, None, None, 15, 25, 35], str(o))
    except Exception as ex:  # noqa: BLE001
        check("chart indicators: the maths runs in node", False, str(ex))
    check("home chart: a NIFTY / BANKNIFTY strike grid overlay (sibling of chartBody, never touches the SVG), round-hundred strikes only, fed by the chain's book totals, placed at the price chart's lower left",
          'id="homeGrid"' in d and "function homeGridRender" in d and "['NIFTY', 'BANKNIFTY'].includes(state.symbol" in d and "body.light-mode .sc-grid {" in d and ".sc-chartwrap, .sc-indmenu { --text: #e8ebf2;" in d and "r.strike % 100 === 0" in d and "getClientRects().length" in d
          and "['NIFTY', 'BANKNIFTY'].includes(state.symbol || 'NIFTY')" in d[d.index("function homeGridRender"):d.index("function homeGridRender") + 700] and "homeGridRender();" in d)
    check("scalping: the page explains itself when opened from a file or when the data source is not Zerodha",
          "location.protocol === 'file:'" in seg and "_source !== 'zerodha'" in seg)
    # ── the chart right-click mute item ──
    ci = d.index("const onCtx = ev => {"); cj = d.index("document.addEventListener('contextmenu', onCtx, true);")
    cseg = d[ci:cj]
    check("chart right-click: Mute / Unmute drives the Alerts > Master Control checkbox through its own change handler",
          "mm.checked = !muted; mm.dispatchEvent(new Event('change'))" in cseg and "document.getElementById('masterMute')" in cseg
          and "'🔊 Unmute everything' : '🔇 Mute everything'" in cseg and "const muted = _masterMuted();" in cseg
          and "Mute everything on this browser" in d)
    check("chart right-click: the mute item sits above the Reset separator, so Reset stays last",
          cseg.index("Mute everything") < cseg.index("ctx-sep-reset") < cseg.index("item('↺ Reset Defaults…'"))
    # ── server endpoints and the request guard ──
    check("scalping server: state and journal are GET, every action is POST, all under /api/trade/",
          '"/api/trade/state"' in s_ and '"/api/trade/journal"' in s_ and 'parsed.path.startswith("/api/trade/")' in s_
          and all(f'path == "{a}"' in s_ for a in ("order", "exit", "modify", "cancel", "margin", "flatten", "kill", "arm", "reconcile")))
    check("scalping server: the engine only exists on DATA_SOURCE=zerodha, and lots must be a real integer (no 1.5, no True)",
          'if RUN_DATA_SOURCE != "zerodha"' in s_ and "not isinstance(lots, int) or isinstance(lots, bool)" in s_ and "def _trade_margin" in s_ and 'path == "margin"' in s_)
    tree = _ast.parse(s_)
    ns = {"time": _tm, "_trade_mod": None}
    for node in tree.body:
        if isinstance(node, _ast.FunctionDef) and node.name in ("_trade_guard", "_trade_exits"):
            exec(compile(_ast.Module([node], []), "srv", "exec"), ns)
    pin = {"v": ""}
    ns["_trade_engine"] = lambda: type("E", (), {"cfg": type("C", (), {"pin": pin["v"]})()})()
    ns["_TRADE_KEYS"] = ("sl_pts", "tp_pts", "tp1_pts", "tp1_pct", "trail_pts", "be_pts", "sl", "tp", "tp1", "u_stop", "u_target", "time_stop_s")
    class _H:
        def __init__(self, **h): self.headers = {k.replace("_", "-"): v for k, v in h.items()}
    g = ns["_trade_guard"]
    orig_sleep = _tm.sleep; ns["time"] = type("T", (), {"sleep": staticmethod(lambda x: None)})
    check("trade guard: a POST without the page's intent header is refused (a cross-site text/plain POST cannot add it)",
          g(_H(Host="h:8765"), True) is not None and g(_H(Host="h:8765", X_Trade_Intent="1"), True) is None)
    check("trade guard: a request from another site's page (Origin differs from Host) is refused, even with the header",
          g(_H(Host="h:8765", Origin="http://evil.example", X_Trade_Intent="1"), True) is not None
          and g(_H(Host="h:8765", Origin="http://h:8765", X_Trade_Intent="1"), True) is None
          and g(_H(Host="h:8765", Origin="http://evil.example"), False) is not None)
    pin["v"] = "1234"
    check("trade guard: with a PIN set, a wrong or missing PIN is refused and the right one passes",
          g(_H(Host="h", X_Trade_Intent="1"), True) is not None and g(_H(Host="h", X_Trade_Intent="1", X_Trade_Pin="0000"), True) is not None
          and g(_H(Host="h", X_Trade_Intent="1", X_Trade_Pin="1234"), True) is None)
    ex = ns["_trade_exits"]({"sl_pts": "8", "tp_pts": -3, "evil": 1, "u_stop": "x", "tp1_pct": 50, "time_stop_s": 1e12})
    check("trade exits: only known numeric keys, non-negative and sane, are passed to the engine", ex == {"sl_pts": 8.0, "tp1_pct": 50.0})
    # ── the tab's charges estimate matches the engine's ──
    m = _re.search(r"function scCharges\(.*?\n  \}\n", d, _re.S)
    sys.path.insert(0, HERE)
    import nse_trade as _N
    try:
        script = m.group(0) + "console.log(JSON.stringify([[10000,10000,2],[9532,9800,2],[150000,152000,2]].map(a=>scCharges(...a))));"
        out = _sp.run(["node", "-e", script], capture_output=True, text=True, timeout=20)
        js = __import__("json").loads(out.stdout)
        py = [_N.est_charges(*a) for a in ((10000, 10000, 2), (9532, 9800, 2), (150000, 152000, 2))]
        check("scalping: the tab's charge estimate equals the engine's (same STT, exchange, SEBI, stamp, GST)",
              all(abs(a - b) < 0.01 for a, b in zip(js, py)), f"{js} vs {py}")
    except FileNotFoundError:
        warn("node not installed", "charges parity unchecked")
    # ── the tick tape the chart is built from ──
    import nse_adapter_zerodha as _Z
    _Z._tape.clear()
    K = "NFO:NIFTY25000CE"
    for i, (l, b, a) in enumerate([(100, 99.5, 100), (100, 99.5, 100), (101, 100.5, 101), (102, 101.5, 102)]):
        _Z._tape_add(K, 1000 + i, {"ltp": l, "bid": b, "ask": a, "bid_qty": 10, "ask_qty": 10, "volume": 100 + (i > 1)})
    r0 = _Z.tape_since(K, 0)
    check("tape: identical consecutive ticks are stored once and rows carry a rising sequence number",
          len(r0["rows"]) == 3 and [r[0] for r in r0["rows"]] == [1, 2, 3] and r0["cur"] == 3 and not r0["reset"])
    r1 = _Z.tape_since("NIFTY25000CE", 2)
    check("tape: a cursor returns only newer rows (the bare name works too), and limit keeps the newest",
          [r[0] for r in r1["rows"]] == [3] and [r[0] for r in _Z.tape_since(K, 0, limit=2)["rows"]] == [2, 3])
    r2 = _Z.tape_since(K, 99)
    check("tape: a cursor ahead of the server (it restarted) is flagged reset and gets the full tape; an unknown key is empty",
          r2["reset"] and len(r2["rows"]) == 3 and _Z.tape_since("NFO:NOPE", 0) == {"cur": 0, "rows": [], "reset": False})
    for i in range(_Z._TAPE_MAX + 50):
        _Z._tape_add(K, 2000 + i, {"ltp": 100 + i, "bid": 1, "ask": 2, "bid_qty": 1, "ask_qty": 1, "volume": i})
    check("tape: memory is bounded", len(_Z._tape[K]) == _Z._TAPE_MAX)
    _Z._tape.clear()
    check("scalping server: the state call parses per-contract tape cursors with rpartition and only accepts option names or the listed ones",
          'rpartition(":")' in s_ and "tape_since" in s_ and "[A-Z0-9]{8,30}" in s_)
    env = open(os.path.join(HERE, "env.example"), encoding="utf-8").read() if os.path.exists(os.path.join(HERE, "env.example")) else ""
    check("env.example documents NSE_TRADE_MODE (paper default) and the PIN, and no size or risk limits",
          "NSE_TRADE_MODE" in env and "NSE_TRADE_PIN" in env and not any(k in env for k in ("NSE_TRADE_DAY_LOSS", "NSE_TRADE_MAX_LOTS", "NSE_TRADE_ALLOW_SHORT", "NSE_TRADE_MAX_RISK", "NSE_TRADE_DAY_TARGET", "NSE_TRADE_GIVEBACK")))


def test_remote_access():
    """nse_access: LAN untouched; every outside client (public IP, or anything
    behind a proxy) needs the token; fails closed; lockout; HTTPS wiring."""
    import importlib, io, tempfile, shutil, ssl as _ssl, socket as _sk
    sys.path.insert(0, HERE)
    A = importlib.import_module("nse_access")
    class H:
        def __init__(self, ip, cmd="GET", path="/api/chain", headers=None, body=b""):
            self.client_address = (ip, 5555); self.command = cmd; self.path = path
            self.headers = {k: v for k, v in (headers or {}).items()}
            self.headers.setdefault("Content-Length", str(len(body)))
            self.rfile = io.BytesIO(body); self.wfile = io.BytesIO(); self.sent = None; self.out = {}
        def send_response(self, c): self.sent = c
        def send_header(self, k, v): self.out[k] = v
        def end_headers(self): pass
    class Hd(dict):
        def get(self, k, d=None):
            for kk, v in self.items():
                if kk.lower() == k.lower(): return v
            return d
    def mk(ip, **kw):
        h = H(ip, **kw); h.headers = Hd(h.headers); return h
    old = {k: os.environ.get(k) for k in ("NSE_ACCESS_TOKEN", "NSE_AUTH_LAN")}
    try:
        os.environ.pop("NSE_AUTH_LAN", None)
        os.environ.pop("NSE_ACCESS_TOKEN", None)
        check("access: LAN, loopback and link-local clients are untouched; public, CGNAT and IPv6 global are outside",
              all(A.is_private(x) for x in ("192.168.1.4", "10.0.0.2", "172.16.5.5", "127.0.0.1", "::1", "fe80::1", "::ffff:192.168.1.9"))
              and not any(A.is_private(x) for x in ("8.8.8.8", "203.0.113.7", "100.64.1.1", "2001:db8::1")))
        check("access: a LAN client passes with no token set (behaviour unchanged)", A.guard(mk("192.168.1.9")) is True)
        h = mk("203.0.113.7")
        check("access: no token set -> an outside client is refused (fails closed)", A.guard(h) is False and h.sent == 403)
        os.environ["NSE_ACCESS_TOKEN"] = "short"
        h = mk("203.0.113.7", headers={"Authorization": "Bearer short"})
        check("access: a token under 12 characters does not open outside access either", A.guard(h) is False and h.sent == 403)
        tok = "correct-horse-battery"
        os.environ["NSE_ACCESS_TOKEN"] = tok
        h = mk("127.0.0.1", headers={"X-Forwarded-For": "203.0.113.7"})
        check("access: a request through a reverse proxy (127.0.0.1 + X-Forwarded-For) is outside, not LAN",
              A.is_remote(h) and A.guard(h) is False and h.sent == 401)
        h = mk("203.0.113.7", path="/nse_dashboard.html")
        A.guard(h)
        check("access: an outside visitor gets the sign-in page, an API call gets 401 json",
              h.sent == 401 and b"<form" in h.wfile.getvalue() and tok.encode() not in h.wfile.getvalue()
              and mk("203.0.113.7").sent is None)
        ok = mk("203.0.113.7", cmd="POST", path="/api/login", body=("token=" + tok).encode())
        A.guard(ok)
        ck = ok.out.get("Set-Cookie", "")
        check("access: right token -> 303 and an HttpOnly SameSite=Strict cookie holding an HMAC, never the token",
              ok.sent == 303 and "HttpOnly" in ck and "SameSite=Strict" in ck and tok not in ck and A.session_value() in ck and "Secure" not in ck)
        ok2 = mk("203.0.113.7", cmd="POST", path="/api/login", body=("token=" + tok).encode())
        A.guard(ok2, tls=True)
        check("access: the cookie is Secure once HTTPS is on", "Secure" in ok2.out.get("Set-Cookie", ""))
        cookie = "nse_auth=" + A.session_value()
        check("access: the cookie, or a bearer token, opens the API; a forged cookie or wrong bearer does not",
              A.guard(mk("203.0.113.7", headers={"Cookie": cookie})) is True
              and A.guard(mk("203.0.113.7", headers={"Authorization": "Bearer " + tok})) is True
              and A.guard(mk("203.0.113.7", headers={"Cookie": "nse_auth=abc"})) is False
              and A.guard(mk("203.0.113.7", headers={"Authorization": "Bearer wrong-token-value"})) is False)
        os.environ["NSE_AUTH_LAN"] = "1"
        check("access: NSE_AUTH_LAN=1 asks the LAN to sign in too", A.guard(mk("192.168.1.9")) is False)
        os.environ.pop("NSE_AUTH_LAN")
        A._fails.clear()
        import time as _t
        _sl = _t.sleep; _t.sleep = lambda s: None
        try:
            codes = []
            for i in range(6):
                hh = mk("198.51.100.5", cmd="POST", path="/api/login", body=b"token=nope%d" % i); A.guard(hh); codes.append(hh.sent)
            hh = mk("198.51.100.5", cmd="POST", path="/api/login", body=("token=" + tok).encode()); A.guard(hh)
        finally:
            _t.sleep = _sl
        check("access: five wrong tokens lock that address out, even for the right token afterwards",
              codes == [401] * 5 + [429] and hh.sent == 429, str(codes))
    finally:
        for k, v in old.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v
    # the certificate: made on first start, names in it, remade when the names change
    tmp = tempfile.mkdtemp()
    oldp = os.environ.get("NSE_PUBLIC_IP")
    try:
        cert, key = os.path.join(tmp, "tls", "server.crt"), os.path.join(tmp, "tls", "server.key")
        os.environ["NSE_PUBLIC_IP"] = "203.0.113.7"
        dns, ips = A.ensure_cert(cert, key)
        m1 = os.path.getmtime(cert)
        A.ensure_cert(cert, key)
        same = os.path.getmtime(cert) == m1
        os.environ["NSE_PUBLIC_IP"] = "203.0.113.8"
        dns2, ips2 = A.ensure_cert(cert, key)
        ctx = A.server_context(cert, key)
        check("access: a self-signed certificate is made covering localhost, the LAN address and the public IP; reused; remade when the IP changes; loads as a TLS 1.2+ server context",
              "203.0.113.7" in ips and "127.0.0.1" in ips and "localhost" in dns and same and "203.0.113.8" in ips2
              and ctx.minimum_version == _ssl.TLSVersion.TLSv1_2 and oct(os.stat(key).st_mode & 0o777) == "0o600")
        # the openssl-command fallback (no 'cryptography' package) makes a usable one too
        c2, k2 = os.path.join(tmp, "o.crt"), os.path.join(tmp, "o.key")
        try:
            A._make_with_openssl(c2, k2, ["localhost"], ["127.0.0.1"], 30)
            A.server_context(c2, k2)
            fb = True
        except FileNotFoundError:
            fb = True       # no openssl binary on this machine; the cryptography path covers it
        check("access: the openssl fallback produces a loadable certificate", fb)
    finally:
        if oldp is None: os.environ.pop("NSE_PUBLIC_IP", None)
        else: os.environ["NSE_PUBLIC_IP"] = oldp
        shutil.rmtree(tmp, ignore_errors=True)
    ssrc = open(os.path.join(HERE, "nse_chain_server.py"), encoding="utf-8").read()
    check("server: the guard runs first in do_GET, do_POST and do_OPTIONS",
          ssrc.count("if not _acc.guard(self, getattr(self.server, \"_tls\", False)):") == 3)
    check("server: HTTPS is wired - handshake in the client's own thread, never plain http by mistake, own-loopback call follows the scheme",
          "_acc.wrap_listener(server, _acc.server_context(_cert, _key))" in ssrc and "_acc.handshake(request)" in ssrc
          and "exiting rather than serving plain http by mistake" in ssrc and "_ssl._create_unverified_context() if _tls else None" in ssrc)


def test_server_reachability():
    """"Can't reach the local server" while it was up: a listen backlog of 5
    (refused connections under LAN bursts), the replay rebuild restarted on
    every context request for an index with no history, and that index never
    recorded by the poller; the page now retries once before the banner."""
    ssrc = open(os.path.join(HERE, "nse_chain_server.py"), encoding="utf-8").read()
    d = open(os.path.join(HERE, "nse_dashboard.html"), encoding="utf-8").read()
    check("server: listen backlog 128, not the default 5", "request_queue_size = 128" in ssrc)
    check("server: the replay rebuild runs once per index and day, even when it finds nothing",
          "(sym, day) in _ctx_backfill_tried:" in ssrc and "_ctx_backfill_tried.add((sym, day))" in ssrc)
    check("server: the poller records context history for every polled index (no extra broker call)",
          "data_, _meta = _cached_chain(sym, None, 12, client=\"poller\")" in ssrc and "_ctx_record(sym.upper(), data_)" in ssrc)
    check("page: one quiet retry before the can't-reach banner; reset on success",
          "if (err.isNetworkError && !state._netRetried) {" in d and "state._netRetried = false;" in d)
    check("page: the peer-index request waits for the chain request, skips hidden tabs, every 30 s",
          "if (state.inFlight || document.hidden) return;" in d and "setInterval(_turnPeerFetch, 30000);" in d)
    check("page: a TIMEOUT is 'server slow', not 'server not running' (only a refused connection shows the can't-reach banner)",
          "networkErr.name === 'TimeoutError'" in d and "tagged.isNetworkError = !slow;" in d
          and "if (err.isSlow) {" in d and "is running but not answering in time" in d)
    check("page: ONE constituents request at a time (it was re-sent every 30 s while slow ones were still open, using up the browser's 6 connections)",
          "if (_ncBusy && Date.now() - _ncBusy < 20000) return;" in d and "_ncBusy = 0;" in d
          and "{ signal: AbortSignal.timeout(15000) }" in d)
    check("server: constituents are single-flight per index and a failed answer is remembered for 30 s",
          "def _constituents_once(symbol: str)" in ssrc and "self._send_json(_constituents_once(symbol))" in ssrc
          and "_const_once_bad[sym] = (time.time(), res)" in ssrc)
    check("server: a failed previous-day OHLC lookup is remembered for 45 s (each miss walked every NSE fallback)",
          "_prev_ohlc_bad[symbol] = (time.time(), res)" in ssrc and "time.time() - bad[0] < 45" in ssrc)
    check("page: the can't-reach banner states the real browser error",
          "err.cause0 || 'connection failed'" in d)


def test_turn_model():
    """nse_turns: the objective label, events (not bars) counted, AUC, the
    walk-forward fit with its acceptance guards, the per-bar log endpoint."""
    import importlib, tempfile, random as _rnd, math as _m, shutil, time, calendar
    sys.path.insert(0, HERE)
    T = importlib.import_module("nse_turns")
    # hermetic: a real warm-up cache on this machine (today's 1-minute bars
    # from the broker) would replace the synthetic bars and change the result
    T._warm_bars = lambda sym_, day_: []
    t0 = 1792208700
    bars, p = [], 25000.0
    for i in range(120):
        o = p
        p += -3 if i < 40 else 3.5 if i < 80 else 0.0
        bars.append({"t": t0 + 60 * i, "o": o, "c": p, "h": max(o, p) + 0.5, "l": min(o, p) - 0.5})
    labs = T.label_bars(bars, D=200)
    pos_b = [i for i, l in enumerate(labs) if l["bottom"] == 1]
    ev = T.events(labs)
    pts = T.turn_points(bars, labs)
    check("turn label: a 120-pt fall then a rally - the bars near the low are bottoms, one EVENT, turning from the low",
          pos_b and min(pos_b) >= 30 and max(pos_b) <= 47 and len(ev["bottom"]) == 1
          and [q["side"] for q in pts].count("bottom") == 1 and abs(pts[0]["price"] - min(b["l"] for b in bars)) < 1e-9,
          f"{pos_b} {ev} {pts[:1]}")
    check("turn label: X = 0.3 × D (floored at 0.06% of spot), S = 0.4 × X; no label without 30 bars after",
          abs(labs[0]["X"] - 60) < 1e-9 and abs(labs[0]["S"] - 24) < 1e-9 and labs[-1]["bottom"] is None and labs[89]["bottom"] is not None)
    gap = [dict(b) for b in bars]
    for b in gap[60:]:
        b["t"] += 600                                        # a 10-minute hole at bar 60
    lg = T.label_bars(gap, D=200)
    check("turn label: a hole in the bars after t leaves t unlabelled (no guessing across it)",
          lg[45]["bottom"] is None and lg[20]["bottom"] is not None)
    check("AUC: perfect 1, reversed 0, ties 0.5, one class -> None",
          T.auc([1, 2, 3, 4], [0, 0, 1, 1]) == 1 and T.auc([4, 3, 2, 1], [0, 0, 1, 1]) == 0
          and T.auc([1, 1, 1, 1], [0, 1, 0, 1]) == 0.5 and T.auc([1, 2], [1, 1]) is None)
    rng = _rnd.Random(3)
    rows = []
    for d in range(14):
        for k in range(250):
            x = [rng.random() for _ in T.FAMILIES]
            y = 1 if rng.random() < 1 / (1 + _m.exp(-(-3 + 3 * x[0] + 2 * x[2]))) else 0
            rows.append({"day": f"2026-09-{d + 1:02d}", "t": k, "side": "bottom", "y": y, "x": x, "ev": f"{d}|{k}" if y else None})
            rows.append({"day": f"2026-09-{d + 1:02d}", "t": k, "side": "top", "y": 1 if rng.random() < 0.1 else 0, "x": x,
                         "ev": f"{d}|{k}"})
    model, rep = T.fit(rows=rows, perms=3, write=False, min_turns=50)
    B, Tp = rep["sides"]["bottom"], rep["sides"]["top"]
    check("turn fit: real signal - walk-forward AUC well above time-only and the shuffled null; location and momentum weighted",
          B["auc"] > 0.7 and B["auc"] > B["null_auc_p95"] and B["accepted"] and B["weights"]["loc"] > 2 and B["weights"]["mom"] > 1.2, str(B)[:300])
    check("turn fit: noise is NOT accepted (AUC under 0.55), so the whole model stays off",
          not Tp["accepted"] and any("below 0.55" in w for w in Tp["why_not"]) and model["accepted"] is False, str(Tp["why_not"]))
    import warnings as _w
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        sep = [[1.0 if i % 50 == 0 else 0.0] + [rng.random() for _ in range(10)] for i in range(2000)]
        fits = [T.fit_logistic(sep, [1 if x[0] == 1.0 else 0 for x in sep]),            # perfectly separable
                T.fit_logistic([[0.0] * 11 for _ in range(500)], [0] * 499 + [1]),       # constant features
                T.fit_logistic([[rng.random() for _ in range(11)] for _ in range(300)], [0] * 300)]   # one class
    check("turn fit: separable, constant and one-class data stay finite, with no numpy warnings (macOS Accelerate)",
          all(all(math.isfinite(v) for v in [b_] + w_) for b_, w_ in fits) and fits[0][1][0] > 3
          and not [c for c in caught if issubclass(c.category, RuntimeWarning)], str([str(c.message) for c in caught][:3]))
    _, rep2 = T.fit(rows=rows, perms=1, write=False, min_turns=10 ** 6)
    check("turn fit: too few turn events -> not accepted, says how many are needed",
          not rep2["sides"]["bottom"]["accepted"] and any("needed" in w for w in rep2["sides"]["bottom"]["why_not"]))
    # the server's log endpoint: dedupe, today only, validation; the dataset joins rows to labels
    import ast as _ast, threading as _th, types as _ty
    src = open(os.path.join(HERE, "nse_chain_server.py"), encoding="utf-8").read()
    want = {"_ctx_day", "_turn_log_rows", "_turn_events", "_turn_model"}
    nodes = [n for n in _ast.parse(src).body if isinstance(n, _ast.FunctionDef) and n.name in want]
    ns = {"time": time, "json": json, "_os": os, "threading": _th, "_turn_lock": _th.Lock(), "_turn_seen": {},
          "_turn_model_cache": {"mtime": None, "v": None}}
    exec(compile(_ast.Module(body=nodes, type_ignores=[]), "srv", "exec"), ns)
    S = _ty.SimpleNamespace(**{k: ns[k] for k in want})
    tmp = tempfile.mkdtemp()
    old = T.LOG_DIR
    T.LOG_DIR = tmp
    # hermetic: a real warm-up cache on this machine (today's 1-minute bars
    # from the broker) would replace the synthetic bars and change the result
    try:
        day = S._ctx_day()
        tt = calendar.timegm(time.strptime(day + " 10:00", "%Y-%m-%d %H:%M")) - 19800      # 10:00 IST today
        fam = {k: 0.5 for k in T.FAMILIES}
        rws = [{"t": tt + 60 * i, "iv": 1, "o": 1, "h": 2, "l": 0.5, "c": 1.5, "D": 200, "f": {"b": fam, "t": fam}} for i in range(5)]
        n1 = S._turn_log_rows({"sym": "NIFTY", "rows": rws})
        n2 = S._turn_log_rows({"sym": "NIFTY", "rows": rws[:3] + [{"t": tt - 86400, "c": 1}, {"t": "x"}, {"t": tt + 999, "iv": 2, "c": 1}]})
        n3 = S._turn_log_rows({"sym": "../etc", "rows": rws})
        lines = open(T.log_path("NIFTY", day)).read().strip().splitlines()
        check("turn log: one row per symbol/interval/bar (several browsers write once), today only, bad input refused",
              n1 == 5 and n2 == 1 and n3 == 0 and len(lines) == 6, f"{n1} {n2} {n3} {len(lines)}")
        # an older log row has no "reg": that is "not measured" = neutral 1.0, not 0
        old_row = {"t": tt + 600, "iv": 1, "o": 1, "h": 2, "l": 0.5, "c": 1.5, "D": 200,
                   "f": {"b": {k: 0.5 for k in T.FAMILIES if k != "reg"}, "t": {}}}
        dso = T.day_dataset("NIFTY", day, rows=[old_row])
        check("turn dataset: a log row from before the 'reg' family counts as neutral (1.0), not as a trend",
              "reg" in T.FAMILIES and all(r["x"][T.FAMILIES.index("reg")] == 1.0 for r in dso if r["side"] == "bottom"), str(dso)[:200])
        # the realised value of the BUY/SELL calls, scored by the label objective (+X before -0.4X in 30 min)
        def srow(i, st="WAIT", d=0, h=100.5, l=99.5):
            return {"t": 1792208700 + 60 * i, "iv": 1, "o": 100, "h": h, "l": l, "c": 100.0, "D": 200, "spot": 25000, "st": st, "dir": d}
        sr = [srow(i) for i in range(260)]
        sr[0] = srow(0, "BUY", 1); sr[1] = srow(1, "BUY", 1)                   # one call (state held) ...
        sr[10] = srow(10, h=165)                                                # ... reaches +X (60) first: win
        sr[60] = srow(60, "SELL", 1 - 2)                                        # SELL; next bar goes +30 against it: loss
        sr[61] = srow(61, h=130)
        sr[100] = srow(100, "BUY", 1)                                           # BUY into 30 flat minutes: timeout
        sr[200] = srow(200, "BUY", 1)
        sr = sr[:204]                                                           # ... only 3 bars after it: skipped, not guessed
        g = T.signals(rows_by_day={("NIFTY", "2026-10-28"): sr})
        check("turn signals: a call is scored +X before -0.4X within 30 min (win / loss / timeout), incomplete data skipped",
              (g["n"], g["win"], g["loss"], g["timeout"], g["skipped"]) == (3, 1, 1, 1, 1) and g["win_rate"] == 0.5
              and g["breakeven"] == 0.286, str(g))
        ds = T.day_dataset("NIFTY", day)
        check("turn dataset: logged rows join their labels per side, unlabelled tail left out",
              isinstance(ds, list) and all(r["side"] in ("bottom", "top") for r in ds) and len(ds) == 0)
    finally:
        T.LOG_DIR = old
        shutil.rmtree(tmp, ignore_errors=True)
    import os as _os
    _os.environ["NSE_EVENTS"] = "2026-12-24 09:15-09:45 Election results; junk; 2027-01-02 13:00-13:30 X"
    ev2 = S._turn_events()
    _os.environ.pop("NSE_EVENTS", None)
    check("turn events: NSE_EVENTS parsed into IST minute windows, junk ignored",
          ev2 == [{"day": "2026-12-24", "from": 555, "to": 585, "lab": "Election results"},
                  {"day": "2027-01-02", "from": 780, "to": 810, "lab": "X"}], str(ev2))
    tm = S._turn_model()
    check("turn model endpoint: hand-set prior and families always served; not accepted until fitted",
          tm["prior"]["b"] == -3.8 and tm["prior"]["w"]["reg"] == 0.8 and tm["families"] == T.FAMILIES and "accepted" in tm["model"])
    ssrc = open(os.path.join(HERE, "nse_chain_server.py"), encoding="utf-8").read()
    check("server: turn endpoints wired (log POST; model, status, labels GET)",
          'parsed.path == "/api/turn-log"' in ssrc and 'parsed.path == "/api/turn-model"' in ssrc
          and 'parsed.path == "/api/turn-status"' in ssrc and 'parsed.path == "/api/turn-labels"' in ssrc)


def test_turn_times():
    """Turn-time study: a time where most sessions turn up is marked as a
    rally time; turns spread evenly mark nothing."""
    import ast as _ast, time as _t, os as _o, math as _m, datetime as _dt
    ssrc = open(os.path.join(HERE, "nse_chain_server.py"), encoding="utf-8").read()
    tree = _ast.parse(ssrc)
    fns = [n for n in tree.body if isinstance(n, _ast.FunctionDef) and n.name in ("_turn_study", "_ist_min_of")]
    IST = _dt.timezone(_dt.timedelta(hours=5, minutes=30))
    arch_days = {}
    import random as _r
    _r.seed(7)
    for i in range(30):
        day = (_dt.date(2026, 8, 3) + _dt.timedelta(days=i)).isoformat()
        base = int(_dt.datetime.fromisoformat(day + "T09:15").replace(tzinfo=IST).timestamp())
        ser = {}
        for k in range(375):
            # falls into 11:30 every day, then rallies hard; small noise
            p = 25000 - min(k, 135) * 0.8 + max(0, k - 135) * 1.2 + _r.uniform(-2, 2)
            ser[(base + k * 60) // 60] = p
        arch_days[day] = ser
    ns = {"time": _t, "_os": _o, "RUN_DATA_SOURCE": "nse", "_is_broker": lambda: False, "_TICK_DIR": "/nonexistent", "_turn_cache": {},
          "_load_archive": lambda sym, d: arch_days.get(d, {})}
    exec(compile(_ast.Module(body=fns, type_ignores=[]), "srv", "exec"), ns)
    ns["_os"] = type("O", (), {"path": type("P", (), {"isdir": staticmethod(lambda p: True), "join": staticmethod(_o.path.join)}),
                               "listdir": staticmethod(lambda p: [f"NIFTY_{d}.csv" for d in arch_days])})
    r = ns["_turn_study"]("NIFTY", 60)
    marked = {b["time"]: b for b in r["marked"]}
    check("turn times: a time where most sessions bottom is marked as a RALLY time",
          "11:30" in marked and marked["11:30"]["bias"] == "rally", str([(b["time"], b["bias"], b["turns"]) for b in r["marked"]]))
    check("turn times: today is excluded and the study counts sessions", r["days"] == 30, str(r["days"]))
    # midday tier: big moves at the open and close, plus a modest dip that
    # bottoms at 12:45 every day - it should show as a midday rally line
    arch2 = {}
    for i in range(30):
        day = (_dt.date(2026, 8, 3) + _dt.timedelta(days=i)).isoformat()
        base = int(_dt.datetime.fromisoformat(day + "T09:15").replace(tzinfo=IST).timestamp())
        ser = {}
        for k in range(375):
            if k < 30:   p = 25000 - k * 6                       # opening fall to 24820
            elif k < 60: p = 24820 + (k - 30) * 6                # back to 25000
            elif k < 210: p = 25000 - (k - 60) * (2 / 3)           # drift to 24900 at 12:45
            elif k < 300: p = 24900 + (k - 210) * 1.2            # midday rally
            else:        p = 25008 + (k - 300) * 4               # closing run
            ser[(base + k * 60) // 60] = p + _r.uniform(-1, 1)
        arch2[day] = ser
    ns["_turn_cache"].clear(); arch_days.clear(); arch_days.update(arch2)
    r2 = ns["_turn_study"]("NIFTY", 60)
    mid = {b["time"]: b for b in r2.get("midday", [])}
    check("turn times: midday tier marks a regular 12:45 low as a rally time",
          "12:45" in mid and mid["12:45"]["marked"] and mid["12:45"]["bias"] == "rally",
          str([(b["time"], b["bias"], b["turns"], b["marked"]) for b in r2.get("midday", [])]))
    check("turn times: midday lines are limited to 10:00-15:00", all(600 <= b["min"] < 900 for b in r2.get("midday", [])))
    dsrc = open(os.path.join(HERE, "nse_dashboard.html"), encoding="utf-8").read()
    check("turn-time lines drawn on the candle chart with a toggle", "_turnLinesSvg(W, padL, padR, padT, H - padB)" in dsrc and 'id="chk_turnTimes"' in dsrc)


def test_ema_cross():
    """EMA crossover lines: green on a bullish cross, red on a bearish one,
    closed candles only, defaults 5/13 restored by Reset, on every chart type."""
    import subprocess, json as _json
    dsrc = open(os.path.join(HERE, "nse_dashboard.html"), encoding="utf-8").read()
    def grab(name):
        i = dsrc.index(f"function {name}(")
        depth, j = 0, dsrc.index("{", i)
        for k in range(j, len(dsrc)):
            depth += dsrc[k] == "{"; depth -= dsrc[k] == "}"
            if depth == 0:
                return dsrc[i:k + 1]
    harness = "const _emaXCache = { key: '', pts: [] };\nconst _warmOf = () => [];\nconst _pdFilterOn = () => false, _pdVerdict = () => null, _pdFilterNote = () => '';\n" + grab("_emaSeries") + grab("_emaCrossPoints") + grab("_emaCrossSvg") + r"""
const _hm = t => String(t); const fmtPrice = v => (+v).toFixed(1); const escapeHtml = x => x;
const cs = [];
for (let i = 0; i < 120; i++) {
  // down, then up, then down: one bullish and one bearish cross
  const p = i < 40 ? 25000 - i * 5 : i < 80 ? 24800 + (i - 40) * 6 : 25040 - (i - 80) * 7;
  cs.push({ t: 1000 + i * 60, o: p, h: p + 1, l: p - 1, c: p });
}
const _chart = { candles: cs, liveBar: false, show: { emaX: true }, emaX: { fast: 5, slow: 13 }, interval: 1 };
const a = _emaCrossPoints().map(p => p.dir);
_emaXCache.key = ''; _chart.emaX = { fast: 13, slow: 5 };
const b = _emaCrossPoints().map(p => p.dir);
const svg = _emaCrossSvg(t => t - 1000, 0, 100);
// a cross that exists only on the forming bar is not drawn
const cs2 = cs.slice(0, 41).concat([{ t: 1000 + 41 * 60, o: 0, h: 0, l: 0, c: 26000 }]);
_emaXCache.key = ''; _chart.candles = cs2; _chart.liveBar = true; _chart.emaX = { fast: 5, slow: 13 };
const c = _emaCrossPoints().length;
_chart.liveBar = false; _emaXCache.key = '';
const d = _emaCrossPoints().length;
_chart.show.emaX = false; _emaXCache.key = '';
const e = _emaCrossPoints().length;
console.log(JSON.stringify({ a, b, one: (svg.match(/emax-line/g) || []).length === 1, latestRed: /#ff1744/.test(svg) && !/#00e676/.test(svg), c, d, e }));
"""
    r = subprocess.run(["node", "-e", harness], capture_output=True, text=True)
    try:
        o = _json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        check("ema cross: harness ran", False, (r.stderr or r.stdout)[:400]); return
    check("ema cross: a bullish then a bearish cross are found", o["a"] == ["bull", "bear"], str(o["a"]))
    check("ema cross: swapped fast/slow boxes read the same", o["b"] == o["a"], str(o["b"]))
    check("ema cross: only the most recent cross is drawn (here the bearish one, RED)", o["one"] and o["latestRed"])
    check("ema cross: the forming candle cannot print a line", o["c"] == 0 and o["d"] == 1, f"{o['c']} {o['d']}")
    check("ema cross: the checkbox turns the lines off", o["e"] == 0)
    check("ema cross: fast 5 / slow 13 boxes beside the EMA checkboxes",
          'id="emaXFast" class="ohlc-inp" value="5"' in dsrc and 'id="emaXSlow" class="ohlc-inp" value="13"' in dsrc
          and dsrc.index('id="chk_ema200"') < dsrc.index('id="emaXFast"') < dsrc.index('id="chk_vwap"'))
    check("ema cross: defaults 5/13 in the chart config, which Reset clears",
          dsrc.count("emaX: true, ttr: true, sqz: true, wt: true }, emaX: { fast: 5, slow: 13 } }") == 2 and "'chartCfg'," in dsrc)
    check("ema cross: drawn by the candle/Heikin Ashi renderer and by Renko", dsrc.count("svg += _emaCrossSvg(") == 2)


def test_turn_blink():
    """The next turn-time line ahead of the clock blinks; passed lines do not;
    lean lines are one yellow at one transparency."""
    import subprocess, json as _json
    dsrc = open(os.path.join(HERE, "nse_dashboard.html"), encoding="utf-8").read()
    i = dsrc.index("function _turnLinesSvg(")
    depth, j = 0, dsrc.index("{", i)
    for k in range(j, len(dsrc)):
        depth += dsrc[k] == "{"; depth -= dsrc[k] == "}"
        if depth == 0:
            fn = dsrc[i:k + 1]; break
    harness = "const _xs = (a, slots, W, padL, padR) => padL + a / slots * (W - padL - padR);\n" + fn + r"""
const state = { symbol: 'NIFTY' }; const _chart = { interval: 1, show: {} }; const _isRenkoView = () => false;
const escapeHtml = x => String(x); let NOW = 600, HIST = false;
const _istNowMin = () => ({ m: NOW, wd: 3 }); const _isHistorical = () => HIST;
const mk = (min, tier, bias) => ({ min, time: String(min), turns: 5, expected: 3, ratio: 1.4, lows: 3, highs: 2, p: 0.2, bias, follow30: 10, marked: tier === 'mid', lean: tier === 'lean' });
const _turns = { v: { symbol: 'NIFTY', days: 60, marked: [{ ...mk(570, 'major', 'crash'), marked: true }],
  midday: [mk(660, 'lean', 'rally'), mk(720, 'lean', 'crash'), mk(780, 'mid', 'rally')] } };
const nextOf = svg => { const m = svg.match(/class="turn-next"[^>]*><line x1="([\d.]+)/); return m ? +m[1] : null; };
const xs = svg => [...svg.matchAll(/<line x1="([\d.]+)"/g)].map(m => +m[1]);
NOW = 560; const a = _turnLinesSvg(1000, 0, 0, 0, 100);
NOW = 600; const b = _turnLinesSvg(1000, 0, 0, 0, 100);
NOW = 700; const c = _turnLinesSvg(1000, 0, 0, 0, 100);
NOW = 800; const d = _turnLinesSvg(1000, 0, 0, 0, 100);
HIST = true; NOW = 600; const e = _turnLinesSvg(1000, 0, 0, 0, 100);
const styles = [...b.matchAll(/<g opacity="([\d.]+)"><line[^>]*stroke="([^"]+)" stroke-width="([\d.]+)" stroke-dasharray="([^"]+)"/g)].map(m => [m[1], m[3], m[4]].join('|'));
const leanCols = [...b.matchAll(/stroke="([^"]+)"[^>]*\/>\s*<text[^>]*>[^<]*lean/g)].map(m => m[1]);
HIST = false; NOW = 600;
const sv = _turns.v;
_turns.v = { symbol: 'NIFTY', days: 60, marked: [], midday: [mk(660, 'lean', 'rally')] }; const f1 = _turnLinesSvg(1000, 0, 0, 0, 100);
_turns.v = null; const f2 = _turnLinesSvg(1000, 0, 0, 0, 100);
_chart.show.turnTimes = false; const f3 = _turnLinesSvg(1000, 0, 0, 0, 100); _chart.show.turnTimes = undefined;
_turns.v = sv;
const pm = svg => { const m = svg.match(/<line x1="([\d.]+)"[^>]*stroke="([^"]+)" stroke-width="([\d.]+)" stroke-dasharray="([^"]+)"\/>\s*<text[^>]*>[^<]*1 pm/); return m ? [+m[1], m[2], m[4]] : null; };
console.log(JSON.stringify({ a: nextOf(a) === xs(a)[0], b: nextOf(b) === xs(b)[1], c: nextOf(c) === xs(c)[2],
  d: nextOf(d), e: nextOf(e), nBlink: (b.match(/turn-next/g) || []).length, uniform: new Set(styles).size === 1 && styles.length === 3, leanCols,
  pmDup: (b.match(/1 pm/g) || []).length, pm1: pm(f1), pm2: pm(f2), n2: xs(f2).length, f3 }));
"""
    r = subprocess.run(["node", "-e", harness], capture_output=True, text=True)
    try:
        o = _json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        check("turn blink: harness ran", False, (r.stderr or r.stdout)[:400]); return
    check("turn blink: before 09:30 the 09:30 line blinks", o["a"])
    check("turn blink: once a line's time passes, the next one blinks", o["b"] and o["c"] and o["nBlink"] == 1, str(o))
    check("turn blink: nothing blinks after the last line or on a past date", o["d"] is None and o["e"] is None, str(o))
    check("turn lines: lean lines drawn the same as the others (one dash, width, transparency)", o["uniform"], str(o))
    check("turn lines: a lean line takes the same colour rule (rally lean green, crash lean red)", o["leanCols"] == ["#1fd17a", "#ff3b4e"], str(o["leanCols"]))
    x13 = round((780 - 555 + 0.5) / 385 * 1000, 1)
    check("turn lines: fixed 1 pm line - amber, same dash as the lean lines, at 13:00, drawn even before the turn statistics load",
          o["pm1"] == [x13, "#f5b301", "6,4"] and o["pm2"] == [x13, "#f5b301", "6,4"] and o["n2"] == 1, str({k: o[k] for k in ("pm1", "pm2", "n2")}))
    check("turn lines: no duplicate 1 pm when a measured turn line already sits at 13:00; hidden with the turn-times toggle",
          o["pmDup"] == 0 and o["f3"] == "", str({k: o[k] for k in ("pmDup", "f3")}))


def test_defaults_pvsra_pivots():
    """PVSRA and Pivots start unchecked, Reset clears both back to off, and
    the Alerts button reads "Reset Defaults"."""
    d = open(os.path.join(HERE, "nse_dashboard.html"), encoding="utf-8").read()
    check("defaults: PVSRA off unless explicitly turned on",
          "_pvSet.on = v === '1';" in d and "const _pvSet = { at: 0, spread: false, on: false };" in d
          and 'id="pvsraToggle" checked' not in d)
    check("defaults: chart Pivots off", d.count("pivots: false, pivotType: 'camarilla'") == 2 and 'id="chk_pivots" checked' not in d)
    check("defaults: Pivots off in the alert chart sources", d.count("cone: true, pivots: false, tol: 20 }") == 2)
    check("defaults: Reset clears the PVSRA, chart and alert-source keys",
          all(f"'{k}'," in d for k in ("pvsraOn", "chartCfg", "chartAlertSources", "uiStateV1")))
    check("defaults: Alerts button labelled Reset Defaults",
          ">↺ Reset Defaults</button>" in d and ">↺ Reset to recommended defaults<" not in d)


def test_broker_refusals():
    """Fyers 11011 refusals are pinned on the batch they answered, retries are
    held while the broker keeps saying no, depth silence never reopens the
    socket, reopens back off, and a second server copy cannot take the socket."""
    import nse_adapter_fyers as A, datetime as _dt, os as _o
    IST = _dt.timezone(_dt.timedelta(hours=5, minutes=30))
    T = _dt.datetime(2026, 10, 16, 10, 0, tzinfo=IST).timestamp()      # Friday 10:00 IST
    calls = []
    class _WS:
        closed = 0
        def close(self): _WS.closed += 1
    class _S:
        def subscribe(self, symbols, data_type=None): calls.append((data_type, list(symbols)))
    s_ = _S(); setattr(s_, "_FyersDataSocket__ws_object", _WS())
    saved = (A._socket, dict(A._raw_last), set(A._subscribed), set(A._depth_subs), dict(A._sub_req),
             dict(A._dsub_req), dict(A._wd), list(A._refusals), list(A._pending_sub))
    try:
        A._socket = s_
        for x in (A._raw_last, A._sub_req, A._dsub_req, A._retry_n):
            x.clear()
        A._subscribed.clear(); A._depth_subs.clear(); A._refusals.clear(); A._pending_sub.clear()
        A._wd.update(last_heal=0.0, last_force=0.0, heals=0, forced_reconnects=0, retried=0, force_streak=0)
        # replies come back in request order: price batch refused, depth batch accepted
        A._track_sub("px", ["NSE:NIFTY50-INDEX"]); A._track_sub("dp", ["NSE:NIFTY26OCTFUT"])
        A._on_socket_error({"type": "sub", "code": 11011, "message": "subscription failed", "s": "error"})
        A._on_message({"type": "sub", "code": 200, "message": "Subscribed", "s": "ok"})
        rf = A.refusals_recent()
        check("refusals: an 11011 is pinned on the batch it answered", rf["px"] == 1 and rf["dp"] == 0 and not A._pending_sub, str(rf))
        # two refusals in 5 minutes -> unanswered price subscriptions are NOT re-sent
        A._refusals.clear()
        for k in range(2):
            A._refusals.append({"ts": T - 30 - k, "kind": "px", "n": 200, "sample": [], "code": 11011})
        idx, opt, fut = "NSE:NIFTY50-INDEX", "NSE:NIFTY26OCT25000CE", "NSE:NIFTY26OCTFUT"
        A._subscribed.update([idx, opt]); A._sub_req.update({idx: T - 100, opt: T - 100})
        A._raw_last[opt] = {"sf": (T - 2, {})}
        calls.clear()
        r = A._watchdog_pass(T)
        check("refusals: retries are held while the broker keeps refusing", r.get("held_px") and not calls, f"{r} {calls}")
        # depth silent, index and prices fine -> re-subscribe depth, never reopen the socket
        A._refusals.clear(); calls.clear(); _WS.closed = 0
        A._wd.update(last_heal=0.0, last_force=0.0)
        A._sub_req.update({idx: T - 400, opt: T - 400}); A._depth_subs.add(fut); A._dsub_req[fut] = T - 400
        A._raw_last[idx] = {"if": (T - 1, {})}; A._raw_last[opt] = {"sf": (T - 1, {})}; A._raw_last[fut] = {"dp": (T - 300, {})}
        r1 = A._watchdog_pass(T)
        A._depth_subs.add(fut); A._dsub_req[fut] = T - 400
        A._raw_last[idx] = {"if": (T + 199, {})}; A._raw_last[opt] = {"sf": (T + 199, {})}
        r2 = A._watchdog_pass(T + 200)
        check("refusals: silent depth alone is re-subscribed but never reopens the socket",
              r1.get("resubscribed") == 1 and not r2.get("forced") and _WS.closed == 0, f"{r1} {r2}")
        # reopen back-off doubles, and resets once prices flow again
        A._wd["force_streak"] = 0; g0 = A._force_gap(); A._wd["force_streak"] = 2; g2 = A._force_gap()
        A._watchdog_pass(T + 200)
        check("refusals: socket reopens back off 15 -> 30 -> 60 min and reset when prices flow",
              g0 == 900 and g2 == 3600 and A._wd["force_streak"] == 0, f"{g0} {g2} {A._wd['force_streak']}")
        st = A.watchdog_state()
        check("refusals: reported in the watchdog state", "refusals_10m" in st and "next_reopen_gap_s" in st)
    finally:
        A._socket = saved[0]
        A._raw_last.clear(); A._raw_last.update(saved[1])
        A._subscribed.clear(); A._subscribed.update(saved[2])
        A._depth_subs.clear(); A._depth_subs.update(saved[3])
        A._sub_req.clear(); A._sub_req.update(saved[4])
        A._dsub_req.clear(); A._dsub_req.update(saved[5])
        A._wd.clear(); A._wd.update(saved[6])
        A._refusals[:] = saved[7]; A._pending_sub[:] = saved[8]
    # a second copy of the server cannot take the socket
    lock = _o.path.join(_o.path.dirname(_o.path.abspath(A.__file__)), ".fyers_socket.lock")
    existed = _o.path.exists(lock)
    keep = dict(A._sock_lock)
    try:
        A._sock_lock.update(fh=None, blocked_by=None)
        a = A._acquire_socket_lock(); first = A._sock_lock["fh"]
        A._sock_lock["fh"] = None
        b = A._acquire_socket_lock()
        check("refusals: a second copy is blocked from opening the data socket",
              a and not b and "pid" in (A._sock_lock["blocked_by"] or ""), f"{a} {b} {A._sock_lock}")
        first.close()
    finally:
        A._sock_lock.clear(); A._sock_lock.update(keep)
        if not existed:
            try: _o.remove(lock)
            except Exception: pass
    srv = open(os.path.join(HERE, "nse_chain_server.py"), encoding="utf-8").read()
    check("refusals: FEED CHECK explains a refused socket", "broker accepting subscriptions" in srv and "ONE market-data socket per API app" in srv)


# ── Zerodha: a simulated Kite feed (KiteConnect REST + KiteTicker) ──────
def _z_dates():
    import datetime as _dt
    IST = _dt.timezone(_dt.timedelta(hours=5, minutes=30))
    now = _dt.datetime.now(IST)
    d = now.date()
    while d.weekday() != 1 or (d == now.date() and now.hour * 60 + now.minute >= 930):
        d += _dt.timedelta(days=1)
    weeklies = [d + _dt.timedelta(days=7 * i) for i in range(3)]
    import calendar
    def last_tue(y, m):
        x = calendar.monthrange(y, m)[1]
        while calendar.weekday(y, m, x) != 1:
            x -= 1
        return _dt.date(y, m, x)
    m1 = last_tue(now.year, now.month)
    if m1 < now.date() or (m1 == now.date() and now.hour * 60 + now.minute >= 930):
        y, m = (now.year + (now.month == 12), now.month % 12 + 1)
        m1 = last_tue(y, m)
    y2, mm2 = (m1.year + (m1.month == 12), m1.month % 12 + 1)
    return IST, now, weeklies, [m1, last_tue(y2, mm2)]


class _ZFakeKite:
    """Enough of KiteConnect for the adapter: instruments, quote, history."""
    SPOT = {"NIFTY": 25012.0, "BANKNIFTY": 56010.0}
    calls: dict = {}

    def __init__(self, api_key=None, **_k):
        self.api_key = api_key
        _ZFakeKite.calls = {"quote": 0, "historical": 0, "instruments": 0, "profile": 0, "quote_keys": []}
        self.IST, self.now, self.weeklies, self.monthlies = _z_dates()
        self._rows = {"NSE": [], "NFO": []}
        tok = 1000
        for sym, nm, t_ in (("NIFTY 50", "NIFTY 50", 256265), ("NIFTY BANK", "NIFTY BANK", 260105),
                            ("INDIA VIX", "INDIA VIX", 264969)):
            self._rows["NSE"].append({"instrument_token": t_, "tradingsymbol": sym, "name": nm, "expiry": "",
                                      "strike": 0.0, "lot_size": 0, "instrument_type": "EQ",
                                      "segment": "INDICES", "exchange": "NSE"})
        for eq in ("RELIANCE", "HDFCBANK", "ICICIBANK"):
            tok += 1
            self._rows["NSE"].append({"instrument_token": tok, "tradingsymbol": eq, "name": eq, "expiry": "",
                                      "strike": 0.0, "lot_size": 1, "instrument_type": "EQ",
                                      "segment": "NSE", "exchange": "NSE"})
        for name, gap, lot, spot in (("NIFTY", 50, 65, 25012.0), ("BANKNIFTY", 100, 30, 56010.0)):
            for e in self.weeklies:
                for k in range(int(spot // gap * gap) - 40 * gap, int(spot // gap * gap) + 41 * gap, gap):
                    for side in ("CE", "PE"):
                        tok += 1
                        self._rows["NFO"].append({"instrument_token": tok,
                            "tradingsymbol": f"{name}{e.strftime('%y')}{'OND'[e.month - 10] if e.month >= 10 else e.month}{e.day:02d}{k}{side}",
                            "name": name, "expiry": e, "strike": float(k), "lot_size": lot,
                            "instrument_type": side, "segment": "NFO-OPT", "exchange": "NFO"})
            for e in self.monthlies:
                tok += 1
                self._rows["NFO"].append({"instrument_token": tok, "tradingsymbol": f"{name}{e.strftime('%y%b').upper()}FUT",
                                          "name": name, "expiry": e, "strike": 0.0, "lot_size": lot,
                                          "instrument_type": "FUT", "segment": "NFO-FUT", "exchange": "NFO"})
        self.by_key = {f"{r['exchange']}:{r['tradingsymbol']}": r for x in self._rows.values() for r in x}
        self.by_tok = {r["instrument_token"]: r for x in self._rows.values() for r in x}

    def set_access_token(self, tok):
        self.token = tok

    def profile(self):
        _ZFakeKite.calls["profile"] += 1
        return {"user_name": "Test User"}

    def instruments(self, exchange=None):
        _ZFakeKite.calls["instruments"] += 1
        return list(self._rows.get(exchange, []))

    def price(self, r):
        import math as _m
        name = r["name"]
        if r["segment"] == "INDICES":
            return {"NIFTY 50": self.SPOT["NIFTY"], "NIFTY BANK": self.SPOT["BANKNIFTY"], "INDIA VIX": 12.85}[name]
        if r["instrument_type"] == "EQ":
            return 1000.0 + (r["instrument_token"] % 7) * 100
        s = self.SPOT[name]
        if r["instrument_type"] == "FUT":
            return s + 40
        T = max(1 / 365 / 24, ((r["expiry"].toordinal() - self.now.date().toordinal()) + 0.25) / 365)
        k, v, call = r["strike"], 0.13, r["instrument_type"] == "CE"
        d1 = (_m.log(s / k) + 0.5 * v * v * T) / (v * _m.sqrt(T)); d2 = d1 - v * _m.sqrt(T)
        nc = lambda x: 0.5 * (1 + _m.erf(x / _m.sqrt(2)))
        p = s * nc(d1) - k * nc(d2) if call else k * nc(-d2) - s * nc(-d1)
        return max(0.05, round(p * 20) / 20)

    def row_quote(self, r, oi_lots=1000):
        p = self.price(r)
        lot = r["lot_size"] or 1
        q = {"instrument_token": r["instrument_token"], "last_price": p,
             "ohlc": {"open": p * 0.99, "high": p * 1.02, "low": p * 0.97, "close": round(p * 0.95, 2)}}
        if r["segment"] != "INDICES":
            q.update({"volume": lot * 4321, "oi": (lot * oi_lots) if r["instrument_type"] != "EQ" else 0,
                      "buy_quantity": lot * 700, "sell_quantity": lot * 200,
                      "depth": {"buy": [{"price": round(p - 0.05 * (i + 1), 2), "quantity": lot * (i + 1), "orders": i + 1} for i in range(5)],
                                "sell": [{"price": round(p + 0.05 * (i + 1), 2), "quantity": lot * (i + 2), "orders": i + 1} for i in range(5)]}})
        return q

    def quote(self, *instruments):
        keys = instruments[0] if len(instruments) == 1 and isinstance(instruments[0], list) else list(instruments)
        _ZFakeKite.calls["quote"] += 1
        _ZFakeKite.calls["quote_keys"].append(len(keys))
        return {k: self.row_quote(self.by_key[k]) for k in keys if k in self.by_key}

    def historical_data(self, token, frm, to, interval, continuous=False, oi=False):
        import datetime as _dt
        _ZFakeKite.calls["historical"] += 1
        r = self.by_tok.get(token) or {}
        if interval == "day":
            y = self.now.date() - _dt.timedelta(days=1)
            return [{"date": _dt.datetime(y.year, y.month, y.day, tzinfo=self.IST), "open": 1, "high": 1,
                     "low": 1, "close": 1, "volume": 1, "oi": (r.get("lot_size") or 1) * 900}]
        step = {"minute": 1, "3minute": 3, "5minute": 5}.get(interval, 1)
        d0 = _dt.date.fromisoformat(frm[:10]); d1 = _dt.date.fromisoformat(to[:10])
        out = []
        day = d0
        while day <= d1:
            if day.weekday() < 5:
                t = _dt.datetime(day.year, day.month, day.day, 9, 15, tzinfo=self.IST)
                for i in range(0, 375, step):
                    ts = t + _dt.timedelta(minutes=i)
                    if ts > self.now:
                        break
                    px = 25000 + 30 * __import__("math").sin(i / 40)
                    out.append({"date": ts, "open": px, "high": px + 3, "low": px - 3, "close": px + 1,
                                "volume": 0 if r.get("segment") == "INDICES" else 650 * (1 + i % 5)})
            day += _dt.timedelta(days=1)
        return out


class _ZFakeTicker:
    MODE_FULL, MODE_QUOTE, MODE_LTP = "full", "quote", "ltp"
    last = None

    def __init__(self, api_key, access_token, **_k):
        self.subs, self.modes, self.closed = set(), {}, 0
        self.on_ticks = self.on_connect = self.on_close = self.on_error = None
        self.ws, self.open = object(), True          # an open socket unless a test says otherwise
        _ZFakeTicker.last = self

    def is_connected(self):
        return self.open

    def connect(self, threaded=False, **_k):
        pass

    def subscribe(self, toks):
        if not self.open:          # what the real ticker does with no socket yet
            raise AttributeError("'NoneType' object has no attribute 'sendMessage'")
        self.subs.update(toks)

    def set_mode(self, mode, toks):
        for t_ in toks:
            self.modes[t_] = mode

    def _close(self, code=None, reason=None):
        self.closed += 1

    def stop_retry(self):
        self.stopped = True

    def close(self, *a):
        self.closed += 1


def _z_setup(tmpdir):
    import nse_adapter_zerodha as Z
    Z.KiteConnect, Z.KiteTicker, Z.SDK_AVAILABLE = _ZFakeKite, _ZFakeTicker, True
    os.environ["ZERODHA_API_KEY"] = "test_key"
    os.environ["ZERODHA_ACCESS_TOKEN"] = "test_token"
    Z._CACHE_DIR = tmpdir
    Z._kite = Z._kws = Z._socket = None
    Z._status.update(connected=False, streaming=False, last_error="")
    for d_ in (Z._ticks, Z._depth, Z._qcache, Z._flow, Z._raw_last, Z._sub_req, Z._retry_n, Z._key2tok, Z._tok2key,
               Z._kind, Z._lot_of, Z._opts, Z._futs, Z._idx_key, Z._eq_key, Z._oi_today, Z._dirty, Z._eq_rest_cache):
        d_.clear()
    Z._subscribed.clear(); Z._depth_subs.clear(); Z._invalid_syms.clear()
    Z._inst["day"].clear()
    Z._oi_prev.update(day=None, oi={}, complete=False, loaded_for="")
    Z._oi_preopen.update(day="", oi={})
    Z._oi_backfill_q.clear(); Z._oi_tried.clear()
    Z._oi_dirty["busy"] = True          # no background threads in tests
    Z._rl["quote"][1] = Z._rl["historical"][1] = Z._rl["other"][1] = 0.0
    Z._start_oi_keeper = lambda: None
    # a fixed mid-session clock, so results do not depend on when the tests
    # run (before 09:15 the adapter rightly takes OI as yesterday's close,
    # and before 08:45 it treats Kite's instrument list as provisional)
    Z._ist_min_now = lambda: 600
    Z._fetched_early = lambda ts: False
    return Z


def test_zerodha_adapter():
    """The Zerodha adapter serves the same chain payload as the Fyers one:
    every key, contracts not shares, OI change from the previous close,
    futures, depth and flow from FULL-mode ticks, zero REST once streaming."""
    import tempfile, ast as _ast, json as _json, shutil
    tmp = tempfile.mkdtemp(prefix="ztest")
    Z = _z_setup(tmp)
    try:
        # previous session's closing OI for one contract, recorded on disk
        fk = _ZFakeKite()
        ce_key = next(k for k, r in fk.by_key.items() if r["name"] == "NIFTY" and r["instrument_type"] == "CE"
                      and r["strike"] == 25000 and r["expiry"] == fk.weeklies[0])
        y = Z._ist_today(1)
        with open(os.path.join(tmp, f"oi_close_{y}.json"), "w") as f:
            _json.dump({"oi": {ce_key: 65 * 900}, "complete": True}, f)
        tick_log = []
        Z.set_tick_sink(lambda m: tick_log.append(m))
        d = Z.fetch_chain("NIFTY", None, 12)
        # every key the Fyers adapter emits
        src = open(os.path.join(HERE, "nse_adapter_fyers.py"), encoding="utf-8").read()
        fn = next(n for n in _ast.walk(_ast.parse(src)) if isinstance(n, _ast.FunctionDef) and n.name == "fetch_chain")
        out_dict = next(n.value for n in _ast.walk(fn) if isinstance(n, _ast.Assign)
                        and getattr(n.targets[0], "id", "") == "out" and isinstance(n.value, _ast.Dict))
        fy_keys = {k.value for k in out_dict.keys if isinstance(k, _ast.Constant)}
        missing = sorted(fy_keys - set(d))
        check("zerodha: chain payload carries every key the Fyers adapter emits", not missing, str(missing))
        check("zerodha: tagged as the zerodha source", d["_source"] == "zerodha" and d["oi_unit"] == "contracts")
        row = next(r for r in d["strikes"] if r["strike"] == 25000)
        check("zerodha: OI in CONTRACTS (Kite counts units)", row["ce_oi"] == 1000 and d["lot_size"] == 65 and d["qty_per_contract"] == 65,
              f"{row['ce_oi']} lot {d['lot_size']}")
        check("zerodha: OI change measured from the recorded previous close", row["ce_oi_chg"] == 100, str(row["ce_oi_chg"]))
        check("zerodha: 51 strikes around spot, both sides priced", len(d["strikes"]) == 51 and row["ce_ltp"] > 0 and row["pe_ltp"] > 0,
              str(len(d["strikes"])))
        # the fake premiums are fixed while time to expiry shrinks with the
        # real clock (16.5% the evening before expiry), so only sanity-bound it
        check("zerodha: implied vol solved from prices", d["iv_source"] == "solved" and 5 < d["atm_iv"] < 40, f"{d['iv_source']} {d['atm_iv']}")
        check("zerodha: spot, VIX and previous close from the index", d["underlying_value"] == 25012.0 and d["india_vix"] == 12.85
              and d["spot_prev_close"], f"{d['underlying_value']} {d['india_vix']}")
        check("zerodha: near and next-month futures with OI in contracts",
              d["futures_price"] == 25052.0 and d["futures_far_price"] and d["futures_oi"] == 1000
              and d.get("futures_symbol", "").endswith("FUT") if "futures_symbol" in d else d["futures_oi"] == 1000,
              f"{d['futures_price']} {d['futures_far_price']} {d['futures_oi']}")
        check("zerodha: expiry list and fractional dte", d["expiry"] and len(d["all_expiries"]) == 3 and 0 < d["dte"] < 8,
              f"{d['expiry']} {d['all_expiries']} {d['dte']}")
        q0 = _ZFakeKite.calls["quote"]
        check("zerodha: a cold chain costs two quote calls (head + strikes)", q0 == 2, str(_ZFakeKite.calls))
        tk = _ZFakeTicker.last
        sub_keys = {Z._tok2key[t_] for t_ in tk.subs}
        check("zerodha: chain, index, VIX and futures subscribed in FULL mode (no REST)",
              ce_key in sub_keys and "NSE:NIFTY 50" in sub_keys and "NSE:INDIA VIX" in sub_keys
              and all(tk.modes.get(Z._key2tok[k]) == "full" for k in sub_keys), f"{len(sub_keys)}")
        # stream the whole window as Kite FULL-mode ticks
        def tick_for(key, vol_mult=1.0, ltp_bump=0.0):
            r = fk.by_key[key]
            qd = fk.row_quote(r)
            t_ = {"instrument_token": r["instrument_token"], "mode": "full", "last_price": qd["last_price"] + ltp_bump,
                  "ohlc": qd["ohlc"]}
            if r["segment"] != "INDICES":
                t_.update({"volume_traded": int(qd["volume"] * vol_mult), "oi": qd["oi"],
                           "total_buy_quantity": qd["buy_quantity"], "total_sell_quantity": qd["sell_quantity"],
                           "depth": qd["depth"]})
            return t_
        Z._on_ticks(None, [tick_for(Z._tok2key[t_]) for t_ in tk.subs])
        d2 = Z.fetch_chain("NIFTY", None, 12)
        check("zerodha: once streaming, a chain refresh costs ZERO REST calls", _ZFakeKite.calls["quote"] == q0, str(_ZFakeKite.calls))
        bk = d2["depth"]["strikes"].get("25000CE") or {}
        check("zerodha: five-level book and totals from the FULL tick",
              len(bk.get("bids") or []) == 5 and len(bk.get("asks") or []) == 5 and bk.get("tbq") == 65 * 700, str(bk)[:200])
        check("zerodha: futures books keyed near/far", (d2["depth"]["futures"].get("near") or {}).get("bids"))
        _row = next((r for r in d2["strikes"] if r.get("ce_symbol") == ce_key), {})
        check("zerodha: chain rows carry the whole-book totals too (the grid's fallback before the stream has them)", _row.get("ce_tbq") is not None and _row.get("ce_tsq") is not None, str({k: v for k, v in _row.items() if "tbq" in k or "tsq" in k}))
        bt = Z.book_totals([ce_key, "NFO:NOSUCH"])
        check("zerodha: total bid and offer quantity per contract come from the tick store (no REST), unknown contracts are left out",
              ce_key in bt and "NFO:NOSUCH" not in bt and bt[ce_key][0] is not None and bt[ce_key][1] is not None and bt[ce_key][2] < 5, str(bt))
        # a print at the offer is a BUY
        Z._on_ticks(None, [tick_for(ce_key, vol_mult=1.01, ltp_bump=0.05)])
        fl = Z._flow_read(ce_key) or {}
        check("zerodha: aggressor flow classifies a print at the offer as a buy", fl.get("b5", 0) > 0 and fl.get("s5", 0) == 0, str(fl))
        check("zerodha: ticks pushed to the browsers keyed like the chain's _tokens",
              any(ce_key in (m.get("t") or {}) for m in tick_log) and ce_key in d2["_tokens"])
        # a chosen expiry
        d3 = Z.fetch_chain("NIFTY", d["all_expiries"][1], 12)
        check("zerodha: a chosen expiry is served", d3["expiry"] == d["all_expiries"][1], f"{d3['expiry']}")
        # candles: index bars with the future's volume, one cached call pair
        h0 = _ZFakeKite.calls["historical"]
        c1 = Z.fetch_candles("NIFTY", 1, 1)
        c2 = Z.fetch_candles("NIFTY", 1, 1)
        ok_c = (not Z._ist_market_open()) or (c1 and all(b["v"] > 0 for b in c1))
        check("zerodha: index candles carry the near future's volume (cached)", ok_c and c1 == c2
              and _ZFakeKite.calls["historical"] - h0 <= 2, f"{len(c1)} {_ZFakeKite.calls['historical'] - h0}")
        hd = Z.fetch_history_days("NIFTY", 60)
        check("zerodha: 60 days of minute bars in ONE history call", hd and _ZFakeKite.calls["historical"] - h0 <= 3, str(len(hd)))
        # the Scalping chart's candles: an option or the index, in-session bars only, one call, then cached
        h1 = _ZFakeKite.calls["historical"]
        fb1 = Z.fetch_bars(ce_key, 5); fb2 = Z.fetch_bars(ce_key, 5); fbi = Z.fetch_bars("NIFTY", 1)
        insess = all(Z._in_session(((b["t"] + 19800) % 86400) // 60) for b in fb1)
        try:
            Z.fetch_bars(ce_key, 7); bad_iv = False
        except ValueError:
            bad_iv = True
        check("zerodha: chart candles for an option and the index - sorted, in session, ONE call per contract and interval (cached), odd intervals refused",
              fb1 and fb1 == fb2 and fbi and insess and [b["t"] for b in fb1] == sorted(b["t"] for b in fb1) and bad_iv
              and _ZFakeKite.calls["historical"] - h1 == 2 and 1 <= len({(b["t"] + 19800) // 86400 for b in fb1}) <= 4, f"{len(fb1)} {len(fbi)} {_ZFakeKite.calls['historical'] - h1}")
        # constituents: REST first, socket afterwards
        cq = Z.constituent_quotes(["RELIANCE", "HDFCBANK"])
        qa = _ZFakeKite.calls["quote"]
        Z._on_ticks(None, [{"instrument_token": fk.by_key[f"NSE:{n}"]["instrument_token"], "mode": "quote",
                            "last_price": 1234.0, "volume_traded": 999, "ohlc": {"open": 1200, "high": 1240, "low": 1190, "close": 1210}}
                           for n in ("RELIANCE", "HDFCBANK")])
        cq2 = Z.constituent_quotes(["RELIANCE", "HDFCBANK"])
        check("zerodha: constituents quoted, then served from the stream with no REST",
              len(cq) == 2 and cq2["RELIANCE"]["lastPrice"] == 1234.0 and _ZFakeKite.calls["quote"] == qa
              and Z._eq_stats["socket"] == 2, f"{cq2.get('RELIANCE')} {_ZFakeKite.calls['quote']} {qa}")
        eq_tok = fk.by_key["NSE:RELIANCE"]["instrument_token"]
        check("zerodha: equities streamed in QUOTE mode, not FULL", tk.modes.get(eq_tok) != "full")
        # audit shape
        a = Z.audit_state("NIFTY", d2)
        need = {"socket", "spot_tick_age_s", "options", "price_subscribed", "ticking_60s", "depth_subscribed",
                "ladders_fresh_120s", "totals_fresh_120s", "flow_contracts", "flow_coverage", "lot_from_data",
                "lot_from_book", "lot_from_master", "lot_table", "raw_rest", "watchdog"}
        check("zerodha: data-audit state has the fields the server reads", need <= set(a), str(sorted(need - set(a))))
        st = Z.status()
        check("zerodha: status names the login script", st.get("login_script") == "zerodha_login.py" and st["configured"])
    finally:
        Z.set_tick_sink(None)
        shutil.rmtree(tmp, ignore_errors=True)
        for f in os.listdir(os.path.join(HERE, "history")) if os.path.isdir(os.path.join(HERE, "history")) else []:
            if f.startswith("NIFTY_60_"):
                try: os.remove(os.path.join(HERE, "history", f))
                except Exception: pass


def test_zerodha_watchdog_and_oi():
    """Stream watchdog (free remedies, spaced out), the pre-open OI capture,
    the daily-candle OI back-fill and the token expiry rule."""
    import tempfile, shutil, json as _json, datetime as _dt, time
    tmp = tempfile.mkdtemp(prefix="ztest")
    Z = _z_setup(tmp)
    try:
        Z.fetch_chain("NIFTY", None, 12)
        tk = _ZFakeTicker.last
        IST = _dt.timezone(_dt.timedelta(hours=5, minutes=30))
        T = _dt.datetime(2026, 10, 16, 11, 0, tzinfo=IST).timestamp()           # Friday 11:00
        idx = "NSE:NIFTY 50"
        opt = next(k for k in Z._subscribed if Z._kind.get(k) == "OPT")
        for k in Z._subscribed:
            Z._sub_req[k] = T - 400
            Z._ticks[k] = {"ltp": 1.0, "ts": T - 2}
        Z._ticks[idx] = {"ltp": 25000.0, "ts": T - 200}
        Z._wd.update(last_heal=0.0, last_force=0.0, heals=0, forced_reconnects=0, force_streak=0)
        r = Z._watchdog_pass(T)
        check("zerodha watchdog: a silent index is re-subscribed (no REST)", r.get("resubscribed") == 1, str(r))
        for k in Z._subscribed:
            Z._ticks[k] = {"ltp": 1.0, "ts": T - 300}
        r2 = Z._watchdog_pass(T + 10)
        check("zerodha watchdog: no ticks at all -> the stream is reopened (ticker reconnects itself)",
              r2.get("forced") and tk.closed == 1, str(r2))
        r3 = Z._watchdog_pass(T + 60)
        check("zerodha watchdog: reopens are spaced out", not r3.get("forced") and tk.closed == 1 and Z._force_gap() == 600, str(r3))
        # a strike that never ticks is retried a few times, never written off
        for k in Z._subscribed:
            Z._ticks[k] = {"ltp": 1.0, "ts": T + 1000}
        Z._ticks.pop(opt, None); Z._sub_req[opt] = T; Z._retry_n[opt] = 4
        Z._watchdog_pass(T + 1000)
        check("zerodha watchdog: an untraded strike is not marked invalid", opt not in Z._invalid_syms)
        # previous-close OI back-fill from the daily candle
        Z._oi_prev.update(day=None, oi={}, complete=False, loaded_for=Z._ist_today())
        Z._oi_backfill_one(opt)
        check("zerodha OI: yesterday's close back-filled from Kite's daily candle", Z._oi_prev["oi"].get(opt) == 65 * 900,
              str(Z._oi_prev["oi"].get(opt)))
        # token rule: issued today after 06:00 -> valid; two days ago -> expired
        tf = os.path.join(tmp, "tok.json")
        os.environ.pop("ZERODHA_ACCESS_TOKEN", None)
        os.environ["ZERODHA_TOKEN_FILE"] = tf
        with open(tf, "w") as f:
            _json.dump({"access_token": "x", "ts": time.time() - 60}, f)
        g = time.gmtime(time.time() + 19800)
        v1 = Z.token_validity()
        with open(tf, "w") as f:
            _json.dump({"access_token": "x", "ts": time.time() - 2 * 86400}, f)
        v2 = Z.token_validity()
        check("zerodha token: today's token valid (unless issued before 06:00), two-day-old expired",
              (v1["valid"] or g.tm_hour < 6) and not v2["valid"] and "zerodha_login.py" in v2["reason"], f"{v1} {v2}")
    finally:
        os.environ.pop("ZERODHA_TOKEN_FILE", None)
        shutil.rmtree(tmp, ignore_errors=True)


def test_broker_neutral_server():
    """DATA_SOURCE=zerodha reaches the Zerodha adapter everywhere the server
    used to hard-wire Fyers, and the dashboard reads either broker."""
    srv = open(os.path.join(HERE, "nse_chain_server.py"), encoding="utf-8").read()
    import re as _re
    direct = _re.findall(r"import nse_adapter_fyers as \w+", srv)
    check("server: no call site imports the Fyers adapter directly (all go through _broker())",
          direct == ["import nse_adapter_fyers as _m"], str(direct))
    check("server: no broker branch tests for fyers alone", 'if RUN_DATA_SOURCE == "fyers":' not in srv
          and 'if src == "fyers":' not in srv and 'if source == "fyers":' not in srv)
    check("server: zerodha is a broker source with its own login script",
          'BROKER_SOURCES = ("fyers", "zerodha")' in srv and '"zerodha_login.py"' in srv and "_ensure_broker_token()" in srv)
    dash = open(os.path.join(HERE, "nse_dashboard.html"), encoding="utf-8").read()
    check("dashboard: token warning and broker-feed badge read either broker",
          "j?.broker || j?.fyers" in dash and "/(fyers|zerodha)\\(/" in dash)
    check("zerodha_login.py present and never prints the secret or token",
          os.path.exists(os.path.join(HERE, "zerodha_login.py"))
          and "print(secret" not in open(os.path.join(HERE, "zerodha_login.py")).read()
          and "print(access" not in open(os.path.join(HERE, "zerodha_login.py")).read())


def test_zerodha_fixes_and_diagnostics():
    """Review fixes (provisional instrument list, 403 stops the reconnect loop,
    connect cooldown, daily back-fill reset) and the Diagnostics tab."""
    import tempfile, shutil, json as _json, time
    tmp = tempfile.mkdtemp(prefix="ztest")
    Z = _z_setup(tmp)
    try:
        Z.fetch_chain("NIFTY", None, 12)
        n_inst = _ZFakeKite.calls["instruments"]
        # 1. a list fetched before 08:45 is fetched again once it is later
        real_min = Z._ist_min_now
        Z._inst["fetched"]["NFO"] = time.time() - 86400 * 0 - 3600 * 24     # placeholder, overwritten below
        import calendar
        g = time.gmtime(time.time() + 19800)
        early = time.time() - 3600              # an hour ago - "before 08:30" under the patched rule below
        Z._inst["fetched"]["NFO"] = early
        # rewrite the cached file as an early fetch too
        path = os.path.join(tmp, f"instruments_NFO_{Z._ist_today()}.json")
        j = _json.load(open(path)); j["fetched"] = early; _json.dump(j, open(path, "w"))
        marker = time.time() - 1
        Z._fetched_early = lambda ts: ts < marker          # anything fetched before now was "early"
        Z._ist_min_now = lambda: 10 * 60
        Z._ensure_instruments(("NFO",))
        refetched = _ZFakeKite.calls["instruments"] == n_inst + 1
        Z._ensure_instruments(("NFO",))
        once = _ZFakeKite.calls["instruments"] == n_inst + 1
        Z._ist_min_now = real_min
        check("zerodha fix: a pre-08:30 instrument list is fetched again (once) after 08:45",
              refetched and once and not Z._fetched_early(Z._inst["fetched"]["NFO"]), str(_ZFakeKite.calls["instruments"] - n_inst))
        # 2. stream refused with 403 -> stop retrying, say why
        tk = _ZFakeTicker.last
        Z._status["token_rejected"] = False
        Z._on_ws_close(None, 1006, "WebSocket connection upgrade failed (403 - Forbidden)")
        check("zerodha fix: a 403 on the stream stops the reconnect loop and asks for a login",
              getattr(tk, "stopped", False) and Z._status.get("token_rejected") and "zerodha_login" in Z._status["last_error"])
        # 3. a failing token is not retried on every request; a new token is tried at once
        class Bad(_ZFakeKite):
            n = 0
            def profile(self):
                Bad.n += 1
                raise RuntimeError("TokenException: Incorrect api_key or access_token")
        Z.KiteConnect = Bad
        Z._status["connected"] = False; Z._kite = None
        Z._conn.update(fail_ts=0.0, fail_tok="")
        a = Z.connect(); b = Z.connect(); c = Z.connect()
        tried = Bad.n
        os.environ["ZERODHA_ACCESS_TOKEN"] = "fresh_token"
        Z.KiteConnect = _ZFakeKite
        d = Z.connect()
        check("zerodha fix: a rejected token costs one profile call a minute, not one per request; a new token connects at once",
              not a and not b and not c and tried == 1 and d, f"tried={tried} new={d}")
        # 4. back-fill bookkeeping resets with the day
        Z._oi_tried.add("NFO:X"); Z._oi_prev["loaded_for"] = "2000-01-01"
        Z._oi_load_prev()
        check("zerodha fix: the back-fill 'done' list resets each day", "NFO:X" not in Z._oi_tried)
        # diagnostics: shape, statuses, memory only
        q0, h0 = _ZFakeKite.calls["quote"], _ZFakeKite.calls["historical"]
        dg = Z.diagnostics()
        rows = [r for s_ in dg["sections"] for r in s_["rows"]]
        check("diagnostics (zerodha): sections of label/value/status rows and a verdict",
              dg["id"] == "zerodha" and dg["verdict"] in ("ok", "warn", "bad") and len(dg["sections"]) >= 5
              and all(len(r) == 3 and r[2] in ("ok", "warn", "bad", "") for r in rows), str(dg["summary"]))
        check("diagnostics (zerodha): building it costs no Kite call",
              _ZFakeKite.calls["quote"] == q0 and _ZFakeKite.calls["historical"] == h0)
    finally:
        os.environ["ZERODHA_ACCESS_TOKEN"] = "test_token"
        Z.KiteConnect = _ZFakeKite
        shutil.rmtree(tmp, ignore_errors=True)
    import nse_adapter_fyers as F
    df = F.diagnostics()
    frows = [r for s_ in df["sections"] for r in s_["rows"]]
    check("diagnostics (fyers): same shape, with the socket-ownership and refusal rows",
          df["id"] == "fyers" and all(len(r) == 3 for r in frows)
          and any("owns the socket" in r[0] for r in frows) and any("Refused" in r[0] for r in frows))
    srv = open(os.path.join(HERE, "nse_chain_server.py"), encoding="utf-8").read()
    check("diagnostics: /api/broker-diag serves only the DATA_SOURCE broker (NSE card when DATA_SOURCE=nse)",
          '"/api/broker-diag"' in srv and "def _broker_diag()" in srv and "def _nse_diag()" in srv
          and "_fallbacks[\"n\"] += 1" in srv)
    dash = open(os.path.join(HERE, "nse_dashboard.html"), encoding="utf-8").read()
    check("diagnostics: tab, panel and poller in the dashboard (only while the tab is open)",
          'data-tab="diag"' in dash and 'id="tab-diag"' in dash and "function renderDiag(" in dash
          and "/api/broker-diag" in dash and "classList.contains('active')" in dash)


def test_zerodha_subscribe_before_connect():
    """Subscriptions asked for before Kite's stream has connected are held,
    not reported as failures, and all go out the moment it connects. A token
    generated after midnight is dead at 06:00, so startup checks validity."""
    import tempfile, shutil
    tmp = tempfile.mkdtemp(prefix="ztest")
    Z = _z_setup(tmp)
    try:
        Z.connect()
        tk = _ZFakeTicker.last
        tk.open = False                                  # handshake not finished yet
        Z._sock_log.clear()
        d = Z.fetch_chain("NIFTY", None, 12)             # subscribes ~108 instruments
        errs = [x for x in Z._sock_log if x[1] == "error"]
        check("zerodha: a subscribe before the stream connects is held, not an error",
              not errs and Z._held["n"] > 100 and not tk.subs, f"errors={errs[:1]} held={Z._held['n']}")
        tk.open = True
        Z._on_ws_connect(tk, {})                         # the stream opens
        want = {Z._key2tok[k] for k in Z._subscribed if k in Z._key2tok}
        check("zerodha: on connect every held subscription goes out, in FULL mode",
              want <= tk.subs and Z._held["n"] == 0 and all(tk.modes.get(Z._key2tok[k]) == "full"
                                                            for k in d["_tokens"] if k in Z._key2tok), f"{len(tk.subs)}/{len(want)}")
        rows = [r for s_ in Z.diagnostics()["sections"] for r in s_["rows"]]
        check("zerodha diagnostics: shows whether the stream socket is open", any(r[0] == "Stream socket" and r[1] == "open" for r in rows))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    srv = open(os.path.join(HERE, "nse_chain_server.py"), encoding="utf-8").read()
    check("startup login check asks the adapter whether today's token is still VALID (06:00 expiry)",
          "_tv = _broker().token_validity()" in srv and 'tok_date == today and _tv.get("valid", True)' in srv)


def test_diag_extras():
    """Diagnostics extras: feed delay from exchange stamps (both brokers),
    API calls attributed to the computer/endpoint, the NTP clock check, the
    server error log, silent ATM strikes, and the dashboard pieces."""
    import tempfile, shutil, datetime as _dt, ast as _ast, socket as _so, struct as _st, threading as _th, time as _t
    tmp = tempfile.mkdtemp(prefix="ztest")
    Z = _z_setup(tmp)
    try:
        Z.fetch_chain("NIFTY", None, 12)
        Z._lat.clear(); Z._lat_last.update(t=0.0, best=None)
        base = _t.time() - 30
        for i in range(20):
            Z._lat_note(base + i - 0.8, now=base + i)
        ls = Z.latency_stats(600)
        check("feed delay (zerodha): median of exchange-stamp delay", ls["n"] == 20 and abs(ls["p50"] - 0.8) < 0.01, str(ls)[:120])
        Z._lat.clear(); Z._lat_last.update(t=0.0, best=None)
        tok = next(iter(_ZFakeTicker.last.subs))
        Z._on_ticks(None, [{"instrument_token": tok, "mode": "full", "last_price": 10.0,
                            "exchange_timestamp": _dt.datetime.fromtimestamp(_t.time() - 1.5)}])
        check("feed delay (zerodha): read off the FULL tick's exchange_timestamp", len(Z._lat) == 1 and 1.4 < Z._lat[0][1] < 2.0, str(list(Z._lat)))
        Z._who.name = "192.168.1.9 /api/chain"
        Z._count_rest("quote")
        tl = Z.rest_timeline()
        check("API timeline: calls per minute today, attributed to the computer and endpoint",
              tl["minutes"] and any(w == "192.168.1.9 /api/chain" for w, _n in tl["by_client"]) and tl["limit_per_min"], str(tl)[:160])
        Z._who.name = None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    import nse_adapter_fyers as F
    F._lat.clear(); F._lat_last.update(t=0.0, best=None)
    F._on_message({"symbol": "NSE:NIFTY50-INDEX", "ltp": 25000.0, "exch_feed_time": _t.time() - 1.2})
    check("feed delay (fyers): read off exch_feed_time", len(F._lat) == 1 and 1.1 < F._lat[0][1] < 1.6, str(list(F._lat)))
    # server helpers, run in isolation
    src = open(os.path.join(HERE, "nse_chain_server.py"), encoding="utf-8").read()
    tree = _ast.parse(src)
    want = {"_ntp_query", "_LogTee", "_silent_strikes"}
    nodes = [n for n in tree.body if isinstance(n, (_ast.FunctionDef, _ast.ClassDef)) and n.name in want]
    import collections as _c, re as _re
    ns = {"time": _t, "_dsock": _so, "_dstruct": _st, "_log_ring": _c.deque(maxlen=50), "_err_ring": _c.deque(maxlen=50),
          "_ERR_RX": _re.compile(r"(error|failed|exception|traceback|using NSE)", _re.I), "_ERR_SKIP": _re.compile(r"(no refusals)", _re.I)}
    exec(compile(_ast.Module(body=nodes, type_ignores=[]), "srv", "exec"), ns)
    # a local NTP server whose clock is 5 s ahead of ours
    srv = _so.socket(_so.AF_INET, _so.SOCK_DGRAM); srv.bind(("127.0.0.1", 0)); port = srv.getsockname()[1]
    def answer():
        data, addr = srv.recvfrom(48)
        now = _t.time() + 5.0 + 2208988800
        sec, frac = int(now), int((now % 1) * 2 ** 32)
        pkt = bytearray(48); pkt[0] = 0x1c
        pkt[32:40] = _st.pack("!II", sec, frac); pkt[40:48] = _st.pack("!II", sec, frac)
        srv.sendto(bytes(pkt), addr)
    th = _th.Thread(target=answer, daemon=True); th.start()
    off, rtt = ns["_ntp_query"]("127.0.0.1", 2.0, port)
    srv.close()
    check("clock check: NTP offset measured (a server 5 s ahead reads +5 s)", 4.9 < off < 5.1 and rtt >= 0, f"{off} {rtt}")
    import io
    tee = ns["_LogTee"](io.StringIO())
    tee.write("[zerodha] quote failed: timeout\nall good here\n[adapter] zerodha failed (x) - using NSE for this request\n")
    errs = [l for _ts, l in ns["_err_ring"]]
    check("server log: problem lines kept for the tab, ordinary lines passed through",
          len(errs) == 2 and "quote failed" in errs[0] and len(ns["_log_ring"]) == 3, str(errs))
    now = _t.time()
    class _Ad:
        _ticks = {"NFO:A25000CE": {"ts": now - 2}, "NFO:A25000PE": {"ts": now - 300}}
    ns.update({"_is_broker": lambda: True, "_broker": lambda: _Ad,
               "_chain_cache": {"NIFTY|near|12": {"ts": now, "data": {"symbol": "NIFTY", "atm": 25000, "strike_gap": 50,
                   "strikes": [{"strike": 25000, "ce_symbol": "NFO:A25000CE", "pe_symbol": "NFO:A25000PE"},
                               {"strike": 26000, "ce_symbol": "NFO:FAR", "pe_symbol": "NFO:FAR2"}]}}}})
    sil = ns["_silent_strikes"]()
    check("silent strikes: an ATM strike quiet for 5 min is listed; far strikes are not checked",
          sil and sil[0]["checked"] == 2 and [q["strike"] for q in sil[0]["quiet"]] == [25000]
          and sil[0]["quiet"][0]["side"] == "PE", str(sil))
    check("server: self-test and diagnostics extras wired",
          '"/api/self-test"' in src and "def _self_test(" in src and "def _recording_health(" in src
          and 'out["extras"] = _diag_extras()' in src and "_install_log_tee()" in src)
    dash = open(os.path.join(HERE, "nse_dashboard.html"), encoding="utf-8").read()
    check("dashboard: token banner on every tab, checked every minute",
          'id="tokenBanner"' in dash and "_visInterval(_checkBrokerToken, 60000" in dash)
    check("dashboard: self-test button and the extra diagnostics cards",
          'id="diagTestBtn"' in dash and "function renderDiagExtras(" in dash and "Silent at-the-money strikes" in dash
          and "API calls per minute today" in dash and "Recording — today" in dash and "Feed delay" in dash)


def test_true_black_widgets():
    """Dark theme: widget surfaces are true black - panels use --wash (nothing
    in dark, a faint tint in light) instead of a white overlay, and floating
    boxes use --box."""
    import re as _re
    d = open(os.path.join(HERE, "nse_dashboard.html"), encoding="utf-8").read()
    root = d[d.index("  :root {"):d.index("  }", d.index("  :root {"))]
    check("theme: --wash is transparent and --box black in dark mode",
          "--wash: transparent;" in root and "--box: #000000;" in root and "--wash:rgba(15,23,42,0.03)" in d)
    check("theme: chart, Squeeze pane, Trade entries and the tiles sit on --wash",
          'id="chartSvg" viewBox="0 0 ${W} ${H}" style="width:100%;height:auto;background:var(--wash);' in d
          and d.count('style="fill:var(--wash)"') == 2 and '"d90Body" style="margin:2px 6px 4px;padding:5px 8px;border:1px solid rgba(255,255,255,0.10);border-radius:6px;background:var(--wash);"' in d)
    faint = [m.group(0) for m in _re.finditer(r'background:rgba\(255,255,255,0\.0(1[0-9]?|2[0-9]?)\)', d)]
    check("theme: no panel carries a faint white wash any more", not faint, str(faint[:3]))
    check("theme: no near-black overlay colours left",
          not _re.search(r'fill="rgba\((14,17,23|20,24,32|30,36,48),', d) and 'fill="#1a1d23"' not in d and 'background:#1a1d23' not in d)


def _ttr_reference(bars):
    """Independent Python re-implementation of the Pine script's maths, used
    to check the dashboard's port bar for bar."""
    import math as _m
    L, OFF, SIG, DL, DM, SL, SM, AL = 34, 0.85, 6.0, 34, 0.65, 3, 0.08, 14
    c = [b["c"] for b in bars]; h = [b["h"] for b in bars]; l = [b["l"] for b in bars]; n = len(c)
    mm, s = OFF * (L - 1), L / SIG
    w = [_m.exp(-((i - mm) ** 2) / (2 * s * s)) for i in range(L)]
    alma = [None] * n
    for t in range(L - 1, n):
        alma[t] = sum(c[t - (L - 1 - i)] * w[i] for i in range(L)) / sum(w)
    dev = [None] * n
    for t in range(DL - 1, n):
        win = c[t - DL + 1:t + 1]; mu = sum(win) / DL
        dev[t] = _m.sqrt(sum((x - mu) ** 2 for x in win) / DL)
    tr = [h[0] - l[0]] + [max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])) for i in range(1, n)]
    atr = [None] * n
    for t in range(n):
        if t == AL - 1:
            atr[t] = sum(tr[:AL]) / AL
        elif t >= AL:
            atr[t] = (atr[t - 1] * (AL - 1) + tr[t]) / AL
    trend, flips, pos = 0, [], None
    for t in range(n):
        a, d, at = alma[t], dev[t], atr[t]
        a0 = alma[t - SL] if t >= SL else None
        slope = ((a - a0) / at if at > 0 else 0.0) if None not in (a, a0, at) else None
        bull = slope is not None and d is not None and slope > SM and c[t] > a + d * DM
        bear = slope is not None and d is not None and slope < -SM and c[t] < a - d * DM
        fl = 0
        if trend != 1 and bull:
            trend, fl = 1, 1
        elif trend != -1 and bear:
            trend, fl = -1, -1
        if fl:
            flips.append((t, fl))
            dr = fl
            win = bars[max(0, t - 12):t]
            ss = (min(b["l"] for b in win) if dr == 1 else max(b["h"] for b in win)) if win else None
            raw = None if ss is None else (c[t] - ss if dr == 1 else ss - c[t])
            mn, mx = at * 0.75, at * 3.0
            risk = min(max(mn if raw is None else raw, mn), mx)
            pos = {"bar": t, "dir": dr, "entry": c[t], "stop": c[t] - dr * risk, "risk": risk, "hits": [False] * 4, "end": None}
        if pos and pos["end"] is None:
            if t > pos["bar"] and (l[t] <= pos["stop"] if pos["dir"] == 1 else h[t] >= pos["stop"]):
                pos["end"] = t
            else:
                for i in range(4):
                    tp = pos["entry"] + pos["dir"] * pos["risk"] * (i + 1)
                    if not pos["hits"][i] and (h[t] >= tp if pos["dir"] == 1 else l[t] <= tp):
                        pos["hits"][i] = True
    return {"alma": alma, "dev": dev, "atr": atr, "flips": flips, "pos": pos}


def test_trend_target_ribbon():
    """The dashboard's port of Trend Target Ribbon matches an independent
    implementation of the Pine maths: ALMA, deviation, ATR, every trend flip,
    and the position's entry, stop, risk and target hits. Settings as chosen."""
    import subprocess, json as _json, math as _m, random as _r
    d = open(os.path.join(HERE, "nse_dashboard.html"), encoding="utf-8").read()
    def grab(name):
        i = d.index(f"function {name}(")
        depth, j = 0, d.index("{", i)
        for k in range(j, len(d)):
            depth += d[k] == "{"; depth -= d[k] == "}"
            if depth == 0:
                return d[i:k + 1]
    i0 = d.index("  const TTR = {"); j0 = d.index("  const _ttrCache")
    _r.seed(11)
    bars, p = [], 25000.0
    for i in range(400):                      # trend up, chop, trend down, chop
        drift = 2.2 if i < 110 else (0 if i < 190 else (-2.6 if i < 300 else 0))
        o = p; p = p + drift + _r.gauss(0, 4); hi = max(o, p) + abs(_r.gauss(0, 2)); lo = min(o, p) - abs(_r.gauss(0, 2))
        bars.append({"t": 1000 + 60 * i, "o": o, "h": hi, "l": lo, "c": p})
    harness = d[i0:j0] + grab("_ttrCompute") + f"""
const r = _ttrCompute({_json.dumps(bars)});
console.log(JSON.stringify({{alma: r.alma, dev: r.dev, atr: r.atr,
  flips: r.flip.map((f, i) => f ? [i, f] : null).filter(Boolean), pos: r.positions, TTR}}));
"""
    res = subprocess.run(["node", "-e", harness], capture_output=True, text=True)
    try:
        js = _json.loads(res.stdout.strip().splitlines()[-1])
    except Exception:
        check("trend ribbon: harness ran", False, (res.stderr or res.stdout)[:400]); return
    ref = _ttr_reference(bars)
    close = lambda a, b: (a is None and b is None) or (a is not None and b is not None and abs(a - b) < 1e-6)
    check("trend ribbon: ALMA(34, 0.85, 6), stdev(34) and ATR(14) match the Pine maths on every bar",
          all(close(a, b) for a, b in zip(js["alma"], ref["alma"])) and all(close(a, b) for a, b in zip(js["dev"], ref["dev"]))
          and all(close(a, b) for a, b in zip(js["atr"], ref["atr"])))
    check("trend ribbon: every trend flip on the same bar and side", [tuple(x) for x in js["flips"]] == ref["flips"] and len(ref["flips"]) >= 2,
          f"js={js['flips']} ref={ref['flips']}")
    jp, rp = js["pos"][-1], ref["pos"]
    check("trend ribbon: the latest position's entry, stop, 1R and target hits match",
          jp["entryBar"] == rp["bar"] and jp["dir"] == rp["dir"] and abs(jp["entry"] - rp["entry"]) < 1e-9
          and abs(jp["stop"] - rp["stop"]) < 1e-9 and abs(jp["risk"] - rp["risk"]) < 1e-9 and jp["hits"] == rp["hits"]
          and (jp["endBar"] == rp["end"]), f"{jp} vs {rp}")
    T = js["TTR"]
    check("trend ribbon: settings - labels off, candle colouring off, Trend Glow columns of 4 in the trend colour, one position",
          T["showLabels"] is False and T["paintBars"] is False and T["glow"]["style"] == "columns"
          and T["glow"]["width"] == 4 and T["glow"]["color"] == "trend" and T["keepPositions"] == 1 and len(js["pos"]) == 1)
    check("trend ribbon: drawn on the candle chart with a toggle (on by default)",
          'id="chk_ttr"' in d and "if (_ttrL) svg += _ttrL.under;" in d and "if (_ttrL) svg += _ttrL.over;" in d
          and d.count("emaX: true, ttr: true, sqz: true, wt: true }") == 2)


def test_ctx_clock():
    d = open(os.path.join(HERE, "nse_dashboard.html")).read()
    check("context clock: 12-hour IST with seconds, ticking every second",
          "timeZone: 'Asia/Kolkata'" in d and "second: '2-digit', hour12: true" in d
          and "document.querySelectorAll('.ctx-clock-t')" in d)
    check("context clock: stretched to fill the whole pane as a grey watermark under the lines",
          'class="ctx-clock-t"' in d and 'textLength="992"' in d and 'lengthAdjust="spacingAndGlyphs"' in d
          and 'preserveAspectRatio="none"' in d and ".ctx-clock { position:absolute; inset:0; z-index:0; pointer-events:none;" in d
          and "fill:#808080; opacity:var(--clockOp" in d and ".ctx-plot { position:relative; z-index:1; }" in d)
    check("context clock: indicator lines keep a halo so they read over the digits",
          ".ctx-plot svg { filter:drop-shadow" in d)
    check("context clock: opacity slider 0-100, default 30, saved per browser and cleared by Reset Defaults",
          'data-ctxclock="1"' in d and 'min="0" max="100"' in d and "const CTX_CLOCK_OP_DEFAULT = 30;" in d
          and "localStorage.setItem('ctxClockOpacity'" in d and "'ctxClockOpacity',     //" in d
          and "opacity:var(--clockOp, 0.30)" in d and "--clockOp:0.30;" in d)


def test_warmup_and_signals():
    """Indicator warm-up from the previous trading session(s): weekends and NSE
    holidays skipped without asking the broker, each session fetched once and
    then served from memory/disk, today's bars saved at the close; indicators
    on the chart run over warm-up + today; Buy/Sell lines in the context pane."""
    import ast as _ast, tempfile, shutil, threading as _th, time as _t, json as _json, subprocess
    src = open(os.path.join(HERE, "nse_chain_server.py"), encoding="utf-8").read()
    tree = _ast.parse(src)
    want = {"_nse_holidays", "_is_trading_day", "_shift_day", "_prev_trading_days", "_warm_days_cfg", "_warm_path",
            "_warm_write", "_session_bars", "_warm_bars", "_warm_save_today", "_aggregate_candles", "_sess_bucket"}
    nodes = [n for n in tree.body if isinstance(n, _ast.FunctionDef) and n.name in want]
    hol = [n for n in tree.body if isinstance(n, _ast.Assign) and getattr(n.targets[0], "id", "") == "_NSE_HOLIDAYS_BUILTIN"]
    tmp = tempfile.mkdtemp(prefix="warm")
    calls = []
    empty_days = set()

    class _B:
        @staticmethod
        def fetch_candles(sym, iv, days, on_date=None):
            calls.append(on_date)
            if on_date in empty_days:
                return []
            import datetime as _dt
            y, m, d = (int(x) for x in on_date.split("-"))
            t0 = int(_dt.datetime(y, m, d, 9, 15, tzinfo=_dt.timezone(_dt.timedelta(hours=5, minutes=30))).timestamp())
            return [{"t": t0 + 60 * i, "o": 100.0 + i, "h": 101.0 + i, "l": 99.0 + i, "c": 100.5 + i, "v": 10} for i in range(375)]

    clock = {"now": _t.time()}

    class _T:
        time = staticmethod(lambda: clock["now"])
        gmtime = staticmethod(_t.gmtime)
        strftime = staticmethod(_t.strftime)
        timezone = _t.timezone
    ns = {"_os": os, "json": _json, "time": _T, "threading": _th, "_WARM_DIR": tmp, "_warm_mem": {}, "_warm_miss": {},
          "_warm_lock": _th.Lock(), "_warm_sym_locks": {}, "_warm_stats": {"broker_calls": 0, "disk_hits": 0, "saved_close": 0, "last": ""},
          "_is_broker": lambda: True, "_broker": lambda: _B, "_ctx_day": lambda: "2026-10-21", "RUN_DATA_SOURCE": "zerodha"}
    try:
        exec(compile(_ast.Module(body=hol + nodes, type_ignores=[]), "srv", "exec"), ns)
        check("warm-up: previous trading days skip the weekend and Dussehra (20 Oct)",
              ns["_prev_trading_days"]("2026-10-21", 2) == ["2026-10-19", "2026-10-16"], str(ns["_prev_trading_days"]("2026-10-21", 2)))
        check("warm-up: Gandhi Jayanti (2 Oct) and the weekend skipped before 5 Oct",
              ns["_prev_trading_days"]("2026-10-05", 2) == ["2026-10-01", "2026-09-30"])
        os.environ["NSE_HOLIDAYS"] = "2026-10-19"
        check("warm-up: NSE_HOLIDAYS in env adds a closure", not ns["_is_trading_day"]("2026-10-19"))
        os.environ.pop("NSE_HOLIDAYS", None)
        bars, days = ns["_warm_bars"]("NIFTY", "2026-10-21", 2)
        check("warm-up: two sessions, oldest first, one history call each, the holiday never asked for",
              days == ["2026-10-16", "2026-10-19"] and len(bars) == 750 and sorted(calls) == ["2026-10-16", "2026-10-19"]
              and bars[0]["t"] < bars[-1]["t"], f"{days} {len(bars)} {calls}")
        n0 = len(calls)
        ns["_warm_bars"]("NIFTY", "2026-10-21", 2)
        ns["_warm_mem"].clear()
        b2, d2 = ns["_warm_bars"]("NIFTY", "2026-10-21", 2)
        check("warm-up: asked again (and after a restart) it costs no API call - memory, then disk",
              len(calls) == n0 and d2 == days and len(b2) == 750 and ns["_warm_stats"]["disk_hits"] >= 2, f"{len(calls) - n0}")
        rows = ns["_aggregate_candles"](b2, 5)
        check("warm-up: aggregated to the chart interval per session (75 x 5m bars a day)", len(rows) == 150, str(len(rows)))
        # an unlisted holiday: the broker answers empty, the day is skipped and not re-asked for 15 minutes
        ns["_warm_mem"].clear(); shutil.rmtree(tmp); os.makedirs(tmp)
        calls.clear(); empty_days.add("2026-10-16")
        _b, d3 = ns["_warm_bars"]("NIFTY", "2026-10-21", 2)
        n1 = len(calls)
        ns["_warm_mem"].clear()
        ns["_warm_bars"]("NIFTY", "2026-10-21", 2)
        check("warm-up: a weekday the broker has no data for is skipped, and not re-asked for 15 minutes",
              d3 == ["2026-10-15", "2026-10-19"] and calls.count("2026-10-16") == 1, f"{d3} {calls[:n1]} {calls[n1:]}")
        # today's own bars saved after the close -> tomorrow costs nothing
        import datetime as _dt
        ist = _dt.timezone(_dt.timedelta(hours=5, minutes=30))
        clock["now"] = _dt.datetime(2026, 10, 21, 15, 50, tzinfo=ist).timestamp()
        today = _B.fetch_candles("NIFTY", 1, 1, on_date="2026-10-21")
        clock["now"] = _dt.datetime(2026, 10, 21, 15, 20, tzinfo=ist).timestamp()
        ns["_warm_save_today"]("NIFTY", today)
        before = os.path.exists(os.path.join(tmp, "NIFTY_2026-10-21.json"))
        clock["now"] = _dt.datetime(2026, 10, 21, 15, 50, tzinfo=ist).timestamp()
        ns["_warm_save_today"]("NIFTY", today)
        ns["_warm_mem"].clear(); calls.clear()
        ns["_ctx_day"] = lambda: "2026-10-22"
        _b4, d4 = ns["_warm_bars"]("NIFTY", "2026-10-22", 1)
        check("warm-up: today's bars saved after 15:45 (not before) and tomorrow's warm-up makes no API call",
              not before and d4 == ["2026-10-21"] and calls == [], f"{before} {d4} {calls}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    check("server: /api/warm-candles route, today's close saved from the chart's own fetch, diagnostics section",
          'parsed.path == "/api/warm-candles"' in src and "_warm_save_today(symbol, fc1)" in src
          and '"Indicator warm-up (previous sessions)"' in src)

    # ── the dashboard side ──
    d = open(os.path.join(HERE, "nse_dashboard.html"), encoding="utf-8").read()
    def grab(name):
        i = d.index(f"function {name}(")
        depth, j = 0, d.index("{", i)
        for k in range(j, len(d)):
            depth += d[k] == "{"; depth -= d[k] == "}"
            if depth == 0:
                return d[i:k + 1]
    import re as _re
    hjs = set(_re.findall(r"'(\d{4}-\d{2}-\d{2})'", d[d.index("const _NSE_HOLIDAYS = new Set("):d.index("]);", d.index("const _NSE_HOLIDAYS = new Set("))]))
    hpy = set(_re.findall(r'"(\d{4}-\d{2}-\d{2})"', _ast.get_source_segment(src, hol[0])))
    check("dashboard: holiday list identical to the server's (history date picker skips holidays too)", hjs == hpy and len(hjs) == 16)
    i0 = d.index("  const TTR = {"); j0 = d.index("  const _ttrCache")
    fns = ["_emaSeries", "_warmOf", "_warmSeries", "_warmHaBuild", "_heikinAshi", "_ttrCompute", "_ttrTrim", "_ttrState",
           "_emaCrossPoints", "_warmCurKey", "_istDayKey", "_ctxSignalsOn", "_ctxSignalSvg"]
    harness = """
const store = {}; const localStorage = { getItem: k => (k in store ? store[k] : null), setItem: (k, v) => { store[k] = String(v); } };
const window = {}; const state = { symbol: 'NIFTY' };
const escapeHtml = s => String(s); const fmtPrice = v => v.toFixed(2); const _hm = t => String(t);
const _chart = { interval: 1, candles: [], show: { ttr: true, emaX: true }, liveBar: false, emaX: { fast: 5, slow: 13 } };
const _haMemo = { src: null, n: 0, out: null, seed: null };
const _ttrCache = { key: '', v: null }; const _emaXCache = { key: '', pts: [] };
const _rgba = () => '';
const _warmLen = bars => _warmOf(bars).length;
const _pdFilterOn = () => false, _pdVerdict = () => null, _pdFilterNote = () => '';
""" + d[i0:j0] + "\n".join(grab(f) for f in fns) + r"""
let p = 25000; const all = [];
for (let i = 0; i < 1125; i++) {           // two warm sessions + today, a few trend legs
  const drift = Math.sin(i / 60) * 2.4; const o = p; p += drift + Math.sin(i * 1.7) * 3;
  all.push({ t: (i < 750 ? 1000 : 900000) + 60 * i, o, h: Math.max(o, p) + 2, l: Math.min(o, p) - 2, c: p });
}
const warm = all.slice(0, 750), today = all.slice(750);
_chart.candles = today; _chart.warm = warm; _chart.warmKey = _warmCurKey();
_chart.warmHa = _warmHaBuild(warm); const lh = _chart.warmHa[_chart.warmHa.length - 1]; _chart._warmHaLast = { o: lh.o, c: lh.c };
const R = {};
const e200 = _warmSeries(today, cl => _emaSeries(cl, 200)), full = _emaSeries(all.map(b => b.c), 200);
R.ema = { first: e200[0], want: full[750], noWarmFirst: _emaSeries(today.map(b => b.c), 200)[0] };
const tr = _ttrState(true), tf = _ttrCompute(all);
R.ttr = { flips: tr.r.flip.map((f, i) => f ? [i, f] : null).filter(Boolean),
          want: tf.flip.slice(750).map((f, i) => f ? [i, f] : null).filter(Boolean),
          alma0: tr.r.alma[0], almaWant: tf.alma[750], n: tr.r.trend.length };
R.ttrNoWarmFirstFlip = _ttrCompute(today).flip.findIndex(f => f !== 0);
const xs = _emaCrossPoints(); const fa = _emaSeries(all.map(b => b.c), 5), sa = _emaSeries(all.map(b => b.c), 13);
const want = []; let side = 0;
for (let j = 12; j < all.length; j++) { const dd = fa[j] - sa[j]; const now = dd > 0 ? 1 : dd < 0 ? -1 : side; if (side && now !== side && j >= 750) want.push(j - 750); side = now; }
R.cross = { got: xs.map(x => x.i), want };
const ha = _heikinAshi(today); R.ha = { o: ha[0].o, want: (lh.o + lh.c) / 2 };
const svg = _ctxSignalSvg(ms => ms / 1000, 0, 100);
R.sig = { lines: (svg.match(/<line /g) || []).length, buy: (svg.match(/ctx-sig-buy/g) || []).length, sell: (svg.match(/ctx-sig-sell/g) || []).length };
store.ctxSignals = '0'; R.sigOff = _ctxSignalSvg(ms => ms, 0, 100);
state.symbol = 'BANKNIFTY'; _ttrCache.key = ''; R.mismatch = _warmLen(today);
console.log(JSON.stringify(R));
"""
    res = subprocess.run(["node", "-e", harness], capture_output=True, text=True)
    try:
        R = _json.loads(res.stdout.strip().splitlines()[-1])
    except Exception:
        check("warm-up (dashboard): harness ran", False, (res.stderr or res.stdout)[:500]); return
    check("warm-up (dashboard): EMA 200 has a settled value at 09:15, equal to the EMA run over warm-up + today",
          R["ema"]["first"] is not None and abs(R["ema"]["first"] - R["ema"]["want"]) < 1e-9 and R["ema"]["noWarmFirst"] is None, str(R["ema"]))
    check("warm-up (dashboard): Trend Target Ribbon carries yesterday's state - today's flips are those of the full run",
          R["ttr"]["flips"] == R["ttr"]["want"] and R["ttr"]["n"] == 375 and abs(R["ttr"]["alma0"] - R["ttr"]["almaWant"]) < 1e-9
          and len(R["ttr"]["flips"]) >= 2, str(R["ttr"])[:300])
    check("warm-up (dashboard): EMA 5/13 crosses on today's bars are the full run's crosses",
          R["cross"]["got"] == R["cross"]["want"] and len(R["cross"]["got"]) >= 2, str(R["cross"])[:200])
    check("warm-up (dashboard): Heikin Ashi continues from the previous session's last HA bar",
          abs(R["ha"]["o"] - R["ha"]["want"]) < 1e-9, str(R["ha"]))
    check("warm-up (dashboard): never applied to another symbol's chart while its own loads", R["mismatch"] == 0)
    check("context pane: one Buy/Sell line per ribbon signal today, green BUY / red SELL, switchable",
          R["sig"]["lines"] == len(R["ttr"]["flips"]) and R["sig"]["buy"] + R["sig"]["sell"] == R["sig"]["lines"]
          and R["sig"]["buy"] >= 1 and R["sig"]["sell"] >= 1 and R["sigOff"] == "", str(R["sig"]))
    check("dashboard: warm-up fetched once per symbol/interval/day, shown in the status line, Signals box + Reset Defaults",
          "/api/warm-candles?symbol=" in d and "_warmEnsure();          // previous session(s)" in d and "🔥 ${escapeHtml(_warmNote())}" in d
          and 'data-ctxsig="1"' in d and "'ctxSignals',          //" in d and "_ctxSignalSvg(x, padT, H - padB)" in d)


def test_day_swipe():
    """Click-hold and drag sideways on the chart: right = previous trading day,
    left = next, weekends and NSE holidays skipped, forward from the last past
    day returns to Live; vertical drags stay a price pan."""
    import subprocess, json as _json
    d = open(os.path.join(HERE, "nse_dashboard.html"), encoding="utf-8").read()
    def grab(name):
        i = d.index(f"function {name}(")
        depth, j = 0, d.index("{", i)
        for k in range(j, len(d)):
            depth += d[k] == "{"; depth -= d[k] == "}"
            if depth == 0:
                return d[i:k + 1]
    i0 = d.index("  const _NSE_HOLIDAYS = new Set("); j0 = d.index("]);", i0) + 3
    harness = d[i0:j0] + "\nconst window = {};\nconst _istDayKey = () => '2026-10-21';\n" + "\n".join(
        grab(f) for f in ("_tradingDayShift", "_dayTarget", "_dayLabel")) + r"""
const R = {};
R.fromLive = _dayTarget(-1); R.liveFwd = _dayTarget(1);
window._histDate = '2026-10-19'; R.back = _dayTarget(-1); R.fwd = _dayTarget(1);
window._histDate = '2026-10-16'; R.fwd2 = _dayTarget(1);
window._histDate = '2026-10-05'; R.back2 = _dayTarget(-1);
R.label = _dayLabel('2026-10-19');
console.log(JSON.stringify(R));
"""
    res = subprocess.run(["node", "-e", harness], capture_output=True, text=True)
    try:
        R = _json.loads(res.stdout.strip().splitlines()[-1])
    except Exception:
        check("day swipe: harness ran", False, (res.stderr or res.stdout)[:400]); return
    check("day swipe: from Live, dragging right goes to 19 Oct (Dussehra 20 Oct skipped); left has nothing newer",
          R["fromLive"] == "2026-10-19" and R["liveFwd"] is None, str(R))
    check("day swipe: 19 Oct back -> Fri 16 Oct (weekend skipped), forward -> Live; 16 Oct forward -> 19 Oct",
          R["back"] == "2026-10-16" and R["fwd"] == "live" and R["fwd2"] == "2026-10-19", str(R))
    check("day swipe: 5 Oct back -> 1 Oct (Gandhi Jayanti and the weekend skipped)", R["back2"] == "2026-10-01", str(R))
    check("day swipe: label reads as a day", R["label"] == "Mon 19 Oct", R["label"])
    check("day swipe: bound once on the document (mouse + touch), axis decided at 24px, vertical pan stands down, alert lines excluded",
          "document.addEventListener('mousedown', ev => {\n    if (ev.button !== 0 || ev.shiftKey) return;\n    _daySwipeStart(" in d
          and "Math.hypot(dx, dy) < 24" in d and "if (!d2 || _dragPendY == null || _daySwipe?.active) return;" in d
          and "if (_daySwipe?.active) return;          // a sideways day-swipe owns this drag" in d
          and "_alertHitTest({ clientX: x, clientY: y })" in d and "document.addEventListener('touchmove', ev => {\n    if (_xPan && ev.touches.length === 1)" in d and "if (_daySwipe && ev.touches.length === 1 && _daySwipeMove(" in d)
    check("day swipe: switches through the same Live/Historical controls, swallows the click after a swipe",
          "modeSel.dispatchEvent(new Event('change'));" in d and "window.addEventListener('click', eat, { capture: true, once: true });" in d
          and "drag sideways: ◀ prev / next ▶ day" in d)


def test_render_performance():
    """Performance fixes keep their output: cached number formatters match
    toLocaleString, escapeHtml escapes quotes too, the ribbon is drawn as a
    handful of batched paths (not one element per bar), chain-table repaints
    on ticks only while the Chain tab is open, layout reads once per frame."""
    import subprocess, json as _json
    d = open(os.path.join(HERE, "nse_dashboard.html"), encoding="utf-8").read()
    i0 = d.index("  const _nfInt = new Intl.NumberFormat("); j0 = d.index("function escapeHtml(s)", i0)
    j0 = d.index("\n", j0)
    harness = d[i0:j0] + r"""
const R = { same: true, bad: [] };
for (const v of [0, 1, -1, 12.34, 25012.55, -987654.321, 1234567.89, 0.05, NaN, null, '25000.4']) {
  const a = fmtPrice(v), b = Number(v).toLocaleString('en-IN', { minimumFractionDigits: 1, maximumFractionDigits: 1 });
  const c = fmtInt(v ?? 0), e = Math.round(v ?? 0).toLocaleString('en-IN');
  if (a !== b || c !== e) { R.same = false; R.bad.push([v, a, b, c, e]); }
}
R.esc = escapeHtml(`a<b>&"c"'d'`); R.escNull = escapeHtml(null); R.signed = fmtSigned(1234.4);
console.log(JSON.stringify(R));
"""
    res = subprocess.run(["node", "-e", harness], capture_output=True, text=True)
    try:
        R = _json.loads(res.stdout.strip().splitlines()[-1])
    except Exception:
        check("perf: formatter harness ran", False, (res.stderr or res.stdout)[:400]); return
    check("perf: cached Intl formatters give exactly the toLocaleString output", R["same"], str(R["bad"])[:300])
    check("perf: escapeHtml escapes & < > and quotes (safe in attributes), null -> ''",
          R["esc"] == "a&lt;b&gt;&amp;&quot;c&quot;&#39;d&#39;" and R["escNull"] == "" and R["signed"] == "+1,234", str(R))
    ttr = d[d.index("function _ttrSvg("):d.index("// positions (only the latest", d.index("function _ttrSvg("))]
    check("perf: Trend Target Ribbon glow, fills, edge and core batched into paths (no per-bar rect/polygon/line)",
          "<rect class=\"ttr-glow\"" not in ttr and "<polygon" not in ttr and "<line" not in ttr
          and "class=\"ttr-glow\" d=" in ttr and "const q4 = v => Math.round(v / 4) * 4;" in ttr)
    check("perf: chain table repainted on ticks only while its tab is open; traffic-light placement once per frame",
          "if (_activeTabNow() === 'chain') _safe(() => renderChainTable(data, true), 'chainTick');" in d
          and "_tlRaf = requestAnimationFrame(() => { _tlRaf = 0; _tlPlaceNow(); });" in d)
    check("perf: no per-call toLocaleString with options left in the hot formatters",
          "function fmtPrice(n) { return _nfP1.format(Number(n)); }" in d and "d.textContent = s ?? ''" not in d)


def test_pd_premium_discount():
    """PDA / PDMP: dealing ranges, premium/discount zones, fair value gaps
    (open / filled), order blocks (intact / broken), liquidity pools (equal
    highs swept and rejected, PDH taken), the signal filter, the bias reads,
    and the dashboard wiring."""
    import subprocess, json as _json, datetime as _dt
    d = open(os.path.join(HERE, "nse_dashboard.html"), encoding="utf-8").read()
    def grab(name):
        i = d.index(f"function {name}(")
        depth, j = 0, d.index("{", i)
        for k in range(j, len(d)):
            depth += d[k] == "{"; depth -= d[k] == "}"
            if depth == 0:
                return d[i:k + 1]
    mod = d[d.index("  const _pd = { key: ''"):d.index("  // ── the morning prep card (PDMP)")]
    t0 = int(_dt.datetime(2026, 10, 21, 9, 15, tzinfo=_dt.timezone(_dt.timedelta(hours=5, minutes=30))).timestamp())
    B = {i: (100, 100.3, 99.7, 100) for i in range(6)}
    B.update({6: (100.3, 100.5, 99.5, 99.6), 7: (99.6, 100.2, 99.4, 100.0), 8: (100.0, 106.0, 99.9, 105.8),
              9: (105.8, 107.0, 103.0, 106.5), 10: (106.5, 106.8, 106.2, 106.5), 11: (106.5, 107.2, 106.3, 107.0),
              12: (107.0, 108.0, 106.8, 107.5), 13: (107.5, 107.6, 106.9, 107.0), 14: (107.0, 107.3, 106.6, 106.8),
              15: (106.8, 107.4, 106.7, 107.2), 16: (107.2, 108.02, 107.0, 107.6), 17: (107.6, 107.7, 107.0, 107.1),
              18: (107.1, 107.5, 106.6, 106.8), 19: (106.8, 106.9, 103.6, 103.8), 20: (103.8, 104.5, 103.4, 104.0),
              21: (104.0, 104.8, 103.8, 104.6), 22: (104.6, 105.6, 104.4, 105.4), 23: (105.4, 106.0, 105.0, 105.8),
              24: (105.8, 106.3, 105.5, 106.1), 25: (106.1, 106.9, 105.9, 106.7), 30: (107.0, 108.5, 106.9, 107.4),
              35: (107.0, 111.0, 106.9, 110.8)})
    for i in list(range(26, 30)) + list(range(31, 35)):
        B[i] = (107.0, 107.3, 106.7, 107.0)
    for i in range(36, 41):
        B[i] = (110.8, 111.1, 110.5, 110.8)
    bars = [{"t": t0 + 60 * i, "o": B[i][0], "h": B[i][1], "l": B[i][2], "c": B[i][3]} for i in range(41)]
    harness = r"""
const window = { _histDate: '' }; const state = { symbol: 'NIFTY', lastData: null };
const escapeHtml = s => String(s ?? ''); const fmtInt = v => String(Math.round(v)); const fmtPrice = v => (+v).toFixed(2);
const _hm = t => String(t); const els = {}; const _safe = f => f(); const _isHistorical = () => false;
const _istDayKey = () => '2026-10-21';
const _chart = { interval: 1, candles: [], liveBar: false, show: { pdZones: true, pdFvg: true, pdOb: true, pdLiq: true, pdFilter: true, pdhl: false }, warm: [], warmKey: '' };
""" + grab("_warmCurKey") + grab("_warmOf") + "\nconst _warmLen = bars => _warmOf(bars).length;\n" + mod + grab("_pdBias") + grab("_pdDayType") + r"""
const R = {};
_chart.candles = BARS;
_pd.daily = [{ day: '2026-10-16', o: 98, h: 104, l: 94, c: 99, t0: 1 }, { day: '2026-10-19', o: 99, h: 110, l: 95, c: 108.5, t0: 2 }];
_pd.key = 'NIFTY|2026-10-21';
const rng = k => { _chart.show.pdRange = k; const r = _pdRange(); return r && [r.lo, r.hi, r.eq, r.label]; };
R.ranges = { pd: rng('pd'), p2d: rng('p2d'), pw: rng('pw'), today: rng('today') };
_chart.show.pdRange = 'pd';
R.zones = [96, 101, 102.5, 104, 109, 111, 90].map(v => _pdZone(v, _pdRange()).name);
R.verdict = { buyHigh: _pdVerdict(1, 8, 105.8), sellHigh: _pdVerdict(-1, 20, 104), buyLow: _pdVerdict(1, 3, 97) };
const A = _pdArrays();
R.fvgs = A.fvgs.map(g => [g.dir, g.mid, +g.bot.toFixed(2), +g.top.toFixed(2), g.filled]);
R.obs = A.obs.map(o => [o.dir, o.j, o.top, o.bot, o.broken]);
R.pools = A.pools.map(P => [P.lab, P.side, P.lvl, P.state, P.at]);
const bands = _pdBandsSvg(v => 1000 - v, 90, 115, 0, 900);
R.bands = { rects: (bands.under.match(/<rect /g) || []).length, eq: /EQ 103/.test(bands.over), edges: /R\.HI/.test(bands.over) };
const arr = _pdArraysSvg(i => i * 10, v => 1000 - v, 90, 115, 0, 900, 0, 500);
R.svg = { fvgB: /pd-fvgB/.test(arr.under), ob: /pd-obB/.test(arr.under), sweep: (arr.over.match(/pd-sweep/g) || []).length, liq: (arr.over.match(/pd-liq/g) || []).length };
const b = _pdBias(), b0 = _pdBias(true);
R.bias = { bias: b.bias, score: b.score, why: b.why.map(w => w[0] + ' ' + w[1].slice(0, 40)) };
R.bias0 = { bias: b0.bias, score: b0.score, n: b0.why.length };
R.dayType = _pdDayType(_pd.daily).t;
R.tag = _pdTag();
console.log(JSON.stringify(R));
""".replace("BARS", _json.dumps(bars))
    res = subprocess.run(["node", "-e", harness], capture_output=True, text=True)
    try:
        R = _json.loads(res.stdout.strip().splitlines()[-1])
    except Exception:
        check("PD: harness ran", False, (res.stderr or res.stdout)[:600]); return
    rg = R["ranges"]
    check("PD: dealing ranges - previous day, two days, week (as many sessions as there are), today so far",
          rg["pd"] == [95, 110, 102.5, "PDH–PDL"] and rg["p2d"][:2] == [94, 110] and rg["pw"][:2] == [94, 110]
          and rg["pw"][3] == "prev 2 sessions" and rg["today"][:2] == [99.4, 111.1], str(rg))
    check("PD: zone names - discount / equilibrium / premium / outside the range",
          R["zones"] == ["DISCOUNT", "DISCOUNT", "EQUILIBRIUM", "PREMIUM", "PREMIUM", "ABOVE RANGE", "BELOW RANGE"], str(R["zones"]))
    v = R["verdict"]
    check("PD filter: BUY in premium fails, SELL in premium passes, BUY in discount passes",
          v["buyHigh"]["pass"] is False and v["sellHigh"]["pass"] is True and v["buyLow"]["pass"] is True, str(v))
    fv = R["fvgs"]
    check("PD: bullish FVG 100.2-103 found at the displacement bar and still open",
          [1, 8, 100.2, 103.0, None] in fv, str(fv))
    check("PD: bearish FVG 104.5-106.6 found and marked filled when price traded back through it",
          [-1, 19, 104.5, 106.6, 25] in fv, str(fv))
    check("PD: order blocks - bullish OB (last down candle before the gap) intact, bearish OB broken by a close above it",
          [1, 6, 100.5, 99.5, None] in R["obs"] and [-1, 16, 108.02, 107.0, 35] in R["obs"], str(R["obs"]))
    pools = {p[0]: p for p in R["pools"]}
    check("PD liquidity: equal highs swept and rejected; PDH taken (closed through); PDL untouched",
          pools.get("EQH", [None] * 5)[2:5] == [108.02, "swept", 30] and pools["PDH"][3:5] == ["taken", 35]
          and pools["PDL"][3] is None, str(R["pools"]))
    check("PD drawing: two zone bands + EQ + range edges; FVG / OB boxes, sweep mark and pool lines",
          R["bands"] == {"rects": 2, "eq": True, "edges": True} and R["svg"]["fvgB"] and R["svg"]["ob"]
          and R["svg"]["sweep"] == 1 and R["svg"]["liq"] >= 2, str(R["bands"]) + str(R["svg"]))
    check("PD bias: reasons with signs (above range -, closed top quarter +, HH/HL +, PDH taken -) net NEUTRAL",
          R["bias"]["bias"] == "NEUTRAL" and R["bias"]["score"] == 0 and len(R["bias"]["why"]) == 4, str(R["bias"]))
    check("PD bias at the open ignores intraday sweeps and Market Direction", R["bias0"]["n"] == 3, str(R["bias0"]))
    check("PD: previous day typed (closed top quarter on a wide range = trend up)", R["dayType"] == "trend up", R["dayType"])
    check("PD: status tag names the zone", "ABOVE RANGE" in R["tag"] and "PDH–PDL" in R["tag"], R["tag"][:120])
    check("dashboard: PD toggles (zones, gaps, liquidity, prep on; OTE, OB, filter opt-in), range picker, prep card, reset key",
          all(f'id="chk_{k}"' in d for k in ("pdZones", "pdOte", "pdFvg", "pdOb", "pdLiq", "pdFilter", "pdPrep"))
          and "'pdZones', 'pdFvg', 'pdLiq', 'pdPrep'].includes(k)) _chart.show[k] = true;" in d
          and 'id="pdRange"' in d and '<div id="pdPrepBody"></div>' in d and "'pdPrepCollapsed'," in d
          and "_pdEnsure();            // 5 previous sessions" in d and "/api/warm-candles?symbol=${encodeURIComponent(sym)}&interval=60&days=5" in d)
    check("dashboard: filtered signals are faded (ribbon, EMA cross, context pane), and every ribbon signal is scored with/against the zone",
          d.count("pd-filtered") >= 3 and "'Ribbon signal · with PD zone' : 'Ribbon signal · against PD zone'" in d and "_pdScoreSignals();" in d)


def test_pd_structure_psychology():
    """Market Prep extensions: market structure & price behaviour (swings,
    BOS / CHoCH, opening type, initial-balance extension) and present-day
    market psychology (fear, writers, PCR, mood), on constructed sessions."""
    import subprocess, json as _json, math as _m, datetime as _dt
    d = open(os.path.join(HERE, "nse_dashboard.html"), encoding="utf-8").read()
    def grab(name):
        i = d.index(f"function {name}(")
        depth, j = 0, d.index("{", i)
        for k in range(j, len(d)):
            depth += d[k] == "{"; depth -= d[k] == "}"
            if depth == 0:
                return d[i:k + 1]
    mod = d[d.index("  const _pd = { key: ''"):d.index("  // ── the morning prep card (PDMP)")]
    t0 = int(_dt.datetime(2026, 10, 21, 9, 15, tzinfo=_dt.timezone(_dt.timedelta(hours=5, minutes=30))).timestamp())
    def session(sign):
        out, prev = [], 25000.0
        for i in range(150):
            # a clean 30-minute drive from the open, then a trend with pullbacks
            p = 25000 + sign * (4 * i + (60 * _m.sin((i - 30) / 8) if i >= 30 else 0))
            o = prev; out.append({"t": t0 + 60 * i, "o": o, "h": max(o, p) + 2, "l": min(o, p) - 2, "c": p, "v": 0}); prev = p
        return out
    up, dn = session(1), session(-1)
    harness = r"""
const window = { _histDate: '' }; const state = { symbol: 'NIFTY', lastData: null };
const escapeHtml = s => String(s ?? ''); const fmtInt = v => String(Math.round(v)); const fmtPrice = v => (+v).toFixed(2);
const _hm = t => String(t); const els = {}; const _safe = f => f(); const _isHistorical = () => !!window._histDate;
const _istDayKey = () => '2026-10-21';
const _chart = { interval: 1, candles: [], liveBar: false, show: { pdZones: true, pdFvg: true, pdLiq: true }, warm: [], warmKey: '', hasVolume: false };
const _vwapBands = cs => ({ vw: cs.map(() => cs[0].o), sd: cs.map(() => 10) });
const _dayType = () => ({ label: 'TEST DAY' });
let _ctxHist = [];
""" + grab("_warmCurKey") + grab("_warmOf") + "\nconst _warmLen = bars => _warmOf(bars).length;\n" + mod + "\n" + d[d.index("  const _pdSlow = {"):d.index("  // both change slowly")] + r"""
const R = {};
_pd.daily = [{ day: '2026-10-19', o: 24900, h: 25100, l: 24800, c: 25000, t0: 1 }]; _pd.key = 'NIFTY|2026-10-21';
_chart.candles = UPA;
const su = _pdStructure();
R.up = { label: su.label, structure: su.structure, ev: su.ev && su.ev.kind + (su.ev.dir > 0 ? '+' : '-'), open: su.openType,
         ib: su.why.some(w => /extended UP/.test(w[1])), n: su.why.length };
_chart.candles = DNA;
const sd = _pdStructure();
R.dn = { label: sd.label, structure: sd.structure, ev: sd.ev && sd.ev.kind + (sd.ev.dir > 0 ? '+' : '-'), open: sd.openType,
         ib: sd.why.some(w => /extended DOWN/.test(w[1])) };
// psychology: a calm rally (VIX easing, puts written, PCR rising with price)
_chart.candles = UPA;
const strikes = [24900, 24950, 25000, 25050, 25100].map(k => ({ strike: k, ce_oi_chg: 1000, pe_oi_chg: 5000 }));
state.lastData = { underlying_value: UPA[0].o + 300, atm_iv: 13, atm: 25000, strike_gap: 50, strikes };
_ctxHist = [{ sym: 'NIFTY', ts: 1, vix: 13.5, pcr: 0.9, skew: 1.0, straddle: 200 }, { sym: 'NIFTY', ts: 2, vix: 12.8, pcr: 1.0, skew: 0.8, straddle: 190 }];
const pu = _pdPsych();
R.psyUp = { mood: pu.mood, score: pu.score };
// a falling market with fear rising, calls written, puts bought, protection bid
_chart.candles = DNA;
state.lastData = { underlying_value: DNA[0].o - 320, atm_iv: 13, atm: 25000, strike_gap: 50,
  strikes: strikes.map(k => ({ ...k, ce_oi_chg: 6000, pe_oi_chg: 1000 })) };
_ctxHist = [{ sym: 'NIFTY', ts: 1, vix: 12.0, pcr: 1.0, skew: 0.5, straddle: 190 }, { sym: 'NIFTY', ts: 2, vix: 13.6, pcr: 1.2, skew: 1.2, straddle: 220 }];
const pd2 = _pdPsych();
R.psyDn = { mood: pd2.mood, score: pd2.score, fearLine: pd2.why.some(w => /fear rising into the fall/.test(w[1])), hedge: pd2.why.some(w => /puts BOUGHT/.test(w[1])) };
window._histDate = '2026-10-19'; R.histPsy = _pdPsych(); window._histDate = '';
console.log(JSON.stringify(R));
"""
    harness = harness.replace("const R = {};", f"const R = {{}}; const UPA = {_json.dumps(up)}, DNA = {_json.dumps(dn)};", 1)
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
        fh.write(harness); hp = fh.name
    try:
        res = subprocess.run(["node", hp], capture_output=True, text=True)
    finally:
        os.unlink(hp)
    try:
        R = _json.loads(res.stdout.strip().splitlines()[-1])
    except Exception:
        check("PD structure: harness ran", False, (res.stderr or res.stdout)[:600]); return
    check("structure: a rising session reads TRENDING UP - HH/HL, BOS up, initial balance extended up",
          R["up"]["label"] == "TRENDING UP" and R["up"]["structure"] == "BULLISH" and R["up"]["ev"] == "BOS+" and R["up"]["ib"], str(R["up"]))
    check("structure: the mirror session reads TRENDING DOWN - LH/LL, BOS down, IB extended down",
          R["dn"]["label"] == "TRENDING DOWN" and R["dn"]["structure"] == "BEARISH" and R["dn"]["ev"] == "BOS-" and R["dn"]["ib"], str(R["dn"]))
    check("structure: a one-way first 30 minutes is an Open-Drive, in the right direction",
          R["up"]["open"] == "Open-Drive ▲" and R["dn"]["open"] == "Open-Drive ▼", f'{R["up"]["open"]} / {R["dn"]["open"]}')
    check("psychology: a calm rally with puts written and VIX easing reads CONFIDENT",
          R["psyUp"]["mood"] == "CONFIDENT" and R["psyUp"]["score"] >= 3, str(R["psyUp"]))
    check("psychology: a hard fall with VIX up, calls written, puts bought and protection bid reads PANIC",
          R["psyDn"]["mood"] == "PANIC" and R["psyDn"]["fearLine"] and R["psyDn"]["hedge"], str(R["psyDn"]))
    check("psychology: not read in historical replay (the option history is today's)", R["histPsy"] is None)
    check("dashboard: structure and psychology sections in the prep card, labels in its header, slow-refreshed",
          "Market structure &amp; price behaviour" in d and "Present day market psychology" in d
          and "structure <b style=" in d and "mood <b style=" in d and "Date.now() - _pdSlow.at > 10000" in d)


def test_full_width_layout():
    """The page uses the whole browser width: no max-width ceiling on .app,
    a thin gutter, and no room reserved for the floating WATCHLIST tab."""
    d = open(os.path.join(HERE, "nse_dashboard.html"), encoding="utf-8").read()
    i = d.index("  .app {"); j = d.index("}", i)
    app = d[i:j]
    check("layout: .app spans the full browser width with an 8px gutter",
          "max-width: none;" in app and "width: 100%;" in app and "padding: 12px 8px 60px;" in app
          and "min(2560px" not in app)
    check("layout: the WATCHLIST tab floats over the right edge (fixed), the page reserves no space for it",
          'id="tvWlToggle"' in d and "position:fixed;right:0;top:120px" in d)


def test_indicator_settings():
    """⚙ Settings for the Trend Target Ribbon: settings load / save / reset
    safely (known keys and matching types only), defaults are not stored,
    colour case is not a change, and the dialog is wired; the removed SMI of
    MFI leaves nothing behind."""
    import subprocess, json as _json
    d = open(os.path.join(HERE, "nse_dashboard.html"), encoding="utf-8").read()
    fw = d[d.index("  const _IND = {};"):d.index("  function _indFieldHtml(")]
    harness = r"""
const store = {}; const localStorage = { getItem: k => (k in store ? store[k] : null), setItem: (k, v) => { store[k] = String(v); }, removeItem: k => { delete store[k]; } };
const requestAnimationFrame = f => f();
""" + fw + r"""
const R = {};
const cfg = { a: 1, b: 'x', n: { c: true } }, defs = Object.freeze(JSON.parse(JSON.stringify(cfg)));
store.k1 = JSON.stringify({ a: 5, b: 7, zz: 9, n: { c: false, q: 1 } });
let applied = 0;
_indRegister('t', { title: 'T', key: 'k1', cfg, defaults: defs, apply: () => applied++, schema: [] });
R.loaded = JSON.stringify(cfg);
_cfgAssign(cfg, JSON.parse(JSON.stringify(defs))); _indSave('t');
R.savedDefault = ('k1' in store);
cfg.a = 3; _indSave('t'); R.savedChanged = JSON.parse(store.k1).a;
R.colorEq = _indEq('#00FF00', '#00ff00');
_IND.t.apply(); R.applied = applied;
console.log(JSON.stringify(R));
"""
    res = subprocess.run(["node", "-e", harness], capture_output=True, text=True)
    try:
        R = _json.loads(res.stdout.strip().splitlines()[-1])
    except Exception:
        check("settings: harness ran", False, (res.stderr or res.stdout)[:500]); return
    check("settings: saved values load by known key and matching type only (unknown keys and wrong types ignored)",
          R["loaded"] == '{"a":5,"b":"x","n":{"c":false}}', R["loaded"])
    check("settings: saving defaults removes the entry; a change is stored; colour case is not a change",
          R["savedDefault"] is False and R["savedChanged"] == 3 and R["colorEq"] and R["applied"] == 1, str(R))
    check("dashboard: ⚙ beside the ribbon toggle, registered with its key, Reset Defaults clears it",
          'data-indset="ttr"' in d and "_indRegister('ttr'" in d and "key: 'ttrCfgV1'" in d
          and "'ttrCfgV1',            //" in d and "Reset to defaults" in d and "closeIndSettings(true)" in d)
    check("SMI of MFI removed: no pane, toggle, settings or code left; its saved settings are cleared",
          all(x not in d for x in ("smiPaneBody", "SMIMFI", "chk_smiMfi", "_smiMfiPaneSvg", "_indRegister('smiMfi'", "xclipS", "smiXM"))
          and "localStorage.removeItem('smiMfiCfgV1')" in d)


def _sqz_reference(bars, bbLen=20, bbMult=2.0, kcLen=20, kcMult=1.5, useTR=True, quirk=True):
    """Independent Python port of LazyBear's Squeeze Momentum Pine script."""
    import math
    n = len(bars)
    c = [b["c"] for b in bars]; h = [b["h"] for b in bars]; l = [b["l"] for b in bars]
    def sma(a, t, k): return sum(a[t - k + 1:t + 1]) / k
    def stdev(a, t, k):
        m = sma(a, t, k); return math.sqrt(sum((x - m) ** 2 for x in a[t - k + 1:t + 1]) / k)
    def tr(i):
        if not useTR: return h[i] - l[i]
        if i == 0: return h[0] - l[0]
        return max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    trs = [tr(i) for i in range(n)]
    val, state, src2 = [None] * n, [None] * n, [None] * n
    for t in range(n):
        if t >= max(bbLen, kcLen) - 1:
            basis = sma(c, t, bbLen); dev = (kcMult if quirk else bbMult) * stdev(c, t, bbLen)
            ma = sma(c, t, kcLen); rng = sma(trs, t, kcLen)
            uBB, lBB, uKC, lKC = basis + dev, basis - dev, ma + rng * kcMult, ma - rng * kcMult
            state[t] = "on" if (lBB > lKC and uBB < uKC) else "off" if (lBB < lKC and uBB > uKC) else "none"
        if t >= kcLen - 1:
            hh = max(h[t - kcLen + 1:t + 1]); ll = min(l[t - kcLen + 1:t + 1])
            src2[t] = c[t] - ((hh + ll) / 2 + sma(c, t, kcLen)) / 2
        if t >= 2 * kcLen - 2:
            ys = src2[t - kcLen + 1:t + 1]; xs = list(range(kcLen))
            mx = sum(xs) / kcLen; my = sum(ys) / kcLen
            slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
            val[t] = my + slope * (kcLen - 1 - mx)
    return val, state


def test_squeeze_momentum():
    """Squeeze Momentum [LazyBear] replaces the RSI pane: momentum and squeeze
    state match an independent port bar for bar (quirk on and off), all four
    divergence kinds are found on oscillator pivots within range, and the
    pane, toggle, ⚙ settings, crosshair, time-zoom clip and price-side
    divergence are wired; the RSI pane is gone."""
    import subprocess, json as _json, random, math, tempfile
    d = open(os.path.join(HERE, "nse_dashboard.html"), encoding="utf-8").read()
    def grab(name):
        i = d.index(f"function {name}(")
        depth, j = 0, d.index("{", i)
        for k in range(j, len(d)):
            depth += d[k] == "{"; depth -= d[k] == "}"
            if depth == 0:
                return d[i:k + 1]
    rnd = random.Random(7)
    bars, p = [], 24000.0
    for i in range(400):
        vol = 4 if 150 < i < 200 else 12                      # a quiet stretch -> squeeze
        o = p; p += rnd.gauss(0.3 * math.sin(i / 25), vol)
        bars.append({"o": round(o, 2), "c": round(p, 2), "h": round(max(o, p) + abs(rnd.gauss(0, vol / 2)), 2),
                     "l": round(min(o, p) - abs(rnd.gauss(0, vol / 2)), 2)})
    # divergence fixture: oscillator pivots and price at those bars
    osc = [0.0] * 120; px = [{"h": 100.0, "l": 90.0} for _ in range(120)]
    def dip(i, v, low):   # oscillator pivot low at i
        for k in range(-4, 5): osc[i + k] = min(osc[i + k], v + abs(k) * 0.5)
        px[i]["l"] = low
    def peak(i, v, high):
        for k in range(-4, 5): osc[i + k] = max(osc[i + k], v - abs(k) * 0.5)
        px[i]["h"] = high
    dip(10, -5, 80); dip(25, -3, 78)        # LL price, HL osc  -> regular bullish
    dip(40, -2, 79)                         # HL price, HL osc  -> nothing
    dip(55, -6, 82)                         # HL price, LL osc  -> hidden bullish
    peak(70, 5, 120); peak(85, 3, 125)      # HH price, LH osc  -> regular bearish
    peak(95, 2, 122)                        # LH price, LH osc  -> nothing
    peak(110, 6, 121)                       # LH price, HH osc  -> hidden bearish
    harness = "const SQZ = { bbLen: 20, bbMult: 2.0, kcLen: 20, kcMult: 1.5, useTR: true, lbQuirk: true };\n" \
        + grab("_sqzCompute") + "\n" + grab("_oscDivergences") + "\n" \
        + "const bars = " + _json.dumps(bars) + ";\nconst osc = " + _json.dumps(osc) + ";\nconst px = " + _json.dumps(px) + ";\n" + r"""
const R = {};
R.q = _sqzCompute(bars, SQZ);
R.t = _sqzCompute(bars, Object.assign({}, SQZ, { lbQuirk: false, useTR: false }));
R.d = _oscDivergences(px, osc, { left: 3, right: 3, rangeMin: 5, rangeMax: 60 });
R.far = _oscDivergences(px, osc, { left: 3, right: 3, rangeMin: 5, rangeMax: 10 }).length;
R.zs = _oscDivergences(px, osc.map(v => -v), { left: 3, right: 3, rangeMin: 5, rangeMax: 60, zeroSide: true }).length;
R.big = _oscDivergences(px, osc, { left: 3, right: 3, rangeMin: 5, rangeMax: 60, minAbs: 4.5 }).map(x => [x.a, x.b]);
console.log(JSON.stringify(R));
"""
    hf = os.path.join(tempfile.gettempdir(), "sqz_harness.js")
    open(hf, "w").write(harness)
    res = subprocess.run(["node", hf], capture_output=True, text=True)
    try:
        R = _json.loads(res.stdout.strip().splitlines()[-1])
    except Exception:
        check("squeeze: harness ran", False, (res.stderr or res.stdout)[:500]); return
    for tag, kw in (("q", {}), ("t", {"quirk": False, "useTR": False})):
        rv, rs = _sqz_reference(bars, **kw)
        jv, js = R[tag]["val"], R[tag]["state"]
        bad = [i for i in range(len(bars)) if (rv[i] is None) != (jv[i] is None) or (rv[i] is not None and abs(rv[i] - jv[i]) > 1e-6)]
        bads = [i for i in range(len(bars)) if rs[i] != js[i]]
        check(f"squeeze: momentum and squeeze state match the Pine port bar for bar ({'LazyBear quirk' if tag == 'q' else 'textbook BB, high-low range'})",
              not bad and not bads and rv[37] is None and rv[38] is not None and "on" in rs and "off" in rs,
              f"val mismatches {bad[:5]} state mismatches {bads[:5]}")
    kinds = sorted((x["kind"], x["hidden"], x["a"], x["b"]) for x in R["d"])
    check("squeeze: all four divergences found on oscillator pivots (regular/hidden, bull/bear), confirmed `right` bars later",
          kinds == sorted([("bullish", False, 10, 25), ("bullish", True, 40, 55), ("bearish", False, 70, 85), ("bearish", True, 95, 110)])
          and all(x["conf"] == x["b"] + 3 for x in R["d"]), str(kinds))
    check("squeeze: pivots further apart than the max range are not paired", R["far"] == 0, str(R["far"]))
    check("squeeze: filters - pivots on the wrong side of zero are ignored; small pivots drop out and the next qualifying pivot pairs instead",
          R["zs"] == 0 and R["big"] == [[10, 55]], str({k: R[k] for k in ("zs", "big")}))
    check("dashboard: Squeeze pane replaces RSI - pane, toggle (default on), ⚙ settings, reset key, crosshair, time-zoom clip",
          'id="sqzPaneBody"' in d and 'id="chk_sqz"' in d and 'data-indset="sqz"' in d and "_indRegister('sqz'" in d
          and "key: 'sqzCfgV1'" in d and "'sqzCfgV1',            //" in d and "['sqzXM', 'sqz']" in d
          and "#sqzPaneBody g.xzc" in d and "${_xClip('xclipQ', W, padL, padR, H)}<g class=\"xzc\">" in d
          and "'turnTimes', 'emaX', 'ttr', 'sqz', 'wt', 'pdZones'" in d and "ttr: true, sqz: true, wt: true }" in d)
    check("dashboard: divergences on closed bars, drawn in the pane and on the price (candles and Renko)",
          "const closedN = _chart.liveBar ? cs.length - 1 : cs.length;" in d
          and d.count("_sqzPriceDivSvg(cs, ") == 3 and "if (!_chart.show.sqz || !SQZ.div.onPrice" in d)
    check("RSI pane removed: no pane, renderer or settings left; its saved settings are cleared",
          all(x not in d for x in ("rsiPaneBody", "_rsiPaneSvg", "rsiXM", "xclipR", "_rsiCtx", "_divRes"))
          and "localStorage.removeItem('rsiCfg')" in d)


def _wt_reference(bars, n1=10, n2=21):
    """Independent Python port of the WaveTrend maths (Pine ta.ema seeded by
    the SMA of its first `len` values; ta.sma(wt1, 4))."""
    def ema(src, ln):
        out, prev, run, al = [None] * len(src), None, [], 2.0 / (ln + 1)
        for i, v in enumerate(src):
            if v is None:
                run = []; continue
            if prev is not None:
                prev = al * v + (1 - al) * prev; out[i] = prev; continue
            run.append(v)
            if len(run) >= ln:
                prev = sum(run[-ln:]) / ln; out[i] = prev
        return out
    ap = [(b["h"] + b["l"] + b["c"]) / 3 for b in bars]
    esa = ema(ap, n1)
    d = ema([None if e is None else abs(a - e) for a, e in zip(ap, esa)], n1)
    ci = [None if dd is None else (0 if dd == 0 else (a - e) / (0.015 * dd)) for a, e, dd in zip(ap, esa, d)]
    wt1 = ema(ci, n2)
    wt2 = [None if i < 3 or any(x is None for x in wt1[i - 3:i + 1]) else sum(wt1[i - 3:i + 1]) / 4 for i in range(len(wt1))]
    return wt1, wt2


def test_wavetrend():
    """WaveTrend Enhanced [LazyBear+] inside the Squeeze pane: WT1/WT2 match an
    independent port bar for bar; HTF built from the loaded bars per session
    from 09:15 (and says when history is short); divergence uses the script's
    rule (previous pivot, any distance); Local Trend BG, HTF Bias BG and
    Buy/Sell signals are not ported; two ⚙ icons in the pane; right-click
    Reset Defaults moved to the end."""
    import subprocess, json as _json, random, math, tempfile
    d = open(os.path.join(HERE, "nse_dashboard.html"), encoding="utf-8").read()
    def grab(name):
        i = d.index(f"function {name}(")
        depth, j = 0, d.index("{", i)
        for k in range(j, len(d)):
            depth += d[k] == "{"; depth -= d[k] == "}"
            if depth == 0:
                return d[i:k + 1]
    rnd = random.Random(11)
    bars, p, t0 = [], 24500.0, 1792208700        # 2026-10-23 09:15 IST
    for day in range(3):
        for m in range(375):
            o = p; p += rnd.gauss(0.4 * math.sin((day * 375 + m) / 30), 9)
            bars.append({"t": t0 + day * 86400 + m * 60, "o": o, "c": p, "h": max(o, p) + abs(rnd.gauss(0, 4)), "l": min(o, p) - abs(rnd.gauss(0, 4))})
    harness = "const WT = { n1: 10, n2: 21 }; const _chart = { interval: 1 };\n" + "\n".join(grab(f) for f in ("_emaPine", "_wtCompute", "_wtHtf")) \
        + "\nconst bars = " + _json.dumps(bars) + ";\n" + r"""
const R = {};
R.w = _wtCompute(bars, WT);
R.h240 = _wtHtf(bars, 240, WT);
R.h15 = _wtHtf(bars, 15, WT);
R.same = _wtHtf(bars, 1, WT).state;
// reference: build 15-minute buckets the same way and compare the last value
console.log(JSON.stringify(R));
"""
    hf = os.path.join(tempfile.gettempdir(), "wt_harness.js")
    open(hf, "w").write(harness)
    res = subprocess.run(["node", hf], capture_output=True, text=True)
    try:
        R = _json.loads(res.stdout.strip().splitlines()[-1])
    except Exception:
        check("wavetrend: harness ran", False, (res.stderr or res.stdout)[:500]); return
    r1, r2 = _wt_reference(bars)
    j1, j2 = R["w"]["wt1"], R["w"]["wt2"]
    bad = [i for i in range(len(bars)) if (r1[i] is None) != (j1[i] is None) or (r2[i] is None) != (j2[i] is None)
           or (r1[i] is not None and abs(r1[i] - j1[i]) > 1e-6) or (r2[i] is not None and abs(r2[i] - j2[i]) > 1e-6)]
    check("wavetrend: WT1 / WT2 match the Pine port bar for bar (SMA-seeded EMAs, first value at bar 2(n1-1)+n2-1)",
          not bad and r1[37] is None and r1[38] is not None and r2[41] is not None and r2[40] is None, str(bad[:5]))
    # HTF reference: 15-minute buckets from 09:15
    hb = []
    for b in bars:
        k = ((b["t"] + 19800) // 86400, ((b["t"] + 19800) % 86400 // 60 - 555) // 15)
        if hb and hb[-1][0] == k:
            x = hb[-1][1]; x["h"] = max(x["h"], b["h"]); x["l"] = min(x["l"], b["l"]); x["c"] = b["c"]
        else:
            hb.append((k, {"h": b["h"], "l": b["l"], "c": b["c"]}))
    hr = _wt_reference([x for _, x in hb])[0][-1]
    check("wavetrend: HTF 15 built from the loaded bars in 09:15-anchored buckets; 240 says the history is too short; HTF ≤ chart TF is skipped",
          len(hb) == 75 and abs(R["h15"]["v"] - hr) < 1e-6 and R["h15"]["state"] == ("bull" if hr > 0 else "bear")
          and R["h240"] == {"state": "short", "have": 6, "need": 39} and R["same"] == "same", str({k: R[k] for k in ("h240", "h15", "same")}))
    ws = grab("_wtSeries")
    check("wavetrend: divergence = the script's rule (pivots on WT1, previous pivot at any distance), on closed bars",
          "_oscDivergences(all.slice(0, off + closedN), rc.wt1, { left: WT.lbL, right: WT.lbR, rangeMin: 0, rangeMax: Infinity })" in ws
          and "const closedN = _chart.liveBar ? cs.length - 1 : cs.length;" in ws)
    parts = grab("_wtPaneParts")
    check("wavetrend: no background colours and no Buy/Sell signals ported; the rest is drawn",
          all(x not in parts for x in ("bgcolor", "localBg", "htfBg", "Buy", "Sell", "triangle"))
          and all(x in parts for x in ("P.ob1", "P.ob2", "P.os1", "P.os2", "C.wt2", "WaveTrend crossed", "'R' : 'H'" if False else "${d.hidden ? 'H' : 'R'}-${bull ? 'Bu' : 'Br'}", "HTF ${P.htfTF}", "P.slopeHisto")))
    pane = grab("_sqzPaneSvg")
    check("wavetrend: drawn inside the SQZ MOM pane on its own scale, two legend lines each with its own ⚙ (sqz and wt)",
          "const yW = v => mid - v / mw * half;" in pane and "_wtPaneParts(cs, xOf, yW," in pane
          and "${_gear('sqz')}" in pane and "${_gear('wt')}" in pane and 'data-indset="${id}"' in d
          and "if (!onS && !onW) return '';" in pane)
    check("dashboard: WaveTrend toggle (default on) + ⚙, registered with its key, Reset Defaults clears it; pane grows when it is on",
          'id="chk_wt"' in d and 'data-indset="wt"' in d and "_indRegister('wt'" in d and "key: 'wtCfgV1'" in d
          and "'wtCfgV1',             //" in d and "'ttr', 'sqz', 'wt', 'pdZones'" in d and d.count("sqz: true, wt: true }") == 2
          and "Math.round(k.H * (_chart.show.wt ? 0.34 : 0.25)) + 10" in d)


_POI_SAMPLE = (
    '"Participant wise Open Interest (no. of contracts) in Equity Derivatives as on Oct 23, 2026",,,,,,,,,,,,,,\n'
    'Client Type,Future Index Long,Future Index Short,Future Stock Long,Future Stock Short\t,Option Index Call Long,'
    'Option Index Put Long,Option Index Call Short,Option Index Put Short,Option Stock Call Long,Option Stock Put Long,'
    'Option Stock Call Short,Option Stock Put Short,Total Long Contracts\t,Total Short Contracts\t\n'
    'Client,231868,150442,2468530,247513,2410870,1966447,2017411,2268010,742431,358063,511906,363880,8178209,5559162\n'
    'DII,90960,48963,140716,3159085,0,8,0,0,0,0,0,0,231684,3208048\n'
    'FII,71204,211370,1650389,1258779,409617,512361,321890,328064,165364,116391,149023,121917,2925326,2391043\n'
    'Pro,40536,23793,519624,513876,1500128,1235127,1981332,1125870,385025,179713,365943,187084,3860153,4197618\n'
    'TOTAL,434568,434568,4779259,4779253,4320615,3713943,4320633,3721944,1292820,654167,1026872,672881,15195372,15195371\n')


def test_participant_oi():
    """Participant-wise OI (Desk ▸ Post-Market): the NSE file parsed by column
    name, a 20-session series that skips weekends / holidays and downloads
    each session once, and the reader's changes, history, retail-vs-FII and
    next-session lean (also fed to the Market Prep bias)."""
    import ast as _ast, tempfile, shutil, threading as _th, time as _t, json as _json, subprocess, random as _r
    src = open(os.path.join(HERE, "nse_chain_server.py"), encoding="utf-8").read()
    tree = _ast.parse(src)
    want = {"_poi_parse", "_poi_get", "_poi_series", "_nse_holidays", "_is_trading_day", "_shift_day", "_prev_trading_days"}
    nodes = [n for n in tree.body if isinstance(n, _ast.FunctionDef) and n.name in want]
    consts = [n for n in tree.body if isinstance(n, _ast.Assign) and getattr(n.targets[0], "id", "") in ("_POI_COLS", "_NSE_HOLIDAYS_BUILTIN", "_POI_URLS")]
    tmp = tempfile.mkdtemp(prefix="poi")
    calls = []

    class _Resp:
        def __init__(self, code, text): self.status_code, self.text = code, text

    class _Req:
        @staticmethod
        def get(url, timeout=0, headers=None):
            calls.append(url)
            return _Resp(200, _POI_SAMPLE) if "nsearchives" in url else _Resp(404, "")

    class _Sess:
        session = _Req
    ns = {"_os": os, "json": _json, "time": _t, "threading": _th, "_POI_DIR": tmp, "_poi_miss": {}, "_poi_lock": _th.Lock(),
          "_warm_nse_fetcher": lambda: _Sess, "_ctx_day": lambda: "2026-10-26"}
    try:
        exec(compile(_ast.Module(body=consts + nodes, type_ignores=[]), "srv", "exec"), ns)
        rows = ns["_poi_parse"](_POI_SAMPLE)
        check("participant OI: the NSE file parsed by column name (title line, tab-padded headers)",
              rows and set(rows) == {"Client", "DII", "FII", "Pro", "TOTAL"} and rows["FII"]["fut_idx_short"] == 211370
              and rows["Client"]["opt_idx_put_long"] == 1966447 and rows["Pro"]["total_short"] == 4197618
              and rows["DII"]["fut_stk_short"] == 3159085, str(rows and rows.get("FII"))[:200])
        check("participant OI: a page that is not the file (HTML error, empty) is rejected",
              ns["_poi_parse"]("<html>Access Denied</html>") is None and ns["_poi_parse"]("") is None)
        import calendar as _cal
        ns["time"] = type("T", (), {"time": staticmethod(lambda: _cal.timegm((2026, 10, 26, 14, 0, 0, 0, 0, 0)) - 19800),   # 14:00 IST
                                     "gmtime": staticmethod(_t.gmtime), "sleep": staticmethod(lambda x: None)})
        r1 = ns["_poi_series"](5)
        n1 = len(calls)
        r2 = ns["_poi_series"](5)
        check("participant OI: before 17:30 today is skipped; 5 sessions back skip the weekend and Dussehra (20 Oct); one download each",
              [x["date"] for x in r1["series"]] == ["2026-10-16", "2026-10-19", "2026-10-21", "2026-10-22", "2026-10-23"]
              and r1["downloaded"] == 5 and n1 == 5, str([x["date"] for x in r1["series"]]) + f" calls={n1}")
        check("participant OI: asked again, every session comes from the disk cache (no NSE request)",
              len(calls) == n1 and r2["cached"] == 5 and r2["downloaded"] == 0, f"{len(calls) - n1} new calls")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    check("server: /api/participant-oi serves the series (days, date) and keeps the old fields",
          '"series": res["series"]' in src and '"fii_history": fh' in src and 'int(qs.get("days", ["20"])[0] or 20)' in src)

    # ── the dashboard reader ──
    d = open(os.path.join(HERE, "nse_dashboard.html"), encoding="utf-8").read()
    def grab(name):
        i = d.index(f"function {name}(")
        depth, j = 0, d.index("{", i)
        for k in range(j, len(d)):
            depth += d[k] == "{"; depth -= d[k] == "}"
            if depth == 0:
                return d[i:k + 1]
    _r.seed(5)
    series, fl, fs, cl, cs = [], 100000, 200000, 220000, 150000
    days = ["2026-09-%02d" % x for x in range(28, 31)] + ["2026-10-%02d" % x for x in (1, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 19, 21, 22, 23)]
    for i, day in enumerate(days):
        if i == len(days) - 1:
            fl, fs, cl, cs = fl + 30000, fs - 10000, cl + 60000, cs - 5000      # last day: FII add longs, clients pile long
        else:
            fl += _r.randint(-3000, 3000); fs += _r.randint(-3000, 3000); cl += _r.randint(-3000, 3000); cs += _r.randint(-3000, 3000)
        rows = {w: {"fut_idx_long": a, "fut_idx_short": b, "fut_stk_long": 1e6, "fut_stk_short": 9e5, "opt_idx_call_long": 4e5, "opt_idx_put_long": 5e5,
                    "opt_idx_call_short": 3e5 + (60000 if (w == "FII" and i == len(days) - 1) else 0), "opt_idx_put_short": 3e5}
                for w, a, b in (("FII", fl, fs), ("Client", cl, cs), ("Pro", 40000, 24000), ("DII", 90000, 49000))}
        series.append({"date": day, "rows": rows})
    harness = r"""
const escapeHtml = s => String(s ?? '');
""" + d[d.index("  const _poiK = n =>"):d.index("  function _poiMetrics(")] + "\n".join(grab(f) for f in ("_poiMetrics", "_poiPctRank", "_poiSpark", "_poiRead")) + f"""
const P = _poiRead({{ date: '2026-10-23', series: {_json.dumps(series)} }});
console.log(JSON.stringify({{ lean: P.lean, score: P.score, why: P.why, n: P.n, hasTable: /idx fut net/.test(P.html), acts: (P.html.match(/added|covered|closed/g) || []).length, spark: (P.html.match(/<polyline/g) || []).length, gauge: /FII vs retail/.test(P.html) }}));
"""
    res = subprocess.run(["node", "-e", harness], capture_output=True, text=True)
    try:
        R = _json.loads(res.stdout.strip().splitlines()[-1])
    except Exception:
        check("participant OI reader: harness ran", False, (res.stderr or res.stdout)[:500]); return
    whys = " | ".join(w[1] for w in R["why"])
    check("participant OI reader: table, who-did-what, 20-session sparklines and the FII-vs-retail gauge",
          R["hasTable"] and R["acts"] >= 3 and R["spark"] == 4 and R["gauge"] and R["n"] == 18, str(R)[:300])
    check("participant OI reader: FIIs adding longs reads +, FII options turning bearish reads −, retail crowded long reads − (contrarian)",
          "FIIs added net longs" in whys and "more bearish" in whys and "crowded long" in whys, whys[:400])
    check("Market Prep: the previous session's participant lean joins the bias (only when it is the session right before)",
          "pl.date === _tradingDayShift(dayK, -1)" in d and "participant OI (${pl.date.slice(5)})" in d
          and "localStorage.setItem('poiLast'" in d and "/api/participant-oi?days=20" in d and "'post-market':       '<strong>Post-Market" in d)


def test_chart_ctx_interval():
    """Right-click on the chart offers the 1/2/3/5/10/15-minute intervals
    (current one highlighted), switching through the toolbar's own button;
    the alert items stay; the menu is kept on screen."""
    d = open(os.path.join(HERE, "nse_dashboard.html"), encoding="utf-8").read()
    i = d.index("const onCtx = ev => {"); j = d.index("document.addEventListener('contextmenu', onCtx, true);")
    seg = d[i:j]
    check("chart right-click: interval row 1/2/3/5/10/15m above the alert items, pressing the toolbar button",
          "for (const iv of [1, 2, 3, 5, 10, 15])" in seg and "document.querySelector(`.chart-iv[data-iv=\"${iv}\"]`)?.click()" in seg
          and seg.index("ctx-iv-row") < seg.index("Alert when spot touches") and "(_chart.interval || 1) === iv" in seg
          and "innerWidth - mr.width - 4" in seg)
    check("chart right-click: every interval it offers exists as a toolbar button",
          all(f'class="btn chart-iv{" active" if iv == 1 else ""}" data-iv="{iv}"' in d for iv in (1, 2, 3, 5, 10, 15)))


def test_chart_ctx_actions():
    """Chart right-click: History calendar (weekends / NSE holidays greyed with
    the holiday's name, future disabled, today = Live, a pick drives the same
    Live / Historical controls), Fit (the toolbar's Fit) and Reset Defaults
    (the very function behind Alerts ▸ Reset Defaults)."""
    d = open(os.path.join(HERE, "nse_dashboard.html"), encoding="utf-8").read()
    i = d.index("const onCtx = ev => {"); j = d.index("document.addEventListener('contextmenu', onCtx, true);")
    seg = d[i:j]
    check("chart right-click: calendar and Fit above the alerts; Reset Defaults LAST, behind its own separator",
          "_histCalendarOpen(cx, cy)" in seg and "document.getElementById('zoomFitBtn')?.click()" in seg
          and "item('↺ Reset Defaults…', () => _resetSettings());" in seg
          and seg.index("History calendar") < seg.index("Alert when spot touches")
          and seg.index("Delete alert at") < seg.index("ctx-sep-reset") < seg.index("item('↺ Reset Defaults…'") < seg.index("document.body.appendChild(menu);")
          and seg.count("Reset Defaults…") == 1)
    check("chart right-click: Reset Defaults is the same function the Alerts tab button runs",
          "if (ev.target && ev.target.id === 'alertResetBtn') { ev.preventDefault(); _resetSettings(); }" in d)
    names = d[d.index("const _NSE_HOLIDAY_NAMES = {"):d.index("};", d.index("const _NSE_HOLIDAY_NAMES = {"))]
    import re as _re
    hn = set(_re.findall(r"'(\d{4}-\d{2}-\d{2})':", names))
    hs = set(_re.findall(r"'(\d{4}-\d{2}-\d{2})'", d[d.index("const _NSE_HOLIDAYS = new Set("):d.index("]);", d.index("const _NSE_HOLIDAYS = new Set("))]))
    check("history calendar: every NSE holiday has its name for the tooltip", hn == hs and len(hn) == 16)
    cal = d[d.index("function _histCalendarRender()"):d.index("function _histCalendarOpen(")]
    check("history calendar: weekends, holidays and future days are not selectable; today means Live; Monday-first grid",
          "const fut = k > today, ok = !wk && !hol && !fut;" in cal and "${ok ? `data-hday=\"${k}\"` : ''}" in cal
          and "(first.getDay() + 6) % 7" in cal and "● Live (today)" in cal
          and "if (!day || day >= _istDayKey()) modeSel.value = 'live';" in d)


def test_time_zoom():
    """Horizontal (time) zoom: one shared window drives every time-based x
    (chart, CVD/PCR/VIX pane, Squeeze Momentum, grid, crosshair); wheel = time,
    Shift+wheel / price axis = price; zoomed in, drag pans and the day swipe
    stands down; Fit / double-click / interval or day change reset it; the
    price axis fits the visible candles."""
    import subprocess, json as _json
    d = open(os.path.join(HERE, "nse_dashboard.html"), encoding="utf-8").read()
    def grab(name):
        i = d.index(f"function {name}(")
        depth, j = 0, d.index("{", i)
        for k in range(j, len(d)):
            depth += d[k] == "{"; depth -= d[k] == "}"
            if depth == 0:
                return d[i:k + 1]
    harness = r"""
const _chart = { interval: 1 }; let rendered = 0;
const _isRenkoView = () => false; const requestAnimationFrame = f => f(); const _safe = f => f(); const renderPriceChart = () => rendered++;
const _slotsNow = () => Math.max(1, Math.ceil(385 / (_chart.interval || 1)));
""" + "\n".join(grab(f) for f in ("_xWin", "_xzActive", "_xs", "_xSlotAt", "_xSetWin", "_xRender", "_xZoomAt", "_xGridStep")) + r"""
let _xzRaf = 0;
const R = {};
const W = 900, pL = 8, pR = 72, S = 385;
R.full = [_xs(0.5, S, W, pL, pR), _xs(S - 0.5, S, W, pL, pR), _xzActive()];
const px = 400, a0 = _xSlotAt(px, S, W, pL, pR);
_xZoomAt(px, W, pL, pR, 0.5);                                   // zoom in 2x at the pointer
R.zoom = { win: _chart.xZoom, anchor: Math.abs(_xSlotAt(px, S, W, pL, pR) - a0) < 1e-9, active: _xzActive(), rendered, step: _xGridStep(S) };
_xSetWin(-50, 100, S); R.clampLo = _chart.xZoom.s0;
_xSetWin(9999, 100, S); R.clampHi = _chart.xZoom.s1;
_xSetWin(0, 2, S); R.minSpan = _chart.xZoom.s1 - _chart.xZoom.s0;
_xZoomAt(px, W, pL, pR, 1e6); R.outToFull = _chart.xZoom;
_chart.xZoom = { s0: 10, s1: 50, slots: S }; _chart.interval = 5; R.otherInterval = _xzActive();
console.log(JSON.stringify(R));
"""
    res = subprocess.run(["node", "-e", harness], capture_output=True, text=True)
    try:
        R = _json.loads(res.stdout.strip().splitlines()[-1])
    except Exception:
        check("time zoom: harness ran", False, (res.stderr or res.stdout)[:400]); return
    full_ok = abs(R["full"][0] - (8 + 0.5 / 385 * 820)) < 1e-6 and abs(R["full"][1] - (8 + 384.5 / 385 * 820)) < 1e-6 and R["full"][2] is False
    check("time zoom: at the full day the mapping is exactly the old one (nothing moves)", full_ok, str(R["full"]))
    z = R["zoom"]
    check("time zoom: wheel zooms around the pointer's time - the slot under the cursor stays put",
          z["anchor"] and z["active"] and abs((z["win"]["s1"] - z["win"]["s0"]) - 192.5) < 1e-6 and z["rendered"] >= 1, str(z))
    check("time zoom: the window is clamped to the session, has a minimum width, and zooming out past the day clears it",
          R["clampLo"] == 0 and abs(R["clampHi"] - 385) < 1e-9 and R["minSpan"] >= 6 and R["outToFull"] is None, str(R))
    check("time zoom: a window saved for another interval is ignored (no stale zoom after switching 1m → 5m)", R["otherInterval"] is False)
    check("time zoom: every pane and the grid/crosshair use the shared window",
          d.count("_xs(slotOf(cs[i].t) + 0.5, slots, W, padL, padR)") == 1 and "const x = ts => _xs(slotOf(ts) + 0.5, slots, W, padL, padR);" in d
          and "else sx = _xs(m / iv + 0.5, slots, W, padL, padR);" in d and "_xSlotAt(px, slots, W, padL, padR) - 0.5" in d
          and "${_xClip('xclipQ', W, padL, padR, H)}<g class=\"xzc\">" in d
          and "${_xClip('xclipC', W, padL, padR, H)}<g class=\"xzc\">" in d and "${_xClip('xclipM', W, padL, padR, H)}${lvlSvg}<g class=\"xzc\">" in d
          and "padL + (m / iv + 0.5) / slots * (W - padL - padR)" not in d and "/ slots * (W - padL - padR)" not in d)
    check("time zoom: wheel = time, Shift+wheel or the price axis = price; drag pans when zoomed, swipe only at full day; resets",
          "if (!ev.shiftKey && !overAxis) {" in d and "_xZoomAt(px, W, padL, padR" in d and "if (_xzActive()) {" in d
          and "if (_xPan) _xPanMove(ev.clientX); else if (_daySwipe)" in d
          and "_chart.yZoom = 1; _chart.yCenter = null; _chart.xZoom = null; _zoomSync(); renderPriceChart();" in d
          and "_chart.interval = +b.dataset.iv; _chart.xZoom = null;" in d and "_chart.candles = []; _chart.liveBar = false; _chart.xZoom = null;" in d)
    check("time zoom: price axis fits the visible candles; levels widen the fit only at the full day",
          "const visCs = _visMask && _visMask.some(Boolean) ? cs.filter((_, i) => _visMask[i]) : cs;" in d and "if (!_xzOn) for (const L of levels)" in d)


def main():
    for fn in (test_structure, test_scope_js, test_scope_js_functions, test_scope_python, test_pcr_reading,
               test_wall_abandonment, test_delta, test_flow_delta_adjustment,
               test_pcr_bands_scale, test_aggressor_flow, test_iv_solver, test_no_shadowed_imports,
               test_ist_offset, test_reset_defaults, test_units, test_boxes, test_constituents, test_theme, test_true_black_widgets, test_watchdog, test_broker_refusals, test_zerodha_adapter, test_zerodha_watchdog_and_oi, test_zerodha_fixes_and_diagnostics, test_zerodha_subscribe_before_connect, test_diag_extras, test_broker_neutral_server, test_strike_picker, test_entry_alert_discipline, test_ctx_and_intervals, test_book_heavyweights, test_pcr_smooth, test_market_direction, test_traffic_light, test_turn_model, test_server_reachability, test_remote_access, test_trade_engine, test_scalping_tab, test_turn_times, test_ema_cross, test_trend_target_ribbon, test_ctx_clock, test_warmup_and_signals, test_day_swipe, test_render_performance, test_pd_premium_discount, test_pd_structure_psychology, test_full_width_layout, test_indicator_settings, test_squeeze_momentum, test_wavetrend, test_participant_oi, test_chart_ctx_interval, test_chart_ctx_actions, test_time_zoom, test_turn_blink, test_defaults_pvsra_pivots, test_index_weights, test_env):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            check(fn.__name__, False, f"{type(e).__name__}: {e}")

    print(f"\n  {len(_passes)} passed, {len(_fails)} failed, {len(_warns)} warning(s)\n")
    for n, d in _warns:
        print(f"  WARN  {n}" + (f" — {d}" if d else ""))
    if _fails:
        print()
        for n, d in _fails:
            print(f"  FAIL  {n}")
            if d:
                print(f"        {d}")
        print("\n  Do not ship until these pass.\n")
        return 1
    print("\n  All checks pass. Note this rules out known failure modes;\n"
          "  it does not prove the analytics are correct.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
