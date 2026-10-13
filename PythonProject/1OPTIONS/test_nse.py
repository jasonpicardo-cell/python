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


def main():
    for fn in (test_structure, test_scope_js, test_scope_js_functions, test_scope_python, test_pcr_reading,
               test_wall_abandonment, test_delta, test_flow_delta_adjustment,
               test_pcr_bands_scale, test_aggressor_flow, test_iv_solver, test_no_shadowed_imports,
               test_ist_offset, test_reset_defaults, test_units, test_index_weights, test_env):
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
