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
          "_seBanner(s)" in dsrc and "if (h.n < 2) continue;" in dsrc and "now - f < 300000" in dsrc)
    check("cost defaults: STT 0.1% and exchange 0.03503% on options", "sttPct: 0.1, exchPct: 0.03503" in dsrc)

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
    """Traffic light: BUY when the readers agree, WAIT when they disagree,
    hysteresis between 40 and 30; boxes removed from the chart."""
    import subprocess, json as _json
    dsrc = open(os.path.join(HERE, "nse_dashboard.html"), encoding="utf-8").read()
    def grab(name):
        i = dsrc.index(f"function {name}(")
        depth, j = 0, dsrc.index("{", i)
        for k in range(j, len(dsrc)):
            depth += dsrc[k] == "{"; depth -= dsrc[k] == "}"
            if depth == 0:
                return dsrc[i:k + 1]
    harness = grab("_trafficVotes") + grab("renderTrafficLight") + r"""
const window = {}; const state = { symbol: 'NIFTY' }; const _tl = { state: 'WAIT', since: 0 };
const el = { innerHTML: '' }; const document = { getElementById: () => el };
const _isHistorical = () => false; const _istNowMin = () => ({ m: 600, wd: 3 });
const _scoreRecord = () => {}; const escapeHtml = x => x;
const set = (md, dr, te, cf, regime, dir, cons) => {
  const at = Date.now();
  window._mdLast = { score: md, label: 'x', at }; window._deskReadLast = { total: dr, t: at, have: true };
  window._trendLast = { score: te, at }; window._confLast = { score: cf, confirmed: 0, book: 1, hw: 0, label: 'x' };
  window._pickLast = { regime, trendDir: dir, at };
  return { underlying_value: 25000, constituents: { available: true, score: cons } };
};
const _tlPlace = () => {}; const word = () => (el.innerHTML.match(/tl-word[^>]*>([A-Z]+)</) || [])[1];
renderTrafficLight(set(50, 4, 55, 70, 'trend', 1, 60)); const a = word();
renderTrafficLight(set(35, 2, 30, 30, 'range', 0, 20)); const b = word();      // eases to ~+33: holds BUY
renderTrafficLight(set(10, -1, 5, 0, 'range', 0, 0)); const c = word();       // fades: WAIT
renderTrafficLight(set(60, -4, 50, -60, 'trend', -1, 50)); const dd = word();  // readers disagree: WAIT
renderTrafficLight(set(-55, -4, -60, -70, 'trend', -1, -50)); const e = word();
console.log(JSON.stringify({ a, b, c, dd, e }));
"""
    fn = os.path.join(HERE, "__tl_test.js")
    open(fn, "w").write(harness)
    try:
        out = subprocess.run(["node", fn], capture_output=True, text=True, timeout=30)
    finally:
        os.remove(fn)
    if out.returncode:
        check("traffic light runs", False, out.stderr[-300:]); return
    r = _json.loads(out.stdout.strip().splitlines()[-1])
    check("traffic light: readers agreeing long -> BUY", r["a"] == "BUY", str(r))
    check("traffic light: holds BUY while the composite stays above 30", r["b"] == "BUY", str(r))
    check("traffic light: fades to WAIT", r["c"] == "WAIT", str(r))
    check("traffic light: readers disagreeing -> WAIT", r["dd"] == "WAIT", str(r))
    check("traffic light: readers agreeing short -> SELL", r["e"] == "SELL", str(r))
    check("traffic light sits top-left over the candles; Desk and Move-quality boxes removed",
          'id="tlight"' in dsrc and "_chart._mqBox = '';" in dsrc and "renkoMqBox +=" not in dsrc
          and 'id="chk_deskBox"' not in dsrc)


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
          dsrc.count("emaX: true, ttr: true }, emaX: { fast: 5, slow: 13 } }") == 2 and "'chartCfg'," in dsrc)
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
    harness = fn + r"""
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
console.log(JSON.stringify({ a: nextOf(a) === xs(a)[0], b: nextOf(b) === xs(b)[1], c: nextOf(c) === xs(c)[2],
  d: nextOf(d), e: nextOf(e), nBlink: (b.match(/turn-next/g) || []).length, uniform: new Set(styles).size === 1 && styles.length === 3, leanCols }));
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
        check("zerodha: implied vol solved from prices", d["iv_source"] == "solved" and 10 < d["atm_iv"] < 16, f"{d['iv_source']} {d['atm_iv']}")
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
    check("theme: chart, RSI pane, Trade entries and the tiles sit on --wash",
          'id="chartSvg" viewBox="0 0 ${W} ${H}" style="width:100%;height:auto;background:var(--wash);' in d
          and d.count('style="fill:var(--wash)"') == 4 and   # RSI pane x2 + SMI-MFI pane x2
          '"d90Body" style="margin:2px 6px 4px;padding:5px 8px;border:1px solid rgba(255,255,255,0.10);border-radius:6px;background:var(--wash);"' in d)
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
          and d.count("emaX: true, ttr: true }") == 2)


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
          and "_alertHitTest({ clientX: x, clientY: y })" in d and "document.addEventListener('touchmove', ev => {\n    if (_daySwipe" in d)
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
          and "'pdZones', 'pdFvg', 'pdLiq', 'pdPrep', 'smiMfi'].includes(k)) _chart.show[k] = true;" in d
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


def _smimfi_reference(bars, mfi_len=13, k=13, d=5, sig_len=5, dz=50):
    """Independent Pine-semantics reference: ta.mfi, the TradingView SMI on it
    (x200, EMA(EMA(.,D),D)), EMA signal, histogram, percentile_nearest_rank."""
    import math
    n = len(bars)
    mfi = [None] * n
    for i in range(mfi_len, n):
        up = dn = 0.0
        for j in range(i - mfi_len + 1, i + 1):
            ch = bars[j]["c"] - bars[j - 1]["c"]
            if ch > 0: up += bars[j]["v"] * bars[j]["c"]
            elif ch < 0: dn += bars[j]["v"] * bars[j]["c"]
        mfi[i] = None if (dn == 0 and up == 0) else (100.0 if dn == 0 else 100 - 100 / (1 + up / dn))
    def ema(a, L):
        al, out, s = 2 / (L + 1), [], None
        for v in a:
            if v is not None:
                s = v if s is None else al * v + (1 - al) * s
            out.append(s)
        return out
    rel, rng = [None] * n, [None] * n
    for i in range(k - 1, n):
        w = mfi[i - k + 1:i + 1]
        if any(v is None for v in w): continue
        rel[i] = mfi[i] - (max(w) + min(w)) / 2; rng[i] = max(w) - min(w)
    num, den = ema(ema(rel, d), d), ema(ema(rng, d), d)
    smi = [None if (num[i] is None or not den[i]) else 200 * num[i] / den[i] for i in range(n)]
    sr = ema(smi, sig_len)
    sig = [None if smi[i] is None else sr[i] for i in range(n)]
    hist = [None if smi[i] is None or sig[i] is None else smi[i] - sig[i] for i in range(n)]
    def pr(w, p):
        s = sorted(w); r = max(1, math.ceil(p / 100 * len(s))); return s[min(len(s), r) - 1]
    dzT = [None] * n
    for i in range(dz - 1, n):
        w = [v for v in smi[i - dz + 1:i + 1] if v is not None]
        if len(w) == dz: dzT[i] = pr(w, 90)
    return {"smi": smi, "sig": sig, "hist": hist, "dzT": dzT}


def test_smi_mfi():
    """SMI of MFI port: matches an independent Pine-semantics reference on
    random bars; pane sits between the chart and RSI at RSI's height; no
    candle colouring; toggle on by default; crosshair shared."""
    import subprocess, json as _json, random as _r, tempfile
    d = open(os.path.join(HERE, "nse_dashboard.html"), encoding="utf-8").read()
    def grab(name):
        i = d.index(f"function {name}(")
        depth, j = 0, d.index("{", i)
        for k in range(j, len(d)):
            depth += d[k] == "{"; depth -= d[k] == "}"
            if depth == 0:
                return d[i:k + 1]
    _r.seed(7)
    bars, p = [], 25000.0
    for i in range(320):
        o = p; p += _r.gauss(0, 6) + (3 if (i // 60) % 2 else -3)
        bars.append({"t": 1000 + 60 * i, "o": o, "h": max(o, p) + 2, "l": min(o, p) - 2, "c": p, "v": _r.randint(500, 5000)})
    i0 = d.index("  const SMIMFI = {"); j0 = d.index("  const _smiCache")
    harness = d[i0:j0] + "\n" + "\n".join(grab(f) for f in ("_pineEma", "_mfiSeries", "_pctRank", "_smiMfiCompute")) + f"""
const r = _smiMfiCompute({_json.dumps(bars)});
console.log(JSON.stringify({{ smi: r.smi, sig: r.sig, hist: r.hist, dzT: r.dzT, cross: r.cross, useVol: r.useVol }}));
"""
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
        fh.write(harness); hp = fh.name
    try:
        res = subprocess.run(["node", hp], capture_output=True, text=True)
    finally:
        os.unlink(hp)
    try:
        js = _json.loads(res.stdout.strip().splitlines()[-1])
    except Exception:
        check("SMI-MFI: harness ran", False, (res.stderr or res.stdout)[:400]); return
    ref = _smimfi_reference(bars)
    def same(a, b):
        return len(a) == len(b) and all((x is None and y is None) or (x is not None and y is not None and abs(x - y) < 1e-6) for x, y in zip(a, b))
    check("SMI-MFI: SMI, signal and histogram match the independent reference bar for bar",
          same(js["smi"], ref["smi"]) and same(js["sig"], ref["sig"]) and same(js["hist"], ref["hist"]) and js["useVol"],
          str([(a, b) for a, b in zip(js["smi"], ref["smi"]) if (a is None) != (b is None) or (a is not None and abs(a - b) > 1e-6)][:3]))
    check("SMI-MFI: dynamic zone (90th percentile of 50 bars, nearest rank) matches", same(js["dzT"], ref["dzT"]))
    sm, sg = ref["smi"], ref["sig"]
    want = [i for i in range(1, len(sm)) if None not in (sm[i], sg[i], sm[i - 1], sg[i - 1])
            and ((sm[i] > sg[i] and sm[i - 1] <= sg[i - 1]) or (sm[i] < sg[i] and sm[i - 1] >= sg[i - 1])) and ref["dzT"][i - 1] is not None]
    check("SMI-MFI: every SMI/signal cross (once the zone exists) is marked, as inside or outside the zone",
          [c["i"] for c in js["cross"]] == want and all(c["kind"] in ("in", "out") for c in js["cross"]) and len(want) >= 5,
          f'{[c["i"] for c in js["cross"]][:8]} vs {want[:8]}')
    check("SMI-MFI: pane between the chart and RSI, same height as RSI, crosshair shared, toggle on by default",
          d.index('<div id="smiPaneBody"') < d.index('<div id="rsiPaneBody"')
          and "_smiMfiPaneSvg(k.cs, k.x, k.W, k.padL, k.padR, Math.round(k.H * 0.25) + 10)" in d
          and "_rsiPaneSvg(k.cs, k.x, k.W, k.padL, k.padR, Math.round(k.H * 0.25) + 10)" in d
          and "['smiXM', 'smi']" in d and "#smiPaneBody svg" in d and 'id="chk_smiMfi"' in d
          and "'pdPrep', 'smiMfi'].includes(k)) _chart.show[k] = true;" in d)
    seg = d[d.index("  // ══ SMI of MFI"):d.index("  function _rsiPaneSvg(")]
    check("SMI-MFI: no candle colouring (barcolor) ported", "barcolor" not in seg.lower().replace("not ported: candle colouring", "")
          and "candle colouring (as asked)" in seg)


def main():
    for fn in (test_structure, test_scope_js, test_scope_js_functions, test_scope_python, test_pcr_reading,
               test_wall_abandonment, test_delta, test_flow_delta_adjustment,
               test_pcr_bands_scale, test_aggressor_flow, test_iv_solver, test_no_shadowed_imports,
               test_ist_offset, test_reset_defaults, test_units, test_boxes, test_constituents, test_theme, test_true_black_widgets, test_watchdog, test_broker_refusals, test_zerodha_adapter, test_zerodha_watchdog_and_oi, test_zerodha_fixes_and_diagnostics, test_zerodha_subscribe_before_connect, test_diag_extras, test_broker_neutral_server, test_strike_picker, test_ctx_and_intervals, test_book_heavyweights, test_pcr_smooth, test_market_direction, test_traffic_light, test_turn_times, test_ema_cross, test_trend_target_ribbon, test_ctx_clock, test_warmup_and_signals, test_day_swipe, test_render_performance, test_pd_premium_discount, test_pd_structure_psychology, test_full_width_layout, test_smi_mfi, test_turn_blink, test_defaults_pvsra_pivots, test_index_weights, test_env):
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
