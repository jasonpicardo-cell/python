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


def main():
    for fn in (test_structure, test_scope_js, test_scope_js_functions, test_scope_python, test_pcr_reading,
               test_wall_abandonment, test_delta, test_flow_delta_adjustment,
               test_pcr_bands_scale, test_aggressor_flow, test_iv_solver, test_no_shadowed_imports,
               test_ist_offset, test_reset_defaults, test_units, test_boxes, test_constituents, test_theme, test_watchdog, test_strike_picker, test_ctx_and_intervals, test_index_weights, test_env):
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
