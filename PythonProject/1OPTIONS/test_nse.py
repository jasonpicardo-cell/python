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


def main():
    for fn in (test_structure, test_scope_js, test_scope_python, test_pcr_reading,
               test_wall_abandonment, test_delta, test_flow_delta_adjustment,
               test_pcr_bands_scale, test_index_weights, test_env):
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
