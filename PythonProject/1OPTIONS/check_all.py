#!/usr/bin/env python3
"""
check_all.py - the four pre-delivery checks in one command.

    python3 check_all.py                  run everything
    python3 check_all.py --update-baseline    accept the current HTML tag counts as the new normal

  1. every .py file compiles
  2. every inline <script> block in nse_dashboard.html passes `node --check`
  3. HTML tag balance did not change since the baseline (check_baseline.json).
     The dashboard has a known surplus of closing </div> tags that sit inside
     JS template strings, so the test is "same as before", not "zero".
  4. test_nse.py runs and its output ends with "All checks pass", with no
     Traceback, SyntaxError or FAIL line (grepping for FAIL alone once missed a
     syntax error inside a new test).

Exit code 0 only if all four pass. Run it from the folder that holds the server.
"""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
HTML = HERE / "nse_dashboard.html"
BASE = HERE / "check_baseline.json"
TAGS = ["div", "span", "table", "tr", "td", "th", "section", "aside", "main", "details", "summary", "select", "button", "style", "script"]

ok_all = True


def report(name, ok, detail=""):
    global ok_all
    ok_all = ok_all and ok
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"\n        {detail}" if detail and not ok else (f"  ({detail})" if detail else "")))


def check_python():
    bad = []
    files = sorted(p for p in HERE.glob("*.py"))
    for p in files:
        try:
            compile(p.read_text(encoding="utf-8"), str(p), "exec")
        except SyntaxError as e:
            bad.append(f"{p.name}:{e.lineno}: {e.msg}")
    report(f"py compile ({len(files)} files)", not bad, "; ".join(bad))


def inline_scripts(html):
    out = []
    for m in re.finditer(r"<script([^>]*)>(.*?)</script>", html, re.S | re.I):
        attrs, body = m.group(1), m.group(2)
        if "src=" in attrs.lower() or not body.strip():
            continue
        if re.search(r'type\s*=\s*["\'](?!text/javascript|module)', attrs, re.I):
            continue                                       # json / template blocks
        out.append(body)
    return out


def check_node(html):
    blocks = inline_scripts(html)
    errs = []
    try:
        subprocess.run(["node", "--version"], capture_output=True, check=True)
    except Exception:
        report("node --check on inline scripts", False, "node is not installed or not on PATH")
        return
    with tempfile.TemporaryDirectory() as d:
        for i, b in enumerate(blocks):
            f = Path(d) / f"block{i}.js"
            f.write_text(b, encoding="utf-8")
            r = subprocess.run(["node", "--check", str(f)], capture_output=True, text=True)
            if r.returncode:
                lines = [l.strip() for l in r.stderr.splitlines() if l.strip()]
                msg = next((l for l in lines if "Error" in l), lines[0] if lines else "syntax error")
                where = next((l for l in lines if re.search(r"block\d+\.js:\d+", l)), "")
                errs.append(f"block {i}: {msg} ({where.split('/')[-1]})")
    report(f"node --check ({len(blocks)} inline script blocks)", bool(blocks) and not errs, "; ".join(errs) or ("no inline scripts found" if not blocks else ""))


def tag_diffs(html):
    d = {}
    for t in TAGS:
        d[t] = len(re.findall(rf"</{t}\s*>", html, re.I)) - len(re.findall(rf"<{t}[\s>/]", html, re.I))
    return d


def check_tags(html, update):
    now = tag_diffs(html)
    if update or not BASE.exists():
        BASE.write_text(json.dumps(now, indent=1, sort_keys=True), encoding="utf-8")
        report("HTML tag balance", True, "baseline " + ("updated" if update else "created") + f": {BASE.name}")
        return
    old = json.loads(BASE.read_text(encoding="utf-8"))
    moved = {t: (old.get(t, 0), now[t]) for t in TAGS if now[t] != old.get(t, 0)}
    report("HTML tag balance vs baseline", not moved,
           ("closing minus opening changed (was, now): " + ", ".join(f"{t} {a}->{b}" for t, (a, b) in moved.items()) +
            "  | a real imbalance: fix it; an intended change in a template string: rerun with --update-baseline") if moved else "")


def check_tests():
    r = subprocess.run([sys.executable, str(HERE / "test_nse.py")], capture_output=True, text=True, cwd=HERE)
    out = (r.stdout or "") + (r.stderr or "")
    fails = [l.strip() for l in out.splitlines() if l.startswith("  FAIL")]
    bad = []
    if "All checks pass" not in out:
        bad.append('output does not contain "All checks pass"')
    if "Traceback" in out or "SyntaxError" in out:
        bad.append("Traceback or SyntaxError in the output")
    bad += fails[:5]
    report("test_nse.py", not bad and r.returncode == 0, " | ".join(bad) + ("" if r.returncode == 0 else f" | exit code {r.returncode}"))
    if bad:
        print("\n".join("        " + l for l in out.strip().splitlines()[-12:]))


def main():
    print("check_all: pre-delivery checks")
    check_python()
    if HTML.exists():
        html = HTML.read_text(encoding="utf-8")
        check_node(html)
        check_tags(html, "--update-baseline" in sys.argv)
    else:
        report("nse_dashboard.html present", False, str(HTML))
    check_tests()
    print("\n" + ("All four checks pass." if ok_all else "NOT READY: fix the failures above."))
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
