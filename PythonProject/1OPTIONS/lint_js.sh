#!/usr/bin/env bash
# Extract the dashboard's <script> blocks and run a REAL scope check on them.
#
# test_nse.py's JS heuristic cannot tell function scopes apart - it missed a
# genuine `spoken` vs `text` mix-up because a variable of that name existed in
# another function. This does the job properly: eslint's no-undef rule
# understands scope, and the whole class of bug (cw, _envbool, j, spoken)
# is exactly what it is designed to find.
#
#   ./lint_js.sh            checks nse_dashboard.html
set -u
DASH="${1:-nse_dashboard.html}"
command -v npx >/dev/null || { echo "npx not found — install Node.js to use this"; exit 1; }
python3 - "$DASH" <<'PY'
import re, sys
c = open(sys.argv[1], encoding="utf-8").read()
for i, b in enumerate(re.findall(r"<script>(.*?)</script>", c, re.S)):
    open(f"/tmp/_lint{i}.js", "w").write(b)
    print(f"/tmp/_lint{i}.js")
PY
for f in /tmp/_lint*.js; do
  echo "── $f"
  npx --yes eslint@8 --no-eslintrc \
      --parser-options ecmaVersion:2021 \
      --env browser,es2021 \
      --rule '{"no-undef":"error","no-dupe-keys":"error","no-unreachable":"error"}' \
      "$f" 2>&1 | grep -E "error|problem" | head -20
done
echo
echo "no-undef findings above are real scope errors, not heuristics."
