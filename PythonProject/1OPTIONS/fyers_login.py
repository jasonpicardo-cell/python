#!/usr/bin/env python3
"""
Fyers daily login helper.

Fyers access tokens expire every day and the login flow is interactive - you
authorise in a browser and paste back an auth code. There is no way to automate
that away without storing your password, which is not a trade worth making, so
this makes the manual step as short as possible instead.

Run it once each morning:   python3 fyers_login.py
It writes .fyers_token next to the server, which the adapter reads.
"""
import json, os, sys, time, webbrowser
from urllib.parse import urlparse, parse_qs

try:
    from fyers_apiv3 import fyersModel
except ImportError:
    sys.exit("fyers-apiv3 not installed.  pip install fyers-apiv3")

HERE = os.path.dirname(os.path.abspath(__file__))


# ".env" is the convention but a plain "env" is common, and silently reading
# nothing because of a filename is a miserable thing to debug. Accept both.
ENV_NAMES = ("env", ".env", ".env.local", "env.txt", ".env.txt")


def _env_file():
    override = os.environ.get("NSE_ENV_FILE")
    if override and os.path.exists(override):
        return override
    for name in ENV_NAMES:
        p = os.path.join(HERE, name)
        if os.path.exists(p):
            return p
    return ""


def _env(k, d=""):
    v = os.environ.get(k)
    if v:
        return v.strip()
    envf = _env_file()
    if envf:
        for line in open(envf):
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            if key.strip() == k:
                return val.strip().strip('"').strip("'")
    return d


def main():
    cid = _env("FYERS_CLIENT_ID")
    secret = _env("FYERS_SECRET_KEY")
    redirect = _env("FYERS_REDIRECT_URI", "https://127.0.0.1/")
    envf = _env_file()
    if envf:
        print(f"   (reading settings from {os.path.basename(envf)})")
    if not cid or not secret:
        target = os.path.basename(envf) if envf else "env"
        sys.exit(f"Set FYERS_CLIENT_ID and FYERS_SECRET_KEY in {target} first "
                 f"(looked in: {', '.join(ENV_NAMES)}).\n"
                 f"Create an app at https://myapi.fyers.in to get them.")

    session = fyersModel.SessionModel(
        client_id=cid, secret_key=secret, redirect_uri=redirect,
        response_type="code", grant_type="authorization_code", state="1opt")
    url = session.generate_authcode()
    print("\n1. Authorise in the browser that just opened (or paste this URL):\n")
    print("   " + url + "\n")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    print("2. After logging in you land on your redirect URL. Copy the WHOLE")
    print("   address bar and paste it here (it contains the auth code).\n")
    pasted = input("   URL or auth code: ").strip()

    code = pasted
    if pasted.startswith("http"):
        q = parse_qs(urlparse(pasted).query)
        code = (q.get("auth_code") or q.get("code") or [""])[0]
    if not code:
        sys.exit("No auth code found in that input.")

    session.set_token(code)
    resp = session.generate_token()
    if not isinstance(resp, dict) or "access_token" not in resp:
        sys.exit(f"Token generation failed: {resp}")

    token = resp["access_token"]

    # ── 1. the dashboard's own token file (JSON, with a timestamp) ──
    # The timestamp is what the server and adapter use to decide whether a
    # login is needed, so this file has to stay structured.
    path = os.path.join(HERE, ".fyers_token")
    with open(path, "w") as f:
        json.dump({"access_token": token, "ts": time.time(),
                   "client_id": cid}, f)
    try:
        os.chmod(path, 0o600)      # it is a credential; do not leave it readable
    except Exception:
        pass
    print(f"\n   Saved to {path} (valid until tomorrow).")

    # ── 2. plain-text copy for the historical-data downloader ──
    # NSE_SYNC reads a bare token, not JSON - writing the whole structure
    # there would hand it a string it cannot parse. Same secret, two formats,
    # written together so they can never drift out of step: a stale copy in
    # one place is worse than no copy, because it fails in a way that looks
    # like a broken API rather than an expired token.
    sync_dir = _env("FYERS_SYNC_DIR") or os.path.join(HERE, "..", "NSE_SYNC")
    sync_path = os.path.abspath(os.path.join(sync_dir, "fyers_token.txt"))
    try:
        os.makedirs(os.path.dirname(sync_path), exist_ok=True)
        with open(sync_path, "w") as f:
            f.write(token)          # the raw token only - no JSON, no newline
        try:
            os.chmod(sync_path, 0o600)
        except Exception:
            pass
        print(f"   Saved to {sync_path} (plain token for NSE_SYNC).")
    except Exception as e:
        # Not fatal: the dashboard works without it, and saying so plainly is
        # better than failing a login that otherwise succeeded.
        print(f"   ! Could not write {sync_path}: {e}")
        print(f"     The dashboard is fine; only the historical downloader "
              f"needs this copy.")

    print("   Restart the server, or it will pick this up on its next connect.\n")


if __name__ == "__main__":
    main()
