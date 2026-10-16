#!/usr/bin/env python3
"""
Daily Zerodha Kite Connect login for 1OPTIONS.

Kite access tokens expire at about 06:00 IST every morning, and the login is
interactive (Zerodha password + 2FA in a browser), so this runs once a day.
The server runs it for you at startup when DATA_SOURCE=zerodha and today's
token is missing; you can also run it yourself:

    python3 zerodha_login.py

Needs, in the env file next to the server:
    ZERODHA_API_KEY=...        from developers.kite.trade (your Kite Connect app)
    ZERODHA_API_SECRET=...
    ZERODHA_REDIRECT_URL=...   optional; the redirect URL set on that app

If the app's redirect URL points at this computer - for example
http://127.0.0.1:5010/kite - the token is captured automatically when the
login finishes. Otherwise the browser lands on whatever page the redirect
names, and you paste that page's full address here.

Writes .zerodha_token (access token + time issued) with owner-only
permissions, and a PLAIN-TEXT copy of the token - just the token string -
to ../NSE_SYNC/zerodha_access_token.txt for the tools that read a bare
token (as fyers_login.py does for Fyers). The folder is ZERODHA_SYNC_DIR,
else FYERS_SYNC_DIR, else ../NSE_SYNC, relative to this script.
The API secret and the token are never printed.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import webbrowser
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))


def _env() -> dict:
    vals = {}
    override = os.environ.get("NSE_ENV_FILE")
    paths = [override] if override else [os.path.join(HERE, n) for n in
                                          ("env", ".env", ".env.local", "env.txt", ".env.txt")]
    for p in paths:
        if p and os.path.exists(p):
            for line in open(p):
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    vals[k.strip()] = v.strip().strip('"').strip("'")
            break
    for k in ("ZERODHA_API_KEY", "ZERODHA_API_SECRET", "ZERODHA_REDIRECT_URL", "ZERODHA_TOKEN_FILE",
              "ZERODHA_SYNC_DIR", "FYERS_SYNC_DIR"):
        if os.environ.get(k):
            vals[k] = os.environ[k]
    return vals


def _write_sync_copy(env: dict, access: str) -> str:
    """The bare token, no JSON and no newline, for tools that read it as-is.
    Returns the path written, or '' (a failure here never fails the login)."""
    d = env.get("ZERODHA_SYNC_DIR") or env.get("FYERS_SYNC_DIR") or os.path.join("..", "NSE_SYNC")
    d = d if os.path.isabs(d) else os.path.normpath(os.path.join(HERE, d))
    path = os.path.join(d, "zerodha_access_token.txt")
    try:
        os.makedirs(d, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            f.write(access)
        try:
            os.chmod(tmp, 0o600)
        except Exception:
            pass
        os.replace(tmp, path)
        return path
    except Exception as e:  # noqa: BLE001
        print(f"[zerodha] could not write the plain-text token copy to {d}: {e}")
        return ""


def _request_token_from(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if "request_token=" in text:
        q = parse_qs(urlparse(text).query)
        return (q.get("request_token") or [""])[0].strip()
    return text if len(text) >= 16 and "/" not in text else ""


def _listen(redirect: str, timeout: float = 300.0) -> str:
    """Catch the redirect on 127.0.0.1 / localhost, if that is where it points."""
    u = urlparse(redirect or "")
    if u.scheme != "http" or u.hostname not in ("127.0.0.1", "localhost") or not u.port:
        return ""
    from http.server import BaseHTTPRequestHandler, HTTPServer
    got = {"tok": ""}

    class H(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            tok = _request_token_from(self.path if "?" in self.path else "")
            ok = bool(tok)
            if ok:
                got["tok"] = tok
            self.send_response(200 if ok else 400)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write((
                "<h3>Zerodha login received - you can close this tab.</h3>" if ok else
                "<h3>No request_token in this address - log in again.</h3>").encode())

        def log_message(self, *a):  # silence
            pass

    try:
        srv = HTTPServer((u.hostname, u.port), H)
    except OSError as e:
        print(f"[zerodha] cannot listen on {u.hostname}:{u.port} ({e}) - paste the address instead")
        return ""
    srv.timeout = 1.0
    end = time.time() + timeout
    print(f"[zerodha] waiting for the login to finish (listening on {u.hostname}:{u.port}) ...")
    while not got["tok"] and time.time() < end:
        srv.handle_request()
    srv.server_close()
    return got["tok"]


def main() -> int:
    try:
        from kiteconnect import KiteConnect
    except Exception as e:  # noqa: BLE001
        print(f"[zerodha] kiteconnect is not installed ({e}). Run:  pip install kiteconnect")
        return 2
    env = _env()
    api_key, secret = env.get("ZERODHA_API_KEY", ""), env.get("ZERODHA_API_SECRET", "")
    if not api_key or not secret:
        print("[zerodha] ZERODHA_API_KEY and ZERODHA_API_SECRET must be set in the env file.")
        print("          Create a Kite Connect app at https://developers.kite.trade to get them.")
        return 2
    kite = KiteConnect(api_key=api_key)
    url = kite.login_url()
    print("=" * 66)
    print("  Zerodha login (Kite tokens expire daily at about 06:00 IST)")
    print("  1. Log in to Kite in the browser window that opens")
    print(f"  2. If it does not open, visit:\n     {url}")
    print("=" * 66)
    try:
        webbrowser.open(url)
    except Exception:
        pass
    tok = ""
    redirect = env.get("ZERODHA_REDIRECT_URL", "")
    if redirect:
        holder = {}
        th = threading.Thread(target=lambda: holder.update(t=_listen(redirect)), daemon=True)
        th.start()
        th.join()
        tok = holder.get("t", "")
    while not tok:
        try:
            pasted = input("Paste the full address the browser landed on (or the request_token): ")
        except (EOFError, KeyboardInterrupt):
            print("\n[zerodha] login cancelled")
            return 1
        tok = _request_token_from(pasted)
        if not tok:
            print("[zerodha] no request_token found in that - copy the whole address bar")
    try:
        sess = kite.generate_session(tok, api_secret=secret)
    except Exception as e:  # noqa: BLE001
        print(f"[zerodha] session could not be created: {type(e).__name__}: {e}")
        print("          A request_token works once and for a few minutes - log in again.")
        return 1
    access = (sess or {}).get("access_token") or ""
    if not access:
        print("[zerodha] Kite returned no access token")
        return 1
    path = env.get("ZERODHA_TOKEN_FILE") or os.path.join(HERE, ".zerodha_token")
    data = {"access_token": access, "ts": time.time(),
            "user_id": (sess or {}).get("user_id"), "user_name": (sess or {}).get("user_name")}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    try:
        os.chmod(tmp, 0o600)
    except Exception:
        pass
    os.replace(tmp, path)
    sync = _write_sync_copy(env, access)
    if sync:
        print(f"[zerodha] plain-text token copy written to {sync}")
    who = data.get("user_name") or data.get("user_id") or ""
    print(f"[zerodha] logged in{f' as {who}' if who else ''} - token saved to {os.path.basename(path)} "
          f"(valid until about 06:00 IST tomorrow)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
