"""
nse_access.py - HTTPS and a login for reaching the dashboard from outside.

Why this exists
---------------
The server used to trust everything that could connect to it. That is fine
inside a home LAN and NOT fine once a router forwards a port to it: anyone on
the internet could read the broker-fed data (and, later, place orders).

Rules (all enforced before any route runs):

  * A client on loopback / the private LAN is treated exactly as before.
  * ANY other client (a public IP) - or anything that arrives through a
    reverse proxy (X-Forwarded-For and friends, which would otherwise make
    every visitor look like 127.0.0.1) - must be logged in.
  * No NSE_ACCESS_TOKEN set, or one shorter than 12 characters: outside
    clients are refused outright. It fails CLOSED.
  * Login is a form on the dashboard URL; the browser gets an HttpOnly,
    SameSite=Strict cookie (Secure once HTTPS is on) that holds an HMAC of the
    token, never the token. A script can send  Authorization: Bearer <token>.
  * 5 wrong tries from one address in 10 minutes locks that address out for
    10 minutes; every wrong try also costs a second.
  * NSE_AUTH_LAN=1 asks for the login on the LAN as well.

HTTPS: NSE_HTTPS=1 serves TLS 1.2+ with a certificate from tls/ (made for you
on first start - self-signed, valid for localhost, this machine's LAN address
and anything in NSE_PUBLIC_IP / NSE_PUBLIC_HOST), or with your own files via
NSE_TLS_CERT / NSE_TLS_KEY (for example a Let's Encrypt certificate).
"""
import hashlib
import hmac
import http.cookies
import ipaddress
import os
import socket
import ssl
import subprocess
import tempfile
import threading
import time
import urllib.parse

COOKIE = "nse_auth"
MIN_TOKEN = 12
LOCK_AFTER = 5
WINDOW_S = 600
LOCK_S = 600
_PROXY_HEADERS = ("X-Forwarded-For", "Forwarded", "X-Real-IP", "CF-Connecting-IP", "True-Client-IP")

_lock = threading.Lock()
_fails = {}          # ip -> [timestamps of recent wrong tries]
_logged = {}         # (ip, why) -> last time we printed it


def _env(name, default=""):
    return (os.environ.get(name) if os.environ.get(name) is not None else default).strip()


def token():
    return _env("NSE_ACCESS_TOKEN")


def token_ok():
    return len(token()) >= MIN_TOKEN


def session_value(tok=None):
    tok = token() if tok is None else tok
    return hmac.new(tok.encode(), b"nse-dashboard-session-v1", hashlib.sha256).hexdigest()


def client_ip(h):
    try:
        return str(h.client_address[0])
    except Exception:
        return "?"


# Spelled out rather than ipaddress.is_private, which is broader than "my home
# network": it also says yes to the documentation, benchmarking and reserved
# ranges, none of which should ever skip the login.
_LAN_NETS = [ipaddress.ip_network(n) for n in (
    "127.0.0.0/8", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "169.254.0.0/16",
    "::1/128", "fe80::/10", "fc00::/7")]


def is_private(ip):
    try:
        a = ipaddress.ip_address(str(ip).split("%")[0])
        if a.version == 6 and a.ipv4_mapped:
            a = a.ipv4_mapped
        return any(a.version == n.version and a in n for n in _LAN_NETS)
    except ValueError:
        return False


def is_remote(h):
    """True for anything that is not a plain LAN / loopback connection."""
    if any(h.headers.get(x) for x in _PROXY_HEADERS):
        return True
    return not is_private(client_ip(h))


def needs_login(h):
    if _env("NSE_AUTH_LAN").lower() in ("1", "true", "yes", "on"):
        return True
    return is_remote(h)


def _say(ip, why, msg):
    now = time.time()
    with _lock:
        if now - _logged.get((ip, why), 0) < 60:
            return
        _logged[(ip, why)] = now
        if len(_logged) > 500:
            _logged.clear()
    print(f"[access] {msg}")


def authed(h):
    """A valid cookie or bearer token on this request."""
    if not token_ok():
        return False
    want = session_value()
    try:
        jar = http.cookies.SimpleCookie(h.headers.get("Cookie") or "")
        if COOKIE in jar and hmac.compare_digest(jar[COOKIE].value, want):
            return True
    except Exception:
        pass
    auth = h.headers.get("Authorization") or ""
    if auth.lower().startswith("bearer ") and hmac.compare_digest(auth[7:].strip(), token()):
        return True
    return False


def _locked(ip):
    now = time.time()
    with _lock:
        v = [t for t in _fails.get(ip, []) if now - t < WINDOW_S]
        _fails[ip] = v
        if len(_fails) > 2000:
            for k in [k for k, x in _fails.items() if not x or now - x[-1] > WINDOW_S]:
                _fails.pop(k, None)
        return len(v) >= LOCK_AFTER and now - v[-1] < LOCK_S


def _fail(ip):
    with _lock:
        _fails.setdefault(ip, []).append(time.time())


_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Sign in</title>
<style>
:root{color-scheme:dark light}
body{margin:0;min-height:100vh;display:grid;place-items:center;background:#0e1116;color:#e6e8eb;font:15px system-ui,sans-serif}
form{width:min(340px,86vw);display:grid;gap:12px;padding:26px;border:1px solid #2a3040;border-radius:10px;background:#151a22}
h1{margin:0 0 4px;font-size:17px;font-weight:600}
input,button{font:inherit;padding:10px 12px;border-radius:6px;border:1px solid #2a3040;background:#0e1116;color:inherit}
button{background:#1fd17a;color:#04140c;border:0;font-weight:600;cursor:pointer}
.err{color:#ff6b79;font-size:13px;min-height:1em;margin:0}
@media (prefers-color-scheme:light){body{background:#f4f6f9;color:#1b2430}form{background:#fff;border-color:#d5dbe5}input{background:#fff;border-color:#c5ccd8}}
</style></head><body>
<form method="post" action="/api/login" autocomplete="on">
<h1>NSE dashboard</h1>
<input type="password" name="token" placeholder="Access token" autofocus required autocomplete="current-password">
<p class="err">__MSG__</p>
<button type="submit">Sign in</button>
</form></body></html>"""


def _send(h, status, body, ctype="text/html; charset=utf-8", extra=()):
    data = body.encode("utf-8") if isinstance(body, str) else body
    try:
        h.send_response(status)
        h.send_header("Content-Type", ctype)
        h.send_header("Content-Length", str(len(data)))
        h.send_header("Cache-Control", "no-store")
        h.send_header("X-Content-Type-Options", "nosniff")
        for k, v in extra:
            h.send_header(k, v)
        h.end_headers()
        h.wfile.write(data)
    except Exception:
        pass


def _login_page(h, status=401, msg=""):
    _send(h, status, _PAGE.replace("__MSG__", msg))


def guard(h, tls=False):
    """Call first in every do_GET / do_POST / do_OPTIONS. True = carry on;
    False = a response has been written and the request is finished."""
    if not needs_login(h):
        return True
    path = urllib.parse.urlparse(h.path).path
    ip = client_ip(h)
    if not token_ok():
        _say(ip, "off", f"refused {ip} {path}: outside access is off - set NSE_ACCESS_TOKEN (12+ characters) in env")
        _send(h, 403, "Outside access is switched off on this server.", "text/plain; charset=utf-8")
        return False
    if path == "/api/login" and h.command == "POST":
        if _locked(ip):
            _say(ip, "lock", f"{ip} locked out after repeated wrong tokens")
            _send(h, 429, "Too many wrong tries. Wait ten minutes.", "text/plain; charset=utf-8")
            return False
        try:
            n = max(0, min(int(h.headers.get("Content-Length") or 0), 2048))
            raw = h.rfile.read(n).decode("utf-8", "replace")
        except Exception:
            raw = ""
        given = urllib.parse.parse_qs(raw).get("token", [""])[0]
        if hmac.compare_digest(given.encode(), token().encode()):
            with _lock:
                _fails.pop(ip, None)
            print(f"[access] sign-in from {ip}")
            ck = f"{COOKIE}={session_value()}; Path=/; HttpOnly; SameSite=Strict; Max-Age=2592000" + ("; Secure" if tls else "")
            _send(h, 303, "", "text/plain; charset=utf-8", (("Location", "/"), ("Set-Cookie", ck)))
            return False
        _fail(ip)
        _say(ip, "bad", f"wrong token from {ip}")
        time.sleep(1.0)
        _login_page(h, 401, "Wrong token.")
        return False
    if authed(h):
        return True
    if _locked(ip):
        _send(h, 429, "Too many wrong tries. Wait ten minutes.", "text/plain; charset=utf-8")
        return False
    if h.command in ("GET", "HEAD") and path in ("/", "/index.html", "/nse_dashboard.html"):
        _login_page(h, 401, "")
    else:
        _send(h, 401, '{"error":"sign in required"}', "application/json")
    return False


# ── TLS ──────────────────────────────────────────────────────────────────
def lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return ""


def wanted_names():
    """(dns names, ip addresses) the certificate has to cover."""
    dns, ips = {"localhost"}, {"127.0.0.1"}
    li = lan_ip()
    if li:
        ips.add(li)
    try:
        dns.add(socket.gethostname())
    except Exception:
        pass
    for part in (_env("NSE_PUBLIC_IP") + "," + _env("NSE_PUBLIC_HOST") + "," + _env("NSE_TLS_NAMES")).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            ips.add(str(ipaddress.ip_address(part)))
        except ValueError:
            dns.add(part)
    return sorted(dns), sorted(ips)


def tls_files(base_dir):
    cert, key = _env("NSE_TLS_CERT"), _env("NSE_TLS_KEY")
    if cert and key:
        return cert, key, False
    d = os.path.join(base_dir, "tls")
    return os.path.join(d, "server.crt"), os.path.join(d, "server.key"), True


def _make_with_cryptography(cert, key, dns, ips, days):
    import datetime
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID
    k = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "NSE dashboard (self-signed)")])
    now = datetime.datetime.now(datetime.timezone.utc)
    san = [x509.DNSName(n) for n in dns] + [x509.IPAddress(ipaddress.ip_address(i)) for i in ips]
    c = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(k.public_key())
         .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(days=1))
         .not_valid_after(now + datetime.timedelta(days=days))
         .add_extension(x509.SubjectAlternativeName(san), critical=False)
         .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
         .add_extension(x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
         .sign(k, hashes.SHA256()))
    with open(key, "wb") as f:
        f.write(k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    with open(cert, "wb") as f:
        f.write(c.public_bytes(serialization.Encoding.PEM))


def _make_with_openssl(cert, key, dns, ips, days):
    alt = [f"DNS.{i + 1} = {n}" for i, n in enumerate(dns)] + [f"IP.{i + 1} = {n}" for i, n in enumerate(ips)]
    cnf = ("[req]\ndistinguished_name=dn\nx509_extensions=v3\nprompt=no\n[dn]\nCN=NSE dashboard (self-signed)\n"
           "[v3]\nsubjectAltName=@alt\nbasicConstraints=critical,CA:FALSE\nextendedKeyUsage=serverAuth\n[alt]\n" + "\n".join(alt) + "\n")
    with tempfile.NamedTemporaryFile("w", suffix=".cnf", delete=False) as f:
        f.write(cnf)
        path = f.name
    try:
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-sha256", "-days", str(days),
                        "-keyout", key, "-out", cert, "-config", path],
                       check=True, capture_output=True, timeout=60)
    finally:
        os.unlink(path)


def ensure_cert(cert, key, days=800):
    """Make the self-signed certificate if it is missing or no longer covers
    this machine's names. Returns the names it covers."""
    dns, ips = wanted_names()
    sig = ",".join(dns) + "|" + ",".join(ips)
    side = cert + ".names"
    have = ""
    try:
        have = open(side).read()
    except Exception:
        pass
    if os.path.exists(cert) and os.path.exists(key) and have == sig:
        return dns, ips
    os.makedirs(os.path.dirname(cert), exist_ok=True)
    try:
        _make_with_cryptography(cert, key, dns, ips, days)
    except ImportError:
        _make_with_openssl(cert, key, dns, ips, days)
    try:
        os.chmod(key, 0o600)
    except Exception:
        pass
    with open(side, "w") as f:
        f.write(sig)
    return dns, ips


def server_context(cert, key):
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_cert_chain(cert, key)
    return ctx


def wrap_listener(server, ctx):
    """TLS on the listening socket, handshake deferred: accept() must not wait
    for a slow or hostile client, so the handshake runs in that client's own
    thread (see handshake)."""
    server.socket = ctx.wrap_socket(server.socket, server_side=True, do_handshake_on_connect=False)


def handshake(sock, timeout=10):
    sock.settimeout(timeout)
    sock.do_handshake()
    sock.settimeout(None)
