"""
Verify TURN credentials actually work, without needing two people and a call.

    py scripts/verify_turn.py                       # read TURN_* from .env
    py scripts/verify_turn.py --url https://azsl-sign-demo.onrender.com \
                            --email you@example.com --password ...

With --url it signs in to a deployment, reads /api/rtc-config, and tests the
relay the server is really handing to browsers — which catches the usual
mistakes (variables set on the wrong host, a typo'd password, a partial
config) that otherwise only show up as a call that silently fails.

It performs a real STUN/TURN Allocate request over UDP and reports what the
server said. A wrong password comes back as 401 Unauthorized, which is the
single most useful thing to learn before blaming the network.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import os
import secrets
import socket
import struct
import sys
import urllib.parse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MAGIC_COOKIE = 0x2112A442
BIND_REQUEST = 0x0001
ALLOCATE_REQUEST = 0x0003

ATTR_USERNAME = 0x0006
ATTR_MESSAGE_INTEGRITY = 0x0008
ATTR_ERROR_CODE = 0x0009
ATTR_REALM = 0x0014
ATTR_NONCE = 0x0015
ATTR_XOR_RELAYED_ADDRESS = 0x0016
ATTR_REQUESTED_TRANSPORT = 0x0019
ATTR_XOR_MAPPED_ADDRESS = 0x0020


def _pad4(b: bytes) -> bytes:
    return b + b"\x00" * ((4 - len(b) % 4) % 4)


def _attr(kind: int, value: bytes) -> bytes:
    return struct.pack("!HH", kind, len(value)) + _pad4(value)


def _build(msg_type: int, txid: bytes, attrs: bytes = b"") -> bytes:
    return struct.pack("!HHI", msg_type, len(attrs), MAGIC_COOKIE) + txid + attrs


def _with_integrity(msg_type: int, txid: bytes, attrs: bytes, key: bytes) -> bytes:
    # Length must include the MESSAGE-INTEGRITY attribute that is about to be
    # appended (4 byte header + 20 byte digest) before the HMAC is computed.
    header = struct.pack("!HHI", msg_type, len(attrs) + 24, MAGIC_COOKIE) + txid
    digest = hmac.new(key, header + attrs, hashlib.sha1).digest()
    return header + attrs + _attr(ATTR_MESSAGE_INTEGRITY, digest)


def _parse_attrs(data: bytes) -> dict[int, bytes]:
    out: dict[int, bytes] = {}
    i = 20
    while i + 4 <= len(data):
        kind, length = struct.unpack("!HH", data[i:i + 4])
        value = data[i + 4:i + 4 + length]
        out[kind] = value
        i += 4 + length + ((4 - length % 4) % 4)
    return out


def _decode_xor_address(value: bytes) -> str | None:
    if len(value) < 8:
        return None
    family = value[1]
    port = struct.unpack("!H", value[2:4])[0] ^ (MAGIC_COOKIE >> 16)
    if family == 0x01:
        raw = struct.unpack("!I", value[4:8])[0] ^ MAGIC_COOKIE
        return f"{socket.inet_ntoa(struct.pack('!I', raw))}:{port}"
    return f"[ipv6]:{port}"


def parse_turn_url(url: str) -> tuple[str, str, int, str]:
    """('turn'|'turns', host, port, transport) from a TURN URI."""
    scheme, _, rest = url.partition(":")
    rest, _, query = rest.partition("?")
    transport = "udp"
    if query:
        params = urllib.parse.parse_qs(query)
        transport = (params.get("transport") or ["udp"])[0]
    host, _, port_s = rest.rpartition(":")
    if not host:
        host, port_s = rest, "3478"
    port = int(port_s) if port_s.isdigit() else 3478
    return scheme, host, port, transport


def probe(url: str, username: str, credential: str, timeout: float = 5.0) -> tuple[str, str]:
    """Send a real Allocate request.

    Returns (status, detail) where status is one of:

      "ok"        the relay allocated a port for these credentials. Confident.
      "bad_creds" the server rejected the credentials. Confident.
      "unknown"   something else happened. NOT a verdict on the credentials --
                  a server may refuse an allocation by policy, and UDP may be
                  blocked on this machine while the TLS/443 URL still works
                  fine from a browser.

    Keeping "unknown" separate matters: reporting it as failure would send you
    hunting a credential problem that may not exist.
    """
    scheme, host, port, transport = parse_turn_url(url)

    if transport != "udp" or scheme == "turns":
        # TLS and TCP relays need a full TLS/TCP STUN client; a UDP probe would
        # be meaningless. Say so rather than implying it passed.
        return "unknown", f"not probed ({scheme}/{transport} needs a TLS/TCP client)"

    try:
        addr = socket.getaddrinfo(host, port, proto=socket.IPPROTO_UDP)[0][4]
    except socket.gaierror as exc:
        return "unknown", f"DNS lookup failed: {exc}"

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(timeout)
    try:
        txid = secrets.token_bytes(12)
        attrs = _attr(ATTR_REQUESTED_TRANSPORT, b"\x11\x00\x00\x00")  # 17 = UDP
        sock.sendto(_build(ALLOCATE_REQUEST, txid, attrs), addr)

        try:
            data, _ = sock.recvfrom(2048)
        except socket.timeout:
            return "unknown", "no reply (UDP blocked on this machine, or wrong host/port)"

        parsed = _parse_attrs(data)
        # The first Allocate is expected to be refused with 401 + realm/nonce.
        realm = parsed.get(ATTR_REALM)
        nonce = parsed.get(ATTR_NONCE)
        if realm is None or nonce is None:
            code = parsed.get(ATTR_ERROR_CODE)
            if code and len(code) >= 4:
                return "unknown", f"server answered {code[2] * 100 + code[3]} without a credential challenge"
            return "unknown", "server did not challenge for credentials (is this a TURN server?)"

        key = hashlib.md5(
            b":".join([username.encode(), realm, credential.encode()])
        ).digest()

        txid2 = secrets.token_bytes(12)
        attrs2 = (
            _attr(ATTR_REQUESTED_TRANSPORT, b"\x11\x00\x00\x00")
            + _attr(ATTR_USERNAME, username.encode())
            + _attr(ATTR_REALM, realm)
            + _attr(ATTR_NONCE, nonce)
        )
        sock.sendto(_with_integrity(ALLOCATE_REQUEST, txid2, attrs2, key), addr)

        try:
            data2, _ = sock.recvfrom(2048)
        except socket.timeout:
            return "unknown", "no reply to the authenticated request"

        msg_type = struct.unpack("!H", data2[0:2])[0]
        parsed2 = _parse_attrs(data2)

        if msg_type == 0x0103:  # Allocate Success
            relayed = _decode_xor_address(parsed2.get(ATTR_XOR_RELAYED_ADDRESS, b""))
            return "ok", f"relay allocated at {relayed} — credentials accepted"

        code = parsed2.get(ATTR_ERROR_CODE)
        if code and len(code) >= 4:
            number = code[2] * 100 + code[3]
            reason = code[4:].decode(errors="replace").strip()
            if number == 401:
                return "bad_creds", "401 Unauthorized — username or credential is wrong"
            if number == 403:
                return "bad_creds", f"403 Forbidden — credentials rejected {reason}".strip()
            # 400/486/508 and friends are server policy, not a credential verdict.
            # Observed in the wild: a retired free tier answers 400 to every
            # allocation regardless of what you send.
            return "unknown", f"{number} {reason} — server declined; not a credential verdict"
        return "unknown", "allocation refused with no error code"
    finally:
        sock.close()


def from_deployment(base_url: str, email: str, password: str):
    """Sign in and read the ICE config the server actually serves."""
    import httpx

    base = base_url.rstrip("/")
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        res = client.post(f"{base}/api/login", json={"email": email, "password": password})
        if res.status_code != 200:
            print(f"  login failed ({res.status_code}): {res.text[:200]}")
            return None
        cfg = client.get(f"{base}/api/rtc-config")
        if cfg.status_code != 200:
            print(f"  /api/rtc-config failed ({cfg.status_code})")
            return None
        return cfg.json()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", help="deployment base URL, e.g. https://azsl-sign-demo.onrender.com")
    parser.add_argument("--email", help="an account on that deployment")
    parser.add_argument("--password")
    args = parser.parse_args()

    if args.url:
        if not (args.email and args.password):
            print("--url needs --email and --password (any account on that deployment)")
            return 2
        print(f"Reading ICE config from {args.url} ...")
        cfg = from_deployment(args.url, args.email, args.password)
        if cfg is None:
            return 1
        servers = cfg.get("iceServers", [])
        print(f"  server reports turn={cfg.get('turn')}")
        if not cfg.get("turn"):
            print("\n  TURN is NOT configured on that host.")
            print("  Set TURN_URL / TURN_USERNAME / TURN_CREDENTIAL there and restart it.")
            print("  Note: the browser reads this from whichever host serves the PAGE.")
            return 1
    else:
        try:
            from dotenv import load_dotenv

            load_dotenv(PROJECT_ROOT / ".env")
        except Exception:
            pass
        from src.web_demo.social import rtc_ice_servers, turn_configured

        servers = rtc_ice_servers()
        print(f"Reading TURN_* from the local environment (turn={turn_configured()})")
        if not turn_configured():
            print("\n  TURN_URL / TURN_USERNAME / TURN_CREDENTIAL are not all set here.")
            print("  That is fine locally — but set them on the host that serves the page.")
            return 1

    counts = {"ok": 0, "bad_creds": 0, "unknown": 0}
    has_tls_fallback = False

    for entry in servers:
        urls = entry.get("urls")
        urls = [urls] if isinstance(urls, str) else list(urls or [])
        username = entry.get("username")
        credential = entry.get("credential")
        if not username:
            print(f"\nSTUN  {', '.join(urls)}")
            continue
        print(f"\nTURN  username={username}")
        for url in urls:
            if url.startswith("turns:") or "transport=tcp" in url:
                has_tls_fallback = True
            status, detail = probe(url, username, credential)
            counts[status] += 1
            mark = {"ok": "OK  ", "bad_creds": "FAIL", "unknown": "?   "}[status]
            print(f"  {mark}  {url}\n        {detail}")

    print("\n" + "=" * 62)
    if sum(counts.values()) == 0:
        print("No TURN entries to test.")
        print("=" * 62)
        return 1

    if counts["bad_creds"]:
        print("Credentials were REJECTED by the server.")
        print("Re-copy TURN_USERNAME and TURN_CREDENTIAL from your provider's")
        print("dashboard — they are usually regenerated when you reset them.")
        print("=" * 62)
        return 1

    if counts["ok"]:
        print(f"Relay works: {counts['ok']} URL(s) allocated successfully.")
        if not has_tls_fallback:
            print("\nConsider also adding the provider's TLS/443 URL to TURN_URL")
            print("(comma-separated). Networks that block UDP are exactly the")
            print("ones that need a relay, and only the TLS entry reaches them.")
        print("=" * 62)
        return 0

    # Nothing confirmed either way.
    print("INCONCLUSIVE — nothing was rejected, but nothing was confirmed either.")
    print("This is common and does not mean the relay is broken:")
    print("  * UDP is often blocked on office/university networks, so the probe")
    print("    times out while a browser using the TLS/443 URL connects fine")
    print("  * some providers refuse raw allocations by policy but still serve")
    print("    real WebRTC clients")
    print("\nThe definitive test is a real call between two networks.")
    if not has_tls_fallback:
        print("Before that, add the provider's TLS/443 URL to TURN_URL as well.")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
