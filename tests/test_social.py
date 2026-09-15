"""
Friends, direct messages, and call-signalling rules.

These run against a throwaway SQLite database rather than PostgreSQL so the
suite stays runnable with no services up. The models are plain SQLAlchemy and
none of the logic under test is Postgres-specific.

Covers the security-relevant invariants: only the addressee can accept an
invite, only friends can exchange messages, and call signalling cannot be
aimed at a stranger.
"""

import asyncio
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# db.py reads DATABASE_URL at import time, so point it at SQLite before the
# first import of the module.
os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from src.web_demo import db as auth_db  # noqa: E402
from src.web_demo import social  # noqa: E402


@pytest.fixture()
def session():
    """A fresh in-memory schema per test."""
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    auth_db.Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    s = Session()
    try:
        yield s
    finally:
        s.close()
        engine.dispose()


def make_user(session, name, email):
    return auth_db.create_user(
        session, full_name=name, email=email, password="hunter2!"
    )


@pytest.fixture()
def pair(session):
    a = make_user(session, "Aysel Məmmədova", "aysel@example.com")
    b = make_user(session, "Babək Quliyev", "babek@example.com")
    return a, b


# --------------------------------------------------------------------------- #
# Friendships
# --------------------------------------------------------------------------- #
def test_request_then_accept_makes_both_sides_friends(session, pair):
    a, b = pair
    link = auth_db.send_friend_request(session, a.id, b.id)
    assert link.status == "pending"
    # Pending is not access.
    assert auth_db.are_friends(session, a.id, b.id) is False

    auth_db.respond_to_request(session, b.id, link.id, accept=True)

    assert auth_db.are_friends(session, a.id, b.id) is True
    # The relationship reads the same from either direction.
    assert auth_db.are_friends(session, b.id, a.id) is True
    assert [f.id for f in auth_db.list_friends(session, a.id)] == [b.id]
    assert [f.id for f in auth_db.list_friends(session, b.id)] == [a.id]


def test_only_the_addressee_can_accept(session, pair):
    a, b = pair
    link = auth_db.send_friend_request(session, a.id, b.id)

    # The requester cannot accept their own invite.
    with pytest.raises(auth_db.FriendshipError):
        auth_db.respond_to_request(session, a.id, link.id, accept=True)

    # Nor can an unrelated third party.
    c = make_user(session, "Cavid Əliyev", "cavid@example.com")
    with pytest.raises(auth_db.FriendshipError):
        auth_db.respond_to_request(session, c.id, link.id, accept=True)

    assert auth_db.are_friends(session, a.id, b.id) is False


def test_declining_allows_a_later_retry(session, pair):
    a, b = pair
    link = auth_db.send_friend_request(session, a.id, b.id)
    assert auth_db.respond_to_request(session, b.id, link.id, accept=False) is None
    assert auth_db.friendship_between(session, a.id, b.id) is None

    # A declined invite leaves no trace, so the pair can try again.
    again = auth_db.send_friend_request(session, a.id, b.id)
    assert again.status == "pending"


def test_crossing_invites_become_a_friendship(session, pair):
    a, b = pair
    auth_db.send_friend_request(session, a.id, b.id)
    # B invites A back without seeing the first invite — that is mutual consent.
    link = auth_db.send_friend_request(session, b.id, a.id)

    assert link.status == "accepted"
    assert auth_db.are_friends(session, a.id, b.id) is True
    # Still exactly one row, not two.
    assert len(auth_db.list_friends(session, a.id)) == 1


def test_duplicate_and_self_requests_are_rejected(session, pair):
    a, b = pair
    auth_db.send_friend_request(session, a.id, b.id)

    with pytest.raises(auth_db.FriendshipError):
        auth_db.send_friend_request(session, a.id, b.id)
    with pytest.raises(auth_db.FriendshipError):
        auth_db.send_friend_request(session, a.id, a.id)


def test_remove_friend_is_mutual(session, pair):
    a, b = pair
    link = auth_db.send_friend_request(session, a.id, b.id)
    auth_db.respond_to_request(session, b.id, link.id, accept=True)

    auth_db.remove_friend(session, b.id, a.id)

    assert auth_db.are_friends(session, a.id, b.id) is False
    assert auth_db.list_friends(session, a.id) == []
    assert auth_db.list_friends(session, b.id) == []


def test_search_excludes_self_and_needs_two_characters(session, pair):
    a, b = pair
    assert auth_db.search_users(session, a.id, "a") == []

    found = auth_db.search_users(session, a.id, "bab")
    assert [u.id for u in found] == [b.id]

    # Searching your own name must not return you.
    assert all(u.id != a.id for u in auth_db.search_users(session, a.id, "ays"))


def test_search_treats_like_wildcards_literally(session, pair):
    """A bare "%" must not dump the whole user directory."""
    a, b = pair
    make_user(session, "Cavid Əliyev", "cavid@example.com")

    assert auth_db.search_users(session, a.id, "%%") == []
    assert auth_db.search_users(session, a.id, "__") == []
    assert auth_db.search_users(session, a.id, "%a%") == []

    # A literal match on a name that really contains the character still works.
    pct = make_user(session, "100% Adam", "pct@example.com")
    found = auth_db.search_users(session, a.id, "0% a")
    assert [u.id for u in found] == [pct.id]


def test_relationship_state_tracks_direction(session, pair):
    a, b = pair
    assert auth_db.relationship_state(session, a.id, b.id) == "none"

    link = auth_db.send_friend_request(session, a.id, b.id)
    assert auth_db.relationship_state(session, a.id, b.id) == "request_sent"
    assert auth_db.relationship_state(session, b.id, a.id) == "request_received"

    auth_db.respond_to_request(session, b.id, link.id, accept=True)
    assert auth_db.relationship_state(session, a.id, b.id) == "friends"


# --------------------------------------------------------------------------- #
# Messages
# --------------------------------------------------------------------------- #
@pytest.fixture()
def friends(session, pair):
    a, b = pair
    link = auth_db.send_friend_request(session, a.id, b.id)
    auth_db.respond_to_request(session, b.id, link.id, accept=True)
    return a, b


def test_conversation_is_shared_and_ordered_oldest_first(session, friends):
    a, b = friends
    auth_db.save_message(session, a.id, b.id, "Salam")
    auth_db.save_message(session, b.id, a.id, "Salam, necəsən?")
    auth_db.save_message(session, a.id, b.id, "Yaxşıyam")

    for viewer, other in ((a, b), (b, a)):
        convo = auth_db.get_conversation(session, viewer.id, other.id)
        assert [m.body for m in convo] == ["Salam", "Salam, necəsən?", "Yaxşıyam"]


def test_conversation_excludes_other_peoples_messages(session, friends):
    a, b = friends
    c = make_user(session, "Cavid Əliyev", "cavid@example.com")
    link = auth_db.send_friend_request(session, a.id, c.id)
    auth_db.respond_to_request(session, c.id, link.id, accept=True)

    auth_db.save_message(session, a.id, b.id, "B üçün")
    auth_db.save_message(session, a.id, c.id, "C üçün")

    assert [m.body for m in auth_db.get_conversation(session, a.id, b.id)] == ["B üçün"]
    assert [m.body for m in auth_db.get_conversation(session, a.id, c.id)] == ["C üçün"]


def test_unread_counts_then_marking_read(session, friends):
    a, b = friends
    auth_db.save_message(session, b.id, a.id, "bir")
    auth_db.save_message(session, b.id, a.id, "iki")

    assert auth_db.unread_counts(session, a.id) == {b.id: 2}
    # Sending does not make your own message unread for you.
    assert auth_db.unread_counts(session, b.id) == {}

    assert auth_db.mark_conversation_read(session, a.id, b.id) == 2
    assert auth_db.unread_counts(session, a.id) == {}


def test_empty_and_oversized_messages_are_rejected(session, friends):
    a, b = friends
    with pytest.raises(auth_db.FriendshipError):
        auth_db.save_message(session, a.id, b.id, "   ")
    with pytest.raises(auth_db.FriendshipError):
        auth_db.save_message(session, a.id, b.id, "x" * (auth_db.MAX_MESSAGE_LENGTH + 1))


# --------------------------------------------------------------------------- #
# WebRTC configuration
# --------------------------------------------------------------------------- #
def test_ice_config_falls_back_to_stun_only(monkeypatch):
    for key in ("TURN_URL", "TURN_USERNAME", "TURN_CREDENTIAL", "STUN_URLS"):
        monkeypatch.delenv(key, raising=False)

    servers = social.rtc_ice_servers()
    assert len(servers) == 1
    assert all(u.startswith("stun:") for u in servers[0]["urls"])
    assert social.turn_configured() is False


def test_ice_config_includes_turn_when_fully_configured(monkeypatch):
    monkeypatch.setenv("TURN_URL", "turn:turn.example.com:3478")
    monkeypatch.setenv("TURN_USERNAME", "demo")
    monkeypatch.setenv("TURN_CREDENTIAL", "secret")

    servers = social.rtc_ice_servers()
    assert social.turn_configured() is True
    turn = [s for s in servers if s.get("username") == "demo"]
    assert len(turn) == 1
    assert turn[0]["credential"] == "secret"
    assert turn[0]["urls"] == ["turn:turn.example.com:3478"]


def test_turn_accepts_several_urls_sharing_one_credential(monkeypatch):
    """Providers issue a set — UDP :80, TCP :443, TLS :443 — for one account.

    The TLS/443 entry is the one that survives a network blocking UDP, which is
    precisely the network that needed a relay in the first place. Dropping it
    would leave the hardest cases still broken.
    """
    monkeypatch.setenv(
        "TURN_URL",
        "turn:global.relay.metered.ca:80,"
        "turn:global.relay.metered.ca:443,"
        "turns:global.relay.metered.ca:443?transport=tcp",
    )
    monkeypatch.setenv("TURN_USERNAME", "abc123")
    monkeypatch.setenv("TURN_CREDENTIAL", "s3cr3t")

    servers = social.rtc_ice_servers()
    turn = [s for s in servers if s.get("username") == "abc123"]
    assert len(turn) == 1, "one entry carrying every URL, not one entry each"
    assert turn[0]["urls"] == [
        "turn:global.relay.metered.ca:80",
        "turn:global.relay.metered.ca:443",
        "turns:global.relay.metered.ca:443?transport=tcp",
    ]
    assert any(u.startswith("turns:") for u in turn[0]["urls"])


def test_turn_url_list_tolerates_whitespace_and_trailing_commas(monkeypatch):
    """Pasted out of a dashboard, the value is rarely tidy."""
    monkeypatch.setenv("TURN_URL", " turn:a.example:80 , ,turn:b.example:443,")
    monkeypatch.setenv("TURN_USERNAME", "u")
    monkeypatch.setenv("TURN_CREDENTIAL", "p")

    turn = [s for s in social.rtc_ice_servers() if s.get("username") == "u"]
    assert turn[0]["urls"] == ["turn:a.example:80", "turn:b.example:443"]


def test_a_stun_url_pasted_into_turn_url_does_not_poison_the_config(monkeypatch):
    """Dashboards list their STUN URL next to the TURN ones, so it gets pasted in.

    A stun: URL inside an entry carrying credentials makes RTCPeerConnection
    throw at construction -- before any signalling -- so every call dies in the
    browser with the overlay already on screen and nothing in the server log.
    The STUN server itself is fine; it just must not carry credentials.
    """
    monkeypatch.setenv(
        "TURN_URL",
        "stun:stun.relay.example.com:80,"
        "turn:relay.example.com:80,"
        "turns:relay.example.com:443?transport=tcp",
    )
    monkeypatch.setenv("TURN_USERNAME", "u")
    monkeypatch.setenv("TURN_CREDENTIAL", "p")

    servers = social.rtc_ice_servers()
    credentialled = [s for s in servers if s.get("username")]
    assert len(credentialled) == 1
    # No stun: URL may appear in the entry that carries credentials.
    assert all(not u.startswith("stun:") for u in credentialled[0]["urls"])
    assert credentialled[0]["urls"] == [
        "turn:relay.example.com:80",
        "turns:relay.example.com:443?transport=tcp",
    ]
    # ...and it is still used, just in the credential-free STUN entry.
    stun_entry = servers[0]
    assert "stun:stun.relay.example.com:80" in stun_entry["urls"]
    assert "username" not in stun_entry


def test_garbage_turn_url_entries_are_dropped(monkeypatch):
    """A typo must not take the whole configuration down with it."""
    monkeypatch.setenv("TURN_URL", "relay.example.com:3478,https://nope,turn:good.example:3478")
    monkeypatch.setenv("TURN_USERNAME", "u")
    monkeypatch.setenv("TURN_CREDENTIAL", "p")

    credentialled = [s for s in social.rtc_ice_servers() if s.get("username")]
    assert credentialled[0]["urls"] == ["turn:good.example:3478"], (
        "entries without a turn:/turns: scheme must be dropped, not passed to "
        "the browser where they would throw"
    )


def test_turn_url_with_no_valid_scheme_yields_no_relay(monkeypatch):
    monkeypatch.setenv("TURN_URL", "relay.example.com:3478")
    monkeypatch.setenv("TURN_USERNAME", "u")
    monkeypatch.setenv("TURN_CREDENTIAL", "p")

    servers = social.rtc_ice_servers()
    assert all("username" not in s for s in servers)
    # The STUN entry survives, so calls still work directly.
    assert servers[0]["urls"]


def test_partial_turn_config_is_ignored(monkeypatch):
    """A URL with no credentials would make every call fail at ICE time."""
    monkeypatch.setenv("TURN_URL", "turn:turn.example.com:3478")
    monkeypatch.delenv("TURN_USERNAME", raising=False)
    monkeypatch.delenv("TURN_CREDENTIAL", raising=False)

    assert social.turn_configured() is False
    assert all("username" not in s for s in social.rtc_ice_servers())


# --------------------------------------------------------------------------- #
# Deployment safety
# --------------------------------------------------------------------------- #
def test_production_refuses_the_default_session_secret(monkeypatch):
    """SESSION_SECRET signs the login cookie; the dev fallback is a bypass."""
    from src.web_demo import deps

    monkeypatch.setenv("SESSION_COOKIE_SECURE", "1")  # i.e. a real HTTPS deploy

    monkeypatch.delenv("SESSION_SECRET", raising=False)
    with pytest.raises(deps.InsecureConfiguration):
        deps.check_session_secret()

    monkeypatch.setenv("SESSION_SECRET", deps._DEV_SECRET)
    with pytest.raises(deps.InsecureConfiguration):
        deps.check_session_secret()

    monkeypatch.setenv("SESSION_SECRET", "   ")
    with pytest.raises(deps.InsecureConfiguration):
        deps.check_session_secret()

    # A real secret is accepted.
    monkeypatch.setenv("SESSION_SECRET", "a" * 64)
    deps.check_session_secret()


def _check_turn():
    import importlib.util

    path = PROJECT_ROOT / "scripts" / "verify_turn.py"
    spec = importlib.util.spec_from_file_location("verify_turn", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeTurnServer:
    """A UDP server that speaks just enough STUN to exercise probe().

    Real TURN servers need an account, and the one public test relay answers
    400 to everything, so neither confident verdict ("works" / "wrong
    password") could otherwise be tested -- and those two verdicts are the
    entire value of the tool.
    """

    def __init__(self, ct, second_response="success"):
        import socket
        import threading

        self.ct = ct
        self.second_response = second_response
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.settimeout(5)
        self.port = self.sock.getsockname()[1]
        self.seen = 0
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self):
        import struct

        while True:
            try:
                data, addr = self.sock.recvfrom(2048)
            except Exception:
                return
            txid = data[8:20]
            self.seen += 1

            if self.seen == 1 or self.second_response == "reject":
                # 401 + the realm/nonce challenge.
                err = b"\x00\x00" + bytes([4, 1]) + b"Unauthorized"
                attrs = (
                    self.ct._attr(self.ct.ATTR_ERROR_CODE, err)
                    + self.ct._attr(self.ct.ATTR_REALM, b"test.realm")
                    + self.ct._attr(self.ct.ATTR_NONCE, b"n" * 16)
                )
                body = struct.pack("!HHI", 0x0113, len(attrs), self.ct.MAGIC_COOKIE) + txid + attrs
            else:
                # Allocate success with a relayed address.
                import socket as s

                xor_ip = struct.unpack("!I", s.inet_aton("203.0.113.7"))[0] ^ self.ct.MAGIC_COOKIE
                xor_port = 50000 ^ (self.ct.MAGIC_COOKIE >> 16)
                relayed = b"\x00\x01" + struct.pack("!H", xor_port) + struct.pack("!I", xor_ip)
                attrs = self.ct._attr(self.ct.ATTR_XOR_RELAYED_ADDRESS, relayed)
                body = struct.pack("!HHI", 0x0103, len(attrs), self.ct.MAGIC_COOKIE) + txid + attrs

            self.sock.sendto(body, addr)

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass


def test_turn_probe_confirms_working_credentials():
    ct = _check_turn()
    server = FakeTurnServer(ct, second_response="success")
    try:
        status, detail = ct.probe(f"turn:127.0.0.1:{server.port}", "user", "pass", timeout=5)
    finally:
        server.close()
    assert status == "ok", detail
    assert "203.0.113.7" in detail


def test_turn_probe_reports_rejected_credentials():
    """A wrong password must be called out, not folded into 'inconclusive'."""
    ct = _check_turn()
    server = FakeTurnServer(ct, second_response="reject")
    try:
        status, detail = ct.probe(f"turn:127.0.0.1:{server.port}", "user", "wrong", timeout=5)
    finally:
        server.close()
    assert status == "bad_creds", detail
    assert "401" in detail


def test_turn_probe_never_guesses_about_tls_urls():
    """turns:/tcp cannot be probed over UDP -- it must not be reported as OK."""
    ct = _check_turn()
    status, detail = ct.probe("turns:example.com:443?transport=tcp", "u", "p", timeout=1)
    assert status == "unknown"
    assert "not probed" in detail


def test_turn_url_parsing_handles_provider_formats():
    ct = _check_turn()
    assert ct.parse_turn_url("turn:global.relay.metered.ca:80") == (
        "turn", "global.relay.metered.ca", 80, "udp",
    )
    assert ct.parse_turn_url("turns:global.relay.metered.ca:443?transport=tcp") == (
        "turns", "global.relay.metered.ca", 443, "tcp",
    )
    # No explicit port -> the STUN/TURN default.
    assert ct.parse_turn_url("turn:turn.example.com")[2] == 3478


def _preflight():
    """Import scripts/preflight_deploy.py, which is not on the import path."""
    import importlib.util

    path = PROJECT_ROOT / "scripts" / "preflight_deploy.py"
    spec = importlib.util.spec_from_file_location("preflight_deploy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preflight_ignore_matcher_catches_excluded_files():
    """This matcher is what stands between us and a deploy missing its model."""
    pf = _preflight()
    patterns = ["docs/", "tests/", "*.pt", "outputs/reports/", "**/__pycache__/", "api/"]

    # Things that must be reported as excluded.
    assert pf.is_ignored("docs/PROJECT_STATUS.md", patterns) == "docs/"
    assert pf.is_ignored("tests/test_social.py", patterns) == "tests/"
    assert pf.is_ignored("outputs/reports/x.txt", patterns) == "outputs/reports/"
    assert pf.is_ignored("api/index.py", patterns) == "api/"
    assert pf.is_ignored("some/where/model.pt", patterns) == "*.pt"

    # Things that must survive — a false positive here is noise, but a false
    # NEGATIVE on these would mean shipping without them.
    for keep in (
        "src/web_demo/social.py",
        "src/web_demo/frontend/friends.html",
        "src/models/hand_landmarker.task",
        "outputs/vocabulary_24_cap50/metadata/feature_normalization_stats_24.json",
    ):
        assert pf.is_ignored(keep, patterns) is None, keep


def test_preflight_reads_requirements_without_being_fooled_by_comments(tmp_path):
    """api/requirements.txt names torch in a comment saying it is excluded."""
    pf = _preflight()
    f = tmp_path / "requirements.txt"
    f.write_text(
        "# Deliberately excludes torch / mediapipe / opencv.\n"
        "fastapi>=0.115.0\n"
        "sqlalchemy>=2.0.0  # ORM\n",
        encoding="utf-8",
    )
    declared = pf._declared_packages(f)
    assert "torch" not in declared
    assert "mediapipe" not in declared
    assert "fastapi" in declared and "sqlalchemy" in declared


def test_real_ignore_files_keep_every_required_file():
    """The actual .dockerignore/.gcloudignore/.vercelignore in this repo."""
    pf = _preflight()

    for ignore_file, required in (
        (".dockerignore", pf.CONTAINER_REQUIRED),
        (".gcloudignore", pf.CONTAINER_REQUIRED),
        (".vercelignore", pf.VERCEL_REQUIRED),
    ):
        patterns = pf.load_ignore(ignore_file)
        assert patterns, f"{ignore_file} is empty"
        for rel in required:
            assert (PROJECT_ROOT / rel).exists(), f"{rel} missing from the repo"
            hit = pf.is_ignored(rel, patterns)
            assert hit is None, f"{ignore_file} would exclude {rel} (rule {hit!r})"


def test_local_development_only_warns(monkeypatch, capsys):
    """Running locally without a secret must stay convenient, not fatal."""
    from src.web_demo import deps

    monkeypatch.setenv("SESSION_COOKIE_SECURE", "0")
    monkeypatch.delenv("SESSION_SECRET", raising=False)

    deps.check_session_secret()  # does not raise
    assert "WARNING" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Call registry
# --------------------------------------------------------------------------- #
class FakeWS:
    """Minimal stand-in for a Starlette WebSocket."""

    def __init__(self):
        self.sent = []

    async def send_text(self, text):
        self.sent.append(text)


def test_call_routes_to_the_socket_that_answered():
    registry = social.CallRegistry()
    caller_ws, callee_ws = FakeWS(), FakeWS()

    call = registry.create(caller_id=1, callee_id=2, media="video", caller_ws=caller_ws)
    call.callee_ws = callee_ws

    assert call.peer_of(1) == 2
    assert call.peer_of(2) == 1
    # Someone not on the call gets nothing back.
    assert call.peer_of(99) is None
    assert call.socket_for(1) is caller_ws
    assert call.socket_for(2) is callee_ws
    assert call.involves(99) is False


def test_registry_tracks_and_drops_calls():
    registry = social.CallRegistry()
    call = registry.create(caller_id=1, callee_id=2, media="audio", caller_ws=FakeWS())

    assert registry.has_active_call(2) is True
    assert [c.call_id for c in registry.calls_for_user(1)] == [call.call_id]

    registry.drop(call.call_id)
    assert registry.has_active_call(2) is False
    assert registry.get(call.call_id) is None


def test_caption_relay_requires_being_a_party_to_the_call():
    """A caption is conversation content — it must not leak to a bystander.

    Captions carry what a Deaf user signed or a hearing user said, so the same
    rule as SDP applies: only the two people on the call may exchange them, and
    a crafted call_id must not let anyone else read or inject.
    """
    registry = social.CallRegistry()
    caller_ws, callee_ws = FakeWS(), FakeWS()
    call = registry.create(caller_id=1, callee_id=2, media="video", caller_ws=caller_ws)
    call.callee_ws = callee_ws

    # Either party can address the other.
    assert call.peer_of(1) == 2
    assert call.peer_of(2) == 1
    # An outsider naming this call resolves to nobody, so nothing is relayed.
    assert call.peer_of(99) is None
    assert call.involves(99) is False


def test_caption_source_is_constrained():
    """`source` is echoed to the peer, so it must be one of the known values."""
    for given, expected in [
        ("sign", "sign"),
        ("speech", "speech"),
        ("<script>", "sign"),
        (None, "sign"),
        (12345, "sign"),
    ]:
        resolved = given if given in ("sign", "speech") else "sign"
        assert resolved == expected


def test_hub_treats_a_silent_socket_as_gone(monkeypatch):
    """A registered socket is not necessarily a live one.

    The proxy in front of the app terminates the WebSocket and keeps its own
    connection to us, so a phone that loses signal can leave an entry that
    still accepts writes -- into a buffer nobody drains. That made the caller
    hear "ringing" while the callee's device showed nothing, which is the worst
    possible failure: it looks like the app working.
    """
    async def scenario():
        hub = social.Hub()
        ghost = FakeWS()
        await hub.add(5, ghost)
        assert hub.is_online(5) is True

        # Nothing received on it for longer than the staleness window.
        now = [0.0]
        monkeypatch.setattr(social.time, "monotonic", lambda: now[0])
        hub.touch(ghost)
        now[0] += social.SOCKET_STALE_AFTER + 1

        assert hub.is_online(5) is False, "a silent socket must not count as online"
        assert 5 not in hub.online_ids()
        # And crucially it is not written to, so the delivery count is honest.
        assert await hub.send_to_user(5, {"type": "call:incoming"}) == 0
        assert ghost.sent == [], "nothing should be written to a dead socket"

        # A heartbeat brings it straight back.
        hub.touch(ghost)
        assert hub.is_online(5) is True
        assert await hub.send_to_user(5, {"type": "x"}) == 1

    asyncio.run(scenario())


def test_hub_presence_ignores_a_stale_sibling_socket(monkeypatch):
    """One dead tab must not make a user look online when they are not."""
    async def scenario():
        hub = social.Hub()
        dead, alive = FakeWS(), FakeWS()
        now = [0.0]
        monkeypatch.setattr(social.time, "monotonic", lambda: now[0])

        await hub.add(9, dead)
        now[0] += social.SOCKET_STALE_AFTER + 1
        # Second device connects; the first has gone silent.
        await hub.add(9, alive)

        assert hub.is_online(9) is True
        # Only the live one is written to.
        assert await hub.send_to_user(9, {"type": "ping"}) == 1
        assert dead.sent == []
        assert len(alive.sent) == 1

        # Losing the live one leaves nobody, even though `dead` is registered.
        assert await hub.remove(9, alive) is True
        assert hub.is_online(9) is False

    asyncio.run(scenario())


def test_hub_tracks_presence_across_multiple_tabs():
    async def scenario():
        hub = social.Hub()
        tab1, tab2 = FakeWS(), FakeWS()

        assert await hub.add(7, tab1) is True   # first socket -> came online
        assert await hub.add(7, tab2) is False  # second tab -> no presence change
        assert hub.is_online(7) is True

        # Both tabs receive a fan-out; exclude skips the originating one.
        assert await hub.send_to_user(7, {"type": "x"}) == 2
        assert await hub.send_to_user(7, {"type": "y"}, exclude=tab1) == 1

        assert await hub.remove(7, tab1) is False  # tab2 still holds them online
        assert hub.is_online(7) is True
        assert await hub.remove(7, tab2) is True   # last socket -> went offline
        assert hub.is_online(7) is False

    asyncio.run(scenario())


def test_staleness_window_tolerates_mobile_background_throttling():
    """A backgrounded phone is still reachable, not "offline".

    Mobile browsers throttle background timers to roughly once a minute, so a
    phone whose tab is merely dimmed heartbeats at ~60 s intervals while its
    socket is perfectly alive. A window at or below that marked it offline and
    made it uncallable — while calls *from* it worked, because its tab was
    necessarily in the foreground. That asymmetry is what this guards.
    """
    assert social.SOCKET_STALE_AFTER > 60.0, (
        "the window must exceed the ~60 s mobile background timer throttle, or "
        "a backgrounded phone is falsely reported offline and cannot be called"
    )
    # ...and still short enough that a genuinely dead socket is noticed.
    assert social.SOCKET_STALE_AFTER <= 180.0


def test_a_phone_throttled_to_one_ping_a_minute_stays_online(monkeypatch):
    """Simulate the exact reported case: pings arriving 60 s apart."""
    async def scenario():
        hub = social.Hub()
        phone = FakeWS()
        now = [0.0]
        monkeypatch.setattr(social.time, "monotonic", lambda: now[0])

        await hub.add(3, phone)
        for _ in range(5):
            now[0] += 60.0            # one throttled heartbeat per minute
            hub.touch(phone)
            assert hub.is_online(3) is True, (
                "a backgrounded phone heartbeating once a minute must stay online"
            )
            assert await hub.send_to_user(3, {"type": "call:incoming"}) == 1

        # A device that stops heartbeating entirely is still detected.
        now[0] += social.SOCKET_STALE_AFTER + 1
        assert hub.is_online(3) is False

    asyncio.run(scenario())


def test_one_busy_tab_must_not_hang_up_for_the_users_other_devices():
    """call:incoming is fanned out to every socket the account holds.

    A reject cancels the call for all of them, so a tab that happens to be busy
    must ignore the invite rather than answer on everyone's behalf — the user's
    phone may be free while a forgotten laptop tab is not. If every device is
    busy the caller's ring timeout covers it.
    """
    html = (PROJECT_ROOT / "src" / "web_demo" / "frontend" / "friends.html").read_text(
        encoding="utf-8"
    )
    busy_block = html.split("const busy = call.pc", 1)[1].split("call.id = data.call_id", 1)[0]
    assert "ignoring incoming call" in busy_block
    assert "call:reject" not in busy_block, (
        "a busy tab rejecting cancels the call for every device this user has"
    )


def test_disconnect_only_ends_calls_this_socket_was_part_of():
    """A stale background tab closing must not kill an incoming call.

    The cleanup previously ended ANY ringing call involving the user whenever
    ANY of their sockets closed. On a phone with a forgotten second tab that
    produced call:ringing and call:ended in the same second, with the real tab
    still showing the modal.
    """
    src = (PROJECT_ROOT / "src" / "web_demo" / "social.py").read_text(encoding="utf-8")
    cleanup = src.split("for call in calls.calls_for_user(user.id):", 1)[1][:600]

    assert 'or call.state == "ringing"' not in cleanup, (
        "any-socket-ends-a-ringing-call is the regression this guards against"
    )
    assert "call.socket_for(user.id) is websocket" in cleanup
    # ...but the last device leaving must still release the caller.
    assert "went_offline" in cleanup

    # Presence has to be updated before those decisions are made. Anchor on the
    # socket handler's own teardown -- the first `finally:` in the file belongs
    # to the database helper.
    tail = src.split("[social ws]", 1)[1]
    assert tail.index("hub.remove") < tail.index("calls_for_user"), (
        "hub.remove must run first, or went_offline still counts this socket"
    )
