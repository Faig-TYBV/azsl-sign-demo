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
