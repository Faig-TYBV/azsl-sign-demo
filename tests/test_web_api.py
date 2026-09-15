"""
End-to-end HTTP tests for the ML-free web layer: auth, friends, and messaging.

These drive the same app object the Vercel entrypoint builds
(``build_web_layer(serves_ws=False)``), through real requests with real session
cookies — so a route that forgets an auth check fails here.

Closes the "no automated tests for src/web_demo" gap in docs/PROJECT_STATUS.md.
"""

import sys
import uuid
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.web_demo import db as auth_db  # noqa: E402
from src.web_demo.webapp import build_web_layer  # noqa: E402


@pytest.fixture(scope="module")
def app():
    auth_db.init_db()
    application = FastAPI()
    # serves_ws=False is the Vercel shape: pages + auth + social REST, no socket.
    build_web_layer(application, serves_ws=False)
    return application


def new_client(app, name):
    """A signed-up user with their own cookie jar, as a separate browser."""
    client = TestClient(app)
    email = f"{uuid.uuid4().hex[:10]}@example.com"
    res = client.post(
        "/api/register",
        json={"full_name": name, "email": email, "password": "sirr12345", "confirm": "sirr12345"},
    )
    assert res.status_code == 200, res.text
    return client, res.json()["user"]


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #
def test_register_login_logout_cycle(app):
    client = TestClient(app)
    email = f"{uuid.uuid4().hex[:10]}@example.com"

    res = client.post(
        "/api/register",
        json={"full_name": "Test İstifadəçi", "email": email,
              "password": "sirr12345", "confirm": "sirr12345"},
    )
    assert res.status_code == 200
    assert res.json()["user"]["email"] == email
    # The password must never come back out of the API.
    assert "password" not in res.text and "hash" not in res.text

    assert client.get("/api/me").json()["user"]["email"] == email

    client.post("/api/logout")
    assert client.get("/api/me").status_code == 401

    assert client.post("/api/login", json={"email": email, "password": "sirr12345"}).status_code == 200
    assert client.post("/api/login", json={"email": email, "password": "yanlis"}).status_code == 401


def test_duplicate_registration_is_a_conflict(app):
    client = TestClient(app)
    email = f"{uuid.uuid4().hex[:10]}@example.com"
    body = {"full_name": "İkinci Dəfə", "email": email,
            "password": "sirr12345", "confirm": "sirr12345"}

    assert client.post("/api/register", json=body).status_code == 200
    assert TestClient(app).post("/api/register", json=body).status_code == 409


def test_password_is_stored_hashed_not_in_clear(app):
    client, user = new_client(app, "Hash Yoxlaması")
    session = auth_db.SessionLocal()
    try:
        row = auth_db.get_user_by_id(session, user["id"])
        assert row.password_hash != "sirr12345"
        assert row.password_hash.startswith("$argon2")
        assert auth_db.verify_password("sirr12345", row.password_hash) is True
        assert auth_db.verify_password("yanlis", row.password_hash) is False
    finally:
        session.close()


@pytest.mark.parametrize(
    "path,method",
    [
        ("/api/friends", "get"),
        ("/api/friends/requests", "get"),
        ("/api/users/search?q=test", "get"),
        ("/api/rtc-config", "get"),
        ("/api/messages/1", "get"),
    ],
)
def test_social_endpoints_require_a_session(app, path, method):
    anonymous = TestClient(app)
    assert getattr(anonymous, method)(path).status_code == 401


def test_pages_redirect_when_signed_out(app):
    anonymous = TestClient(app)
    for path in ("/app", "/friends"):
        res = anonymous.get(path, follow_redirects=False)
        assert res.status_code == 302
        assert res.headers["location"] == "/login"


def test_friends_page_renders_for_a_signed_in_user(app):
    client, _ = new_client(app, "Səhifə Baxışı")
    res = client.get("/friends")
    assert res.status_code == 200
    assert "__AZSL_CONFIG__" in res.text
    # serves_ws=False and no RECOGNITION_WS_URL -> no socket anywhere.
    assert '"socialWsUrl": null' in res.text


FRONTEND = PROJECT_ROOT / "src" / "web_demo" / "frontend"


@pytest.mark.parametrize("page", ["friends.html", "index.html"])
def test_hidden_attribute_is_made_authoritative(page):
    """Any page that toggles el.hidden must force the attribute to win.

    `hidden` is only a UA-stylesheet rule, and author declarations beat the UA
    origin, so a single `display: grid` anywhere silently turns el.hidden into
    a no-op. That shipped once: `.call-audio-face` (absolute, inset 0, opaque,
    display: grid) stayed painted over the remote video on every video call, so
    both callers saw their own preview and neither saw the other.

    Without the override the failure is invisible in review — the JS looks
    correct — so assert the rule is present rather than trusting the next
    author to remember.
    """
    css = (FRONTEND / page).read_text(encoding="utf-8")
    if ".hidden = " not in css and "hidden>" not in css:
        pytest.skip(f"{page} does not toggle the hidden attribute")

    normalised = css.replace(" ", "").replace("\n", "")
    assert "[hidden]{display:none!important;}" in normalised, (
        f"{page} toggles the `hidden` attribute but never forces it to beat "
        f"author `display` rules. Add: [hidden] {{ display: none !important; }}"
    )


def test_call_overlay_backdrop_cannot_cover_the_remote_video():
    """The audio-only backdrop must be behind the remote video, or hideable.

    It is absolutely positioned with an opaque background over the full
    overlay, so if it ever renders during a video call the remote picture is
    gone. Two independent things keep that from happening; require both.
    """
    html = (FRONTEND / "friends.html").read_text(encoding="utf-8")

    # 1. The backdrop is only revealed for audio calls.
    assert "$('call-audio-face').hidden = !audioOnly;" in html

    # 2. And the hidden attribute actually takes effect (see the test above).
    assert "[hidden] { display: none !important; }" in html

    # 3. The remote video is never itself hidden on a video call.
    assert "$('remote-video').hidden = audioOnly;" in html


def test_remote_video_playback_is_requested_explicitly():
    """autoplay alone is not enough for a stream that carries audio.

    Browsers block unmuted autoplay without fresh user activation, and by the
    time the remote track arrives the click that started the call is several
    awaits old. Without an explicit play() the stream attaches but never
    starts — indistinguishable from a broken connection.
    """
    html = (FRONTEND / "friends.html").read_text(encoding="utf-8")
    assert "function playRemote()" in html
    assert "remote.play()" in html
    # And a visible fallback for when the browser still says no.
    assert 'id="tap-to-play"' in html


def test_socket_url_injection_per_deployment_shape(monkeypatch):
    """The three deploy shapes must each hand the page the right socket URL."""
    from src.web_demo import webapp

    # build_web_layer writes module globals; register them with monkeypatch so
    # they are restored afterwards and cannot leak into other tests.
    monkeypatch.setattr(webapp, "_social_ws_url", webapp._social_ws_url)
    monkeypatch.setattr(webapp, "_injected_ws_url", webapp._injected_ws_url)

    def build(serves_ws, recognition_url):
        monkeypatch.setattr(webapp, "RECOGNITION_WS_URL", recognition_url)
        application = FastAPI()
        webapp.build_web_layer(application, serves_ws=serves_ws)
        return webapp._social_ws_url, webapp._injected_ws_url

    # Container / Cloud Run: one origin serves everything.
    assert build(True, "") == ("", "")

    # Vercel pages + a separate recognition host: the social socket rides along
    # with it, so calling keeps working instead of silently disappearing.
    assert build(False, "wss://azsl-api.fly.dev") == (
        "wss://azsl-api.fly.dev",
        "wss://azsl-api.fly.dev",
    )

    # Vercel alone: nothing to connect to.
    assert build(False, "") == (None, None)


# --------------------------------------------------------------------------- #
# Friends
# --------------------------------------------------------------------------- #
def test_full_friend_and_chat_flow(app):
    alice, alice_user = new_client(app, "Aysel Zəngin")
    bob, bob_user = new_client(app, "Babək Zəngin")

    # Alice finds Bob by name.
    found = alice.get(f"/api/users/search?q={bob_user['full_name'][:6]}").json()["results"]
    match = [u for u in found if u["id"] == bob_user["id"]]
    assert match and match[0]["state"] == "none"

    # She invites him.
    assert alice.post("/api/friends/request", json={"user_id": bob_user["id"]}).json()["status"] == "pending"

    # Until he accepts, neither side is a friend and chat is closed.
    assert alice.get("/api/friends").json()["friends"] == []
    assert alice.get(f"/api/messages/{bob_user['id']}").status_code == 403

    # Bob sees it and accepts.
    incoming = bob.get("/api/friends/requests").json()["incoming"]
    assert len(incoming) == 1
    assert incoming[0]["user"]["id"] == alice_user["id"]
    assert bob.post(
        "/api/friends/respond",
        json={"request_id": incoming[0]["request_id"], "accept": True},
    ).json()["status"] == "accepted"

    # Both now list each other.
    assert [f["id"] for f in alice.get("/api/friends").json()["friends"]] == [bob_user["id"]]
    assert [f["id"] for f in bob.get("/api/friends").json()["friends"]] == [alice_user["id"]]

    # Alice writes; Bob receives it with an unread badge.
    assert alice.post(
        "/api/messages", json={"to": bob_user["id"], "body": "Salam Babək!"}
    ).status_code == 200

    bob_friends = bob.get("/api/friends").json()["friends"]
    assert bob_friends[0]["unread"] == 1

    convo = bob.get(f"/api/messages/{alice_user['id']}").json()["messages"]
    assert [m["body"] for m in convo] == ["Salam Babək!"]

    # Opening the conversation clears the badge.
    assert bob.get("/api/friends").json()["friends"][0]["unread"] == 0

    # Bob replies and Alice sees both sides in order.
    bob.post("/api/messages", json={"to": alice_user["id"], "body": "Salam, xoş gördük!"})
    alice_convo = alice.get(f"/api/messages/{bob_user['id']}").json()["messages"]
    assert [m["body"] for m in alice_convo] == ["Salam Babək!", "Salam, xoş gördük!"]

    # Unfriending closes the conversation again for both of them.
    assert alice.post("/api/friends/remove", json={"user_id": bob_user["id"]}).status_code == 200
    assert alice.get(f"/api/messages/{bob_user['id']}").status_code == 403
    assert bob.get(f"/api/messages/{alice_user['id']}").status_code == 403


def test_a_stranger_cannot_message_or_read(app):
    alice, alice_user = new_client(app, "Aysel Qapalı")
    mallory, mallory_user = new_client(app, "Mallory Kənar")

    assert mallory.post(
        "/api/messages", json={"to": alice_user["id"], "body": "icazəsiz"}
    ).status_code == 403
    assert mallory.get(f"/api/messages/{alice_user['id']}").status_code == 403


def test_cannot_accept_someone_elses_request(app):
    alice, alice_user = new_client(app, "Aysel Sorğu")
    bob, bob_user = new_client(app, "Babək Sorğu")
    mallory, _ = new_client(app, "Mallory Sorğu")

    alice.post("/api/friends/request", json={"user_id": bob_user["id"]})
    request_id = bob.get("/api/friends/requests").json()["incoming"][0]["request_id"]

    # Mallory knows the id but is not the addressee.
    assert mallory.post(
        "/api/friends/respond", json={"request_id": request_id, "accept": True}
    ).status_code == 404
    assert bob.get("/api/friends").json()["friends"] == []


def test_declining_removes_the_request(app):
    alice, alice_user = new_client(app, "Aysel Rədd")
    bob, bob_user = new_client(app, "Babək Rədd")

    alice.post("/api/friends/request", json={"user_id": bob_user["id"]})
    request_id = bob.get("/api/friends/requests").json()["incoming"][0]["request_id"]

    assert bob.post(
        "/api/friends/respond", json={"request_id": request_id, "accept": False}
    ).json()["status"] == "declined"

    assert bob.get("/api/friends/requests").json()["incoming"] == []
    assert alice.get("/api/friends/requests").json()["outgoing"] == []
    assert alice.get("/api/friends").json()["friends"] == []


def test_sender_can_withdraw_a_pending_request(app):
    alice, _ = new_client(app, "Aysel Geri")
    bob, bob_user = new_client(app, "Babək Geri")

    alice.post("/api/friends/request", json={"user_id": bob_user["id"]})
    outgoing = alice.get("/api/friends/requests").json()["outgoing"]
    assert len(outgoing) == 1

    alice.post("/api/friends/cancel", json={"request_id": outgoing[0]["request_id"], "accept": False})
    assert bob.get("/api/friends/requests").json()["incoming"] == []


def test_empty_message_is_rejected(app):
    alice, alice_user = new_client(app, "Aysel Boş")
    bob, bob_user = new_client(app, "Babək Boş")
    alice.post("/api/friends/request", json={"user_id": bob_user["id"]})
    rid = bob.get("/api/friends/requests").json()["incoming"][0]["request_id"]
    bob.post("/api/friends/respond", json={"request_id": rid, "accept": True})

    assert alice.post("/api/messages", json={"to": bob_user["id"], "body": "   "}).status_code == 422


def test_rtc_config_is_served_to_signed_in_users(app):
    client, _ = new_client(app, "Zəng Konfiqi")
    data = client.get("/api/rtc-config").json()
    assert data["iceServers"]
    assert isinstance(data["turn"], bool)
