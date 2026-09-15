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


def test_incoming_call_busy_check_ignores_stale_call_id():
    """A leftover call.id must not silently swallow every future call.

    call.id is set the moment call:incoming arrives, and survives a dropped
    socket because the server's call:ended cannot reach a socket that is
    already gone. Phones drop the socket whenever the tab is backgrounded, so
    guarding on call.id meant a phone stopped ringing permanently after its
    first interrupted call — with no error anywhere.

    Busy must therefore mean a live peer connection or a modal already on
    screen, never merely a non-null id.
    """
    html = (FRONTEND / "friends.html").read_text(encoding="utf-8")

    assert "const busy = call.pc || !$('incoming-modal').hidden;" in html, (
        "the busy check must test a live connection or a visible modal"
    )
    assert "if (call.pc || call.id) { wsSend({ type: 'call:reject'" not in html, (
        "the call.id-based busy check is the regression this guards against"
    )
    # And a reconnect must clear whatever the dead socket left behind.
    assert "if (!call.pc) {" in html and "clearing stale call state after reconnect" in html


def test_socket_state_is_visible_and_recovers_on_mobile():
    """Silent disconnects are the hardest failure to report; make them visible."""
    html = (FRONTEND / "friends.html").read_text(encoding="utf-8")

    assert 'id="conn-pill"' in html
    assert "setConnState('live', 'Canlı')" in html
    assert "setConnState('down', 'Bağlantı kəsildi')" in html
    # Backgrounded phone tabs must not wait out the backoff before reconnecting.
    assert "visibilitychange" in html


def test_conversation_panel_is_wired_for_both_input_modes():
    """Sign->text and speech->text, the accessibility feature of the product."""
    html = (FRONTEND / "friends.html").read_text(encoding="utf-8")

    # Both directions of the conversation exist.
    assert 'id="mode-sign"' in html and 'id="mode-speech"' in html
    assert 'id="sign-alphabet"' in html and 'id="sign-word"' in html

    # Sign recognition re-uses the workspace's server-side models over /ws
    # rather than opening a second camera.
    assert "recognitionEndpoint" in html
    assert "'/ws?token='" in html or "'/ws'" in html
    assert "$('local-video')" in html, "frames must come from the existing call stream"

    # Speech uses the browser engine: free, no API key.
    assert "webkitSpeechRecognition" in html
    # The language is selectable, defaulting to Azerbaijani. It was hard-coded
    # until Chrome was seen silently falling back to Russian for az-AZ.
    assert "rec.lang = conv.speechLang" in html
    assert "speechLang: 'az-AZ'" in html
    assert 'id="speech-lang"' in html
    # ...and a wrong-language result must not reach the other person.
    assert "function scriptMismatch" in html

    # Recognised text is composed before sending, not sent blind.
    assert "function sendCaption()" in html
    assert 'id="conv-input"' in html


def test_conversation_resources_are_released_when_the_call_ends():
    """A forgotten mode would keep streaming camera frames and hold the mic."""
    html = (FRONTEND / "friends.html").read_text(encoding="utf-8")
    reset = html.split("function resetCall()", 1)[1].split("function ", 1)[0]

    assert "stopSpeech()" in reset, "the microphone must be released on hang-up"
    assert "closeRecognition()" in reset, "frame streaming must stop on hang-up"
    assert "conv.mode = 'off'" in reset


def test_word_mode_does_not_stream_frames_while_idle():
    """The backend ignores frames outside a trial; sending them wastes CPU.

    The free instance is 0.1 CPU and is already encoding WebRTC video, so
    pushing ~10 JPEGs a second that the server will discard is not free.
    """
    html = (FRONTEND / "friends.html").read_text(encoding="utf-8")
    assert "conv.wordState !== 'COUNTDOWN' && conv.wordState !== 'RECORDING'" in html


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


def test_health_reports_turn_config_without_leaking_it(app, monkeypatch):
    """The flag must be checkable without an account, but reveal nothing."""
    from src.web_demo import webapp

    anonymous = TestClient(app)

    monkeypatch.setenv("TURN_URL", "turn:relay.example.com:3478")
    monkeypatch.setenv("TURN_USERNAME", "secret-user")
    monkeypatch.setenv("TURN_CREDENTIAL", "secret-pass")
    body = anonymous.get("/health")
    assert body.status_code == 200
    data = body.json()
    assert data["ok"] is True
    assert data["turn"] is True
    # The whole point: a boolean, never the credentials.
    assert "secret-user" not in body.text
    assert "secret-pass" not in body.text
    assert "relay.example.com" not in body.text

    monkeypatch.delenv("TURN_CREDENTIAL")
    assert anonymous.get("/health").json()["turn"] is False, (
        "a partial config must report false — it is ignored at runtime, so "
        "reporting true would be actively misleading"
    )


def test_rtc_config_is_served_to_signed_in_users(app):
    client, _ = new_client(app, "Zəng Konfiqi")
    data = client.get("/api/rtc-config").json()
    assert data["iceServers"]
    assert isinstance(data["turn"], bool)


def test_landmarker_bundle_is_served_for_browser_detection():
    """The browser cannot run MediaPipe without this file.

    It lives under src/models/ rather than the frontend directory, so it needs
    its own route; without it the page silently falls back to shipping JPEGs,
    which is the bandwidth problem this replaced.
    """
    from src.web_demo import webapp

    assert webapp.HAND_LANDMARKER_TASK.is_file(), "the .task bundle is missing"
    assert webapp.HAND_LANDMARKER_TASK.stat().st_size > 1_000_000


def test_browser_feature_extraction_is_wired_into_the_page():
    html = (FRONTEND / "friends.html").read_text(encoding="utf-8")

    # Both pure-compute modules must load before the page script uses them.
    assert '<script src="/static/js/azsl_alphabet.js"></script>' in html
    assert '<script src="/static/js/azsl_features.js"></script>' in html

    # Detection happens locally, and alphabet never reaches the network.
    assert "initLocalRecognition" in html
    assert "detectForVideo" in html
    assert "classifyAlphabetLocally" in html

    # Word mode sends vectors, not images.
    assert "frameFeatures126(hands)" in html
    assert "features: feats.map" in html

    # ...but the JPEG path survives for browsers that cannot do the above.
    assert "toDataURL('image/jpeg'" in html, "the server-side fallback was removed"
    assert "if (local.ready) {" in html


def test_server_validates_browser_supplied_features():
    """Vectors from the page are untrusted input reaching a model directly."""
    backend = (PROJECT_ROOT / "src" / "web_demo" / "backend.py").read_text(encoding="utf-8")

    assert 'message.get("features")' in backend
    assert ".reshape(126)" in backend, "length must be enforced"
    assert "np.all(np.isfinite(feat_126))" in backend, (
        "a NaN would poison the normalizer and yield a confident-looking "
        "prediction from nonsense"
    )
    # Alphabet is client-side; the server should not accept features for it.
    assert 'state.active_mode != "word"' in backend


def test_word_recording_has_a_single_implementation():
    """Both input paths must share the trial state machine, or they drift."""
    backend = (PROJECT_ROOT / "src" / "web_demo" / "backend.py").read_text(encoding="utf-8")
    assert backend.count("async def _record_word_frame") == 1
    assert backend.count("await _record_word_frame(") == 2, (
        "expected exactly the JPEG path and the landmark path to call it"
    )


def test_call_screen_splits_responsively_when_conversation_opens():
    """Video and captions share the screen instead of one covering the other.

    In a sign conversation you are reading the person as well as the text, so
    the captions cannot sit on top of the video. Which axis it splits on is a
    CSS decision driven by the window's shape, not a device guess in JS.
    """
    html = (FRONTEND / "friends.html").read_text(encoding="utf-8")

    # The video half is a real container, so the picture-in-picture preview and
    # the banner anchor to it rather than to the whole screen.
    assert 'id="call-stage"' in html
    assert ".call-stage {" in html

    # Three named regions, wired consistently.
    for area in ("stage", "conv", "controls"):
        assert f"grid-area: {area};" in html, f"no element claims the '{area}' region"

    # Stacked by default (portrait/narrow)...
    assert '"stage"\n        "conv"\n        "controls";' in html
    # ...and side by side when the window is wide and landscape.
    assert "(min-width: 900px) and (orientation: landscape)" in html
    assert '"stage conv"' in html
    # A short landscape window must not end up with two flat rows.
    assert "(max-height: 560px) and (orientation: landscape)" in html

    # The split is applied and removed with the panel.
    assert "classList.toggle('with-conv', conv.open)" in html
    assert "overlay.classList.remove('with-conv')" in html


def test_captions_area_grows_instead_of_being_capped():
    """Given half the screen, the transcript should use it."""
    html = (FRONTEND / "friends.html").read_text(encoding="utf-8")
    captions = html.split(".conv-captions {", 1)[1].split("}", 1)[0]
    assert "flex: 1" in captions and "min-height: 0" in captions
    assert "max-height: 28vh" not in captions, (
        "the old fixed cap would waste the space the split now provides"
    )


def test_interim_speech_is_shown_while_it_is_still_being_heard():
    """An empty box while you are mid-sentence reads as the app missing you.

    Chrome does not finalise a speech result until it detects a pause, so
    writing only final results left several seconds of speaking with nothing on
    screen. The provisional text is now rendered in place and firms up when the
    engine commits.
    """
    html = (FRONTEND / "friends.html").read_text(encoding="utf-8")

    assert "function renderCompose()" in html
    assert "committed: ''" in html, "confirmed and provisional text must be separable"
    # The interim result goes to the compose line, not just the status strip.
    onresult = html.split("rec.onresult = (ev) => {", 1)[1].split("rec.onerror", 1)[0]
    assert "renderCompose()" in onresult
    # And it is visibly provisional.
    assert "classList.toggle('interim'" in html
    assert ".conv-compose .field.interim" in html


def test_send_uses_the_visible_text_including_provisional():
    """Waiting for finalisation before you may send would reintroduce the delay."""
    html = (FRONTEND / "friends.html").read_text(encoding="utf-8")
    send = html.split("function sendCaption() {", 1)[1].split("addCaptionLine", 1)[0]
    assert "input.value.trim()" in send, (
        "sending must read what is on screen, so provisional text can be sent "
        "without waiting for the engine to commit"
    )


def test_typing_by_hand_is_not_overwritten_by_recognition():
    """Recognition renders into the same field the user can correct."""
    html = (FRONTEND / "friends.html").read_text(encoding="utf-8")
    assert "$('conv-input').addEventListener('input'" in html
    handler = html.split("$('conv-input').addEventListener('input'", 1)[1][:260]
    assert "conv.committed = e.target.value" in handler, (
        "a hand edit must become the confirmed text, or the next render "
        "discards it"
    )


def test_hand_landmarks_are_drawn_over_the_local_preview():
    """Seeing the skeleton separates "read the hand wrongly" from "never saw it".

    Those need opposite responses — move into frame, versus the gesture was
    ambiguous — and without the overlay they are indistinguishable.
    """
    html = (FRONTEND / "friends.html").read_text(encoding="utf-8")

    assert 'id="hand-overlay"' in html
    assert "HAND_CONNECTIONS" in html
    assert "function drawHandOverlay" in html
    # Drawn from the same detection that feeds recognition, so what is shown is
    # what the model is actually being given.
    pump = html.split("if (local.ready) {", 1)[1][:400]
    assert "drawHandOverlay(hands)" in pump

    # It must mirror with the preview, or a landmark at the fingertip lands on
    # the wrong side of the screen.
    overlay_css = html.split("#hand-overlay {", 1)[1].split("}", 1)[0]
    assert "transform: scaleX(-1)" in overlay_css
    assert "pointer-events: none" in overlay_css


def test_alphabet_uses_the_real_classifier_api():
    """Guards the call that threw on every frame.

    predictGesture(landmarks, mirrorX, velocity, ...) dereferences velocity.x,
    so a one-argument call raised TypeError for every frame and the mode
    produced nothing at all.
    """
    html = (FRONTEND / "friends.html").read_text(encoding="utf-8")
    # Anchor on a top-level declaration: splitting on "function " alone cuts at
    # the inline `lm.map(function (p) ...)` inside the block being examined.
    block = html.split("function classifyAlphabetLocally", 1)[1].split("\nfunction ", 1)[0]

    assert "A.predictGesture(" not in block, (
        "predictGesture needs a velocity argument and reorders control "
        "gestures; call the pieces directly as predict_frame() does"
    )
    assert "A.detectControlGesture(coords)" in block, "control gestures come first"
    assert "A.classifyHierarchical(coords, { x: 0, y: 0 })" in block
    # The module wants {x, y, z} objects, not the [x, y, z] arrays it is given.
    assert "{ x: p[0], y: p[1], z: p[2] }" in block


def test_degenerate_scaler_variance_is_guarded_in_both_implementations():
    """One feature was constant in training; dividing by its ~4e-7 std
    amplified a sub-microscopic float difference into a 0.23 confidence gap
    between the browser and the server."""
    py_src = (PROJECT_ROOT / "src" / "inference" / "alphabet_classifier.py").read_text(
        encoding="utf-8"
    )
    js_src = (FRONTEND / "js" / "azsl_alphabet.js").read_text(encoding="utf-8")

    assert "SCALER_MIN_STD" in py_src and "SCALER_MIN_STD" in js_src
    assert "std < SCALER_MIN_STD" in py_src
    assert "s > SCALER_MIN_STD ? s : 1" in js_src
    assert "np.where(std == 0, 1.0, std)" not in py_src, (
        "an exact-zero test misses a std of 4.4e-07, which is the case that bit"
    )
