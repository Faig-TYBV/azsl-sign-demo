"""
Live delivery over ``/ws/social``: group chat, typing, read receipts, photos.

The REST tests cover what is stored; this covers what is *delivered*, which is a
different failure mode. The bugs this path has had - fanning out to the wrong
people, an ack that never arrives, a socket that dies on a malformed frame - are
all invisible to a request/response test, because none of them touch the
database.

Two real WebSocket connections, authenticated by real session cookies, against
the container shape of the app (``serves_ws=True``).
"""

import os
import sys
import uuid
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("SESSION_SECRET", "test-secret-not-used-in-production")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.web_demo import db as auth_db  # noqa: E402
from src.web_demo import webapp  # noqa: E402

# A real JPEG header, so the server's magic-byte check accepts it.
JPEG = bytes.fromhex("ffd8ffe000104a464946000101") + b"\x00" * 64


@pytest.fixture(scope="module")
def ws_app():
    """The container deployment: pages + REST + both sockets.

    ``build_web_layer`` writes module-level globals that decide what
    ``socialWsUrl`` a page is served with, and test_web_api.py asserts on those
    for the serverless shape. Restoring them on teardown keeps the two modules
    independent of the order pytest happens to run them in.
    """
    auth_db.init_db()
    previous = (webapp._injected_ws_url, webapp._social_ws_url)

    application = FastAPI()
    webapp.build_web_layer(application, serves_ws=True)
    try:
        yield application
    finally:
        webapp._injected_ws_url, webapp._social_ws_url = previous


def signup(app, name):
    client = TestClient(app)
    res = client.post("/api/register", json={
        "full_name": name,
        "email": f"{uuid.uuid4().hex[:10]}@example.com",
        "password": "sirr12345",
        "confirm": "sirr12345",
    })
    assert res.status_code == 200, res.text
    return client, res.json()["user"]


def read_until(ws, wanted, limit=8):
    """The next frame of type ``wanted``, skipping whatever else is queued.

    A socket carries presence, friend notifications and chat on one wire, so
    asserting on "the next frame" makes a test depend on unrelated traffic.
    """
    for _ in range(limit):
        frame = ws.receive_json()
        if frame.get("type") == wanted:
            return frame
    raise AssertionError(f"no {wanted!r} frame arrived within {limit} frames")


@pytest.fixture()
def group_of_two(ws_app):
    """An admin and a member in one group, plus an unrelated third account."""
    admin, admin_user = signup(ws_app, "Sosket Admini")
    member, member_user = signup(ws_app, "Sosket Üzvü")
    outsider, outsider_user = signup(ws_app, "Sosket Kənarı")

    gid = admin.post("/api/groups", json={"name": "Sosket qrupu"}).json()["group"]["id"]
    assert member.post(f"/api/groups/{gid}/join").status_code == 200

    return {
        "gid": gid,
        "admin": admin, "admin_user": admin_user,
        "member": member, "member_user": member_user,
        "outsider": outsider, "outsider_user": outsider_user,
    }


def test_a_group_message_is_acked_and_delivered_live(ws_app, group_of_two):
    g = group_of_two
    with g["admin"].websocket_connect("/ws/social") as ws_a, \
         g["member"].websocket_connect("/ws/social") as ws_b:
        assert ws_a.receive_json()["type"] == "hello"
        assert ws_b.receive_json()["type"] == "hello"

        ws_a.send_json({
            "type": "group:send", "group_id": g["gid"],
            "body": "Salam qrup", "client_id": "c1",
        })

        # The ack is what turns the sender's optimistic bubble into a stored row.
        ack = read_until(ws_a, "group:sent")
        assert ack["client_id"] == "c1"
        assert ack["message"]["body"] == "Salam qrup"
        assert ack["message"]["id"], "the ack must carry the real id"
        # In a group the bubble has to say who is talking.
        assert ack["message"]["sender_name"] == "Sosket Admini"

        delivered = read_until(ws_b, "group:message")
        assert delivered["group_id"] == g["gid"]
        assert delivered["message"]["body"] == "Salam qrup"
        assert delivered["message"]["sender_name"] == "Sosket Admini"


def test_typing_in_a_group_names_who_is_typing(ws_app, group_of_two):
    """"Someone is typing" is useless in a group of thirty."""
    g = group_of_two
    with g["admin"].websocket_connect("/ws/social") as ws_a, \
         g["member"].websocket_connect("/ws/social") as ws_b:
        ws_a.receive_json()
        ws_b.receive_json()

        ws_b.send_json({"type": "group:typing", "group_id": g["gid"], "state": True})
        frame = read_until(ws_a, "group:typing")

        assert frame["from"] == g["member_user"]["id"]
        assert frame["name"] == "Sosket Üzvü"
        assert frame["state"] is True


def test_an_outsider_cannot_post_into_a_group(ws_app, group_of_two):
    g = group_of_two
    with g["outsider"].websocket_connect("/ws/social") as ws_e:
        ws_e.receive_json()
        ws_e.send_json({
            "type": "group:send", "group_id": g["gid"], "body": "içəri girdim",
        })
        refusal = read_until(ws_e, "error")
        assert "üzvü deyilsiniz" in refusal["detail"]

    stored = g["admin"].get(f"/api/groups/{g['gid']}/messages").json()["messages"]
    assert all(m["body"] != "içəri girdim" for m in stored)


def test_a_removed_member_stops_being_able_to_post(ws_app, group_of_two):
    """Membership is re-read per frame rather than cached on the socket.

    A removed member keeps their connection open. Caching membership at connect
    time would let them carry on posting until they happened to reload.
    """
    g = group_of_two
    with g["member"].websocket_connect("/ws/social") as ws_b:
        ws_b.receive_json()

        ws_b.send_json({
            "type": "group:send", "group_id": g["gid"],
            "body": "hələ üzvəm", "client_id": "m1",
        })
        assert read_until(ws_b, "group:sent")["message"]["body"] == "hələ üzvəm"

        g["admin"].delete(f"/api/groups/{g['gid']}/members/{g['member_user']['id']}")

        ws_b.send_json({
            "type": "group:send", "group_id": g["gid"],
            "body": "artıq üzv deyiləm", "client_id": "m2",
        })
        assert "üzvü deyilsiniz" in read_until(ws_b, "error")["detail"]

    stored = g["admin"].get(f"/api/groups/{g['gid']}/messages").json()["messages"]
    bodies = [m["body"] for m in stored]
    assert "hələ üzvəm" in bodies
    assert "artıq üzv deyiləm" not in bodies


def test_a_photo_travels_the_same_path_as_text(ws_app, group_of_two):
    g = group_of_two
    attachment = g["admin"].post(
        "/api/uploads/image?w=640&h=480", content=JPEG,
        headers={"Content-Type": "image/jpeg"},
    ).json()["attachment"]["id"]

    with g["admin"].websocket_connect("/ws/social") as ws_a, \
         g["member"].websocket_connect("/ws/social") as ws_b:
        ws_a.receive_json()
        ws_b.receive_json()

        ws_a.send_json({
            "type": "group:send", "group_id": g["gid"], "body": "",
            "attachment_id": attachment, "client_id": "p1",
        })

        ack = read_until(ws_a, "group:sent")
        assert ack["message"]["kind"] == "image"
        assert ack["message"]["attachment_id"] == attachment
        assert ack["message"]["body"] == "", "a photo needs no caption"

        got = read_until(ws_b, "group:message")
        assert got["message"]["attachment_id"] == attachment
        # ...and the member can now actually fetch it.
        assert g["member"].get(f"/api/attachments/{attachment}").status_code == 200


def test_the_socket_refuses_an_attachment_you_do_not_own(ws_app, group_of_two):
    """Same rule as REST: visibility is derived from where an attachment appears,
    so attaching someone else's id would hand out access to their photo."""
    g = group_of_two
    attachment = g["admin"].post(
        "/api/uploads/image", content=JPEG, headers={"Content-Type": "image/jpeg"},
    ).json()["attachment"]["id"]

    with g["member"].websocket_connect("/ws/social") as ws_b:
        ws_b.receive_json()
        ws_b.send_json({
            "type": "group:send", "group_id": g["gid"], "body": "",
            "attachment_id": attachment, "client_id": "steal",
        })
        assert "Şəkil tapılmadı" in read_until(ws_b, "error")["detail"]

    stored = g["admin"].get(f"/api/groups/{g['gid']}/messages").json()["messages"]
    assert all(m["kind"] != "image" for m in stored)


def test_a_read_receipt_over_the_socket_clears_the_badge(ws_app, group_of_two):
    g = group_of_two
    g["admin"].post(f"/api/groups/{g['gid']}/messages", json={"body": "oxunmamış"})

    before = g["member"].get("/api/groups").json()["groups"]
    assert next(x["unread"] for x in before if x["id"] == g["gid"]) >= 1

    with g["member"].websocket_connect("/ws/social") as ws_b:
        ws_b.receive_json()
        ws_b.send_json({"type": "group:read", "group_id": g["gid"]})
        # A ping/pong round trip proves the read was processed before we look.
        ws_b.send_json({"type": "ping"})
        assert read_until(ws_b, "pong")

    after = g["member"].get("/api/groups").json()["groups"]
    assert next(x["unread"] for x in after if x["id"] == g["gid"]) == 0


def test_a_bad_group_frame_does_not_take_the_socket_down(ws_app, group_of_two):
    """One malformed frame must not cost the user their live connection - that
    would also drop their presence and any call they are in."""
    g = group_of_two
    with g["admin"].websocket_connect("/ws/social") as ws_a:
        ws_a.receive_json()

        # A group that does not exist, then one with no id at all.
        ws_a.send_json({"type": "group:send", "group_id": 999999, "body": "yoxdur"})
        assert read_until(ws_a, "error")

        ws_a.send_json({"type": "group:send", "body": "id yoxdur"})
        ws_a.send_json({"type": "ping"})
        assert read_until(ws_a, "pong"), "the socket should still be alive"

        # ...and it still works afterwards.
        ws_a.send_json({
            "type": "group:send", "group_id": g["gid"],
            "body": "hələ işləyir", "client_id": "ok",
        })
        assert read_until(ws_a, "group:sent")["message"]["body"] == "hələ işləyir"


def test_a_direct_message_photo_is_delivered_live(ws_app, group_of_two):
    g = group_of_two
    admin, member = g["admin"], g["member"]
    admin.post("/api/friends/request", json={"user_id": g["member_user"]["id"]})
    request_id = member.get("/api/friends/requests").json()["incoming"][0]["request_id"]
    member.post("/api/friends/respond", json={"request_id": request_id, "accept": True})

    attachment = admin.post(
        "/api/uploads/image", content=JPEG, headers={"Content-Type": "image/jpeg"},
    ).json()["attachment"]["id"]

    with admin.websocket_connect("/ws/social") as ws_a, \
         member.websocket_connect("/ws/social") as ws_b:
        ws_a.receive_json()
        ws_b.receive_json()

        ws_a.send_json({
            "type": "chat:send", "to": g["member_user"]["id"], "body": "bax",
            "attachment_id": attachment, "client_id": "d1",
        })

        ack = read_until(ws_a, "chat:sent")
        assert ack["message"]["kind"] == "image"
        assert ack["message"]["attachment_id"] == attachment

        got = read_until(ws_b, "chat:message")
        assert got["message"]["attachment_id"] == attachment
        assert member.get(f"/api/attachments/{attachment}").status_code == 200


def test_group_events_reach_the_member_who_was_added(ws_app, group_of_two):
    """Being added to a group has to show up without a reload."""
    g = group_of_two
    with g["outsider"].websocket_connect("/ws/social") as ws_e:
        ws_e.receive_json()

        res = g["admin"].post(
            f"/api/groups/{g['gid']}/members",
            json={"user_id": g["outsider_user"]["id"]},
        )
        assert res.status_code == 200, res.text

        joined = read_until(ws_e, "group:joined")
        assert joined["group"]["id"] == g["gid"]
        assert joined["group"]["role"] == "member"


def test_being_removed_is_pushed_to_the_removed_member(ws_app, group_of_two):
    g = group_of_two
    with g["member"].websocket_connect("/ws/social") as ws_b:
        ws_b.receive_json()

        g["admin"].delete(f"/api/groups/{g['gid']}/members/{g['member_user']['id']}")

        removed = read_until(ws_b, "group:removed")
        assert removed["group_id"] == g["gid"]


def test_renaming_a_group_is_pushed_to_its_members(ws_app, group_of_two):
    g = group_of_two
    with g["member"].websocket_connect("/ws/social") as ws_b:
        ws_b.receive_json()

        g["admin"].patch(f"/api/groups/{g['gid']}", json={"name": "Yeni qrup adı"})

        updated = read_until(ws_b, "group:updated")
        assert updated["group"]["name"] == "Yeni qrup adı"


def test_deleting_a_group_tells_everyone_who_was_in_it(ws_app, group_of_two):
    g = group_of_two
    with g["member"].websocket_connect("/ws/social") as ws_b:
        ws_b.receive_json()

        assert g["admin"].delete(f"/api/groups/{g['gid']}").status_code == 200

        gone = read_until(ws_b, "group:deleted")
        assert gone["group_id"] == g["gid"]
