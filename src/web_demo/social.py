"""
Friends, direct messages, and peer-to-peer audio/video calls.

Two halves, deliberately split by what each deployment can host:

  * **REST** (``/api/friends/*``, ``/api/messages/*``) — plain request/response.
    Works everywhere, including the Vercel serverless function. This is the
    source of truth; the socket below is an accelerator, not a separate store.
  * **WebSocket** (``/ws/social``) — live delivery, presence, typing, and the
    signalling that sets up a call. Needs a long-lived process, so it is only
    registered when ``build_web_layer(serves_ws=True)`` (the container backend).
    With no socket the page falls back to polling REST and hides calling.

Media never touches this server. Calls are WebRTC: the browsers exchange an
SDP offer/answer and ICE candidates *through* us, then send audio and video
directly to each other. We relay signalling text only — a few KB per call —
which is why call quality does not depend on the free-tier instance size.

Nothing here imports the ML stack.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Set

from fastapi import APIRouter, Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel, field_validator

from src.web_demo import db as auth_db
from src.web_demo.deps import get_db, require_user, verify_ws_token


# --------------------------------------------------------------------------- #
# WebRTC ICE configuration
# --------------------------------------------------------------------------- #
# STUN alone lets two browsers discover their public address and connect
# directly. That covers most home networks. It does NOT cover symmetric NAT —
# many mobile carriers and corporate firewalls — where the only way through is
# a TURN server that relays the media. TURN costs bandwidth, so it is never
# free and always needs credentials; set these three env vars to enable it:
#
#   TURN_URL         turn:turn.example.com:3478   (or turns: on 5349)
#   TURN_USERNAME    <username>
#   TURN_CREDENTIAL  <password>
#
# Without them, calls still work for most pairs of users and simply fail to
# connect for the rest — see the notes in DEPLOY_RECOGNITION.md.
DEFAULT_STUN = [
    "stun:stun.l.google.com:19302",
    "stun:stun1.l.google.com:19302",
]


def rtc_ice_servers() -> list[dict]:
    """The iceServers array handed to RTCPeerConnection in the browser."""
    stun_env = os.getenv("STUN_URLS", "").strip()
    stun_urls = [u.strip() for u in stun_env.split(",") if u.strip()] or DEFAULT_STUN
    servers: list[dict] = [{"urls": stun_urls}]

    # TURN_URL may hold several comma-separated URLs sharing one credential.
    # Providers hand out a set on purpose — typically UDP :80, TCP :443 and
    # TLS :443 — because networks that block UDP outright are exactly the ones
    # that need a relay. Offering only the UDP entry fails on those.
    raw_turn = [u.strip() for u in os.getenv("TURN_URL", "").split(",") if u.strip()]
    turn_user = os.getenv("TURN_USERNAME", "").strip()
    turn_cred = os.getenv("TURN_CREDENTIAL", "").strip()

    # Provider dashboards list their STUN URL alongside the TURN ones, so it is
    # natural to paste the whole block into TURN_URL. That would put a stun:
    # URL in an entry carrying credentials, and RTCPeerConnection REJECTS that
    # outright — the constructor throws, so every call dies in the browser
    # before a single packet is sent, with no clue as to why. Sort them.
    turn_urls = [u for u in raw_turn if u.startswith(("turn:", "turns:"))]
    stray_stun = [u for u in raw_turn if u.startswith("stun:")]
    unusable = [u for u in raw_turn if u not in turn_urls and u not in stray_stun]

    if stray_stun:
        # Perfectly good servers, just in the wrong variable: use them, without
        # attaching credentials.
        servers[0]["urls"] = list(dict.fromkeys(stun_urls + stray_stun))
    if unusable:
        print(
            f"[rtc] ignoring {len(unusable)} TURN_URL entr(ies) with no turn:/turns:/stun: "
            f"scheme: {unusable}. A malformed entry would make RTCPeerConnection "
            "throw and break every call.",
            flush=True,
        )

    if turn_urls and turn_user and turn_cred:
        servers.append(
            {"urls": turn_urls, "username": turn_user, "credential": turn_cred}
        )
    elif raw_turn and not turn_urls:
        print(
            "[rtc] TURN_URL is set but contains no turn:/turns: URL — no relay "
            "will be offered.",
            flush=True,
        )
    return servers


def turn_configured() -> bool:
    return all(
        os.getenv(k, "").strip() for k in ("TURN_URL", "TURN_USERNAME", "TURN_CREDENTIAL")
    )


# --------------------------------------------------------------------------- #
# Running DB work off the event loop
# --------------------------------------------------------------------------- #
async def _in_db(fn: Callable, *args, **kwargs):
    """Run a sync ``fn(session, *args)`` in a worker thread with its own session.

    SQLAlchemy here is synchronous; calling it directly from the socket handler
    would block the event loop for every other connected user for the duration
    of the query. On a 0.1-CPU free instance that is very visible.
    """

    def run():
        session = auth_db.SessionLocal()
        try:
            return fn(session, *args, **kwargs)
        finally:
            session.close()

    return await asyncio.to_thread(run)


# --------------------------------------------------------------------------- #
# Connection hub
# --------------------------------------------------------------------------- #
# A socket is considered dead if nothing has been received on it for this long.
#
# The page heartbeats every 10 s, which suggests a much shorter window would do.
# It will not: mobile browsers throttle background timers to roughly ONCE PER
# MINUTE, so a phone whose tab is merely dimmed or behind another app pings at
# ~60 s intervals while its socket is perfectly alive. At 45 s that phone was
# marked offline and could not be called — while calls *from* it worked, since
# its tab was necessarily in the foreground. That asymmetry is the bug this
# number caused.
#
# 120 s tolerates one missed throttled ping. It does weaken ghost detection,
# but a false offline is far worse: it breaks calling outright, whereas a ghost
# is already caught by the delivery count on invite and the caller's ring
# timeout.
SOCKET_STALE_AFTER = 120.0


class Hub:
    """Who is connected, and how to reach them.

    A user may hold several sockets at once (phone + laptop + a second tab), so
    every entry is a set. State is per-process and in-memory: with more than one
    instance behind a load balancer, two users on different instances would not
    see each other. Scaling past one instance means putting a Redis pub/sub (or
    equivalent) behind ``send_to_user``; nothing else in this file would change.

    **Presence is based on traffic, not registration.** A registered socket is
    not necessarily a live one: the proxy in front of this app terminates the
    WebSocket and holds its own connection to us, so when a phone loses signal
    or freezes its tab, we can keep a dead entry long after the device is gone.
    Writes to it succeed -- they land in a buffer nobody drains -- so the caller
    is told the callee is ringing while the callee sees nothing at all. Every
    inbound frame calls :meth:`touch`, and anything silent for
    ``SOCKET_STALE_AFTER`` is treated as gone.
    """

    def __init__(self) -> None:
        self._sockets: Dict[int, Set[WebSocket]] = {}
        self._last_seen: Dict[WebSocket, float] = {}
        self._lock = asyncio.Lock()

    def touch(self, ws: WebSocket) -> None:
        """Record that this socket just proved it is alive."""
        self._last_seen[ws] = time.monotonic()

    def _is_fresh(self, ws: WebSocket) -> bool:
        seen = self._last_seen.get(ws)
        return seen is not None and (time.monotonic() - seen) < SOCKET_STALE_AFTER

    def _live_sockets(self, user_id: int) -> list[WebSocket]:
        return [ws for ws in self._sockets.get(user_id, ()) if self._is_fresh(ws)]

    async def add(self, user_id: int, ws: WebSocket) -> bool:
        """Register a socket. Returns True if this user was previously offline."""
        async with self._lock:
            sockets = self._sockets.setdefault(user_id, set())
            was_offline = not any(self._is_fresh(s) for s in sockets)
            sockets.add(ws)
            self.touch(ws)
            return was_offline

    async def remove(self, user_id: int, ws: WebSocket) -> bool:
        """Drop a socket. Returns True if the user now has none left."""
        async with self._lock:
            self._last_seen.pop(ws, None)
            sockets = self._sockets.get(user_id)
            if not sockets:
                return False
            sockets.discard(ws)
            if not sockets:
                self._sockets.pop(user_id, None)
                return True
            return not any(self._is_fresh(s) for s in sockets)

    def is_online(self, user_id: int) -> bool:
        return bool(self._live_sockets(user_id))

    def describe(self, user_id: int) -> str:
        """'2 live / 3 registered' — for log lines that have to explain a
        delivery failure after the fact, from a hosted log viewer."""
        registered = len(self._sockets.get(user_id, ()))
        live = len(self._live_sockets(user_id))
        return f"{live} live / {registered} registered"


    def online_ids(self) -> Set[int]:
        return {uid for uid in self._sockets if self._live_sockets(uid)}

    async def send_to_user(
        self, user_id: int, payload: dict, *, exclude: Optional[WebSocket] = None
    ) -> int:
        """Fan a payload out to this user's live sockets. Returns how many.

        Silent sockets are skipped rather than written to, so the count is a
        usable answer to "did this actually reach them?" -- which is what the
        call invite relies on to avoid ringing into a void.
        """
        text = json.dumps(payload, ensure_ascii=False)
        sent = 0
        for ws in self._live_sockets(user_id):
            if ws is exclude:
                continue
            try:
                await ws.send_text(text)
                sent += 1
            except Exception:
                # A socket that is already gone will be cleaned up by its own
                # handler's finally block; losing this one send is harmless.
                pass
        return sent

    async def send_to_socket(self, ws: WebSocket, payload: dict) -> None:
        try:
            await ws.send_text(json.dumps(payload, ensure_ascii=False))
        except Exception:
            pass


hub = Hub()


# --------------------------------------------------------------------------- #
# Call registry
# --------------------------------------------------------------------------- #
@dataclass
class Call:
    """One in-flight call. Exists from invite until either side hangs up."""

    call_id: str
    caller_id: int
    callee_id: int
    media: str  # "video" | "audio"
    state: str = "ringing"  # "ringing" | "active"
    # The exact sockets in the call. The caller's is known at invite; the
    # callee's is pinned when they answer, so if they had three tabs ringing,
    # offer/answer/ICE go only to the tab that actually picked up.
    caller_ws: Optional[WebSocket] = None
    callee_ws: Optional[WebSocket] = None

    def peer_of(self, user_id: int) -> Optional[int]:
        if user_id == self.caller_id:
            return self.callee_id
        if user_id == self.callee_id:
            return self.caller_id
        return None

    def socket_for(self, user_id: int) -> Optional[WebSocket]:
        if user_id == self.caller_id:
            return self.caller_ws
        if user_id == self.callee_id:
            return self.callee_ws
        return None

    def involves(self, user_id: int) -> bool:
        return user_id in (self.caller_id, self.callee_id)


class CallRegistry:
    def __init__(self) -> None:
        self._calls: Dict[str, Call] = {}

    def create(self, caller_id: int, callee_id: int, media: str, caller_ws: WebSocket) -> Call:
        call = Call(
            call_id=uuid.uuid4().hex,
            caller_id=caller_id,
            callee_id=callee_id,
            media=media,
            caller_ws=caller_ws,
        )
        self._calls[call.call_id] = call
        return call

    def get(self, call_id: str) -> Optional[Call]:
        return self._calls.get(call_id)

    def drop(self, call_id: str) -> Optional[Call]:
        return self._calls.pop(call_id, None)

    def calls_for_user(self, user_id: int) -> list[Call]:
        return [c for c in self._calls.values() if c.involves(user_id)]

    def has_active_call(self, user_id: int) -> bool:
        return any(c.involves(user_id) for c in self._calls.values())


calls = CallRegistry()


# --------------------------------------------------------------------------- #
# REST request models
# --------------------------------------------------------------------------- #
class FriendRequestIn(BaseModel):
    user_id: int


class RespondIn(BaseModel):
    request_id: int
    accept: bool


class MessageIn(BaseModel):
    to: int
    body: str

    @field_validator("body")
    @classmethod
    def _body_ok(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("Mesaj boş ola bilməz.")
        if len(v) > auth_db.MAX_MESSAGE_LENGTH:
            raise ValueError(
                f"Mesaj çox uzundur (maks. {auth_db.MAX_MESSAGE_LENGTH} simvol)."
            )
        return v


# --------------------------------------------------------------------------- #
# REST routes
# --------------------------------------------------------------------------- #
social_router = APIRouter(prefix="/api", tags=["social"])


def _friend_payload(user: auth_db.User, unread: int = 0) -> dict:
    return {
        "id": user.id,
        "full_name": user.full_name,
        "email": user.email,
        "online": hub.is_online(user.id),
        "unread": unread,
    }


@social_router.get("/rtc-config")
async def api_rtc_config(user=Depends(require_user)):
    """ICE servers for RTCPeerConnection, plus whether a relay is available.

    ``turn`` being false means calls between two users on restrictive networks
    (most commonly mobile data) may fail to connect — the UI warns about that
    rather than leaving them staring at a frozen "connecting" spinner.
    """
    return {"iceServers": rtc_ice_servers(), "turn": turn_configured()}


@social_router.get("/friends")
async def api_friends(user=Depends(require_user), session=Depends(get_db)):
    friends = auth_db.list_friends(session, user.id)
    unread = auth_db.unread_counts(session, user.id)
    return {
        "friends": [_friend_payload(f, unread.get(f.id, 0)) for f in friends],
        "online": sorted(hub.online_ids()),
    }


@social_router.get("/friends/requests")
async def api_friend_requests(user=Depends(require_user), session=Depends(get_db)):
    incoming = auth_db.list_incoming_requests(session, user.id)
    outgoing = auth_db.list_outgoing_requests(session, user.id)
    return {
        "incoming": [
            {"request_id": link.id, "user": _friend_payload(u)} for link, u in incoming
        ],
        "outgoing": [
            {"request_id": link.id, "user": _friend_payload(u)} for link, u in outgoing
        ],
    }


@social_router.get("/users/search")
async def api_user_search(q: str = "", user=Depends(require_user), session=Depends(get_db)):
    found = auth_db.search_users(session, user.id, q)
    return {
        "results": [
            {
                **_friend_payload(u),
                "state": auth_db.relationship_state(session, user.id, u.id),
            }
            for u in found
        ]
    }


@social_router.post("/friends/request")
async def api_send_request(
    payload: FriendRequestIn, user=Depends(require_user), session=Depends(get_db)
):
    try:
        link = auth_db.send_friend_request(session, user.id, payload.user_id)
    except auth_db.FriendshipError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    other = auth_db.get_user_by_id(session, payload.user_id)
    if link.status == "accepted":
        # Crossing invites resolved into a friendship — tell both sides.
        await hub.send_to_user(other.id, {"type": "friend:accepted", "user": _friend_payload(user)})
        await hub.send_to_user(user.id, {"type": "friend:accepted", "user": _friend_payload(other)})
    else:
        await hub.send_to_user(
            other.id,
            {
                "type": "friend:request",
                "request": {"request_id": link.id, "user": _friend_payload(user)},
            },
        )
    return {"status": link.status, "request_id": link.id}


@social_router.post("/friends/respond")
async def api_respond_request(
    payload: RespondIn, user=Depends(require_user), session=Depends(get_db)
):
    try:
        link = auth_db.respond_to_request(session, user.id, payload.request_id, payload.accept)
    except auth_db.FriendshipError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    if link is not None:
        requester = auth_db.get_user_by_id(session, link.requester_id)
        await hub.send_to_user(
            requester.id, {"type": "friend:accepted", "user": _friend_payload(user)}
        )
        return {"status": "accepted", "user": _friend_payload(requester)}
    return {"status": "declined"}


@social_router.post("/friends/cancel")
async def api_cancel_request(
    payload: RespondIn, user=Depends(require_user), session=Depends(get_db)
):
    try:
        auth_db.cancel_request(session, user.id, payload.request_id)
    except auth_db.FriendshipError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    return {"status": "cancelled"}


@social_router.post("/friends/remove")
async def api_remove_friend(
    payload: FriendRequestIn, user=Depends(require_user), session=Depends(get_db)
):
    try:
        auth_db.remove_friend(session, user.id, payload.user_id)
    except auth_db.FriendshipError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    await hub.send_to_user(payload.user_id, {"type": "friend:removed", "user_id": user.id})
    return {"status": "removed"}


@social_router.get("/messages/{friend_id}")
async def api_conversation(
    friend_id: int, user=Depends(require_user), session=Depends(get_db)
):
    if not auth_db.are_friends(session, user.id, friend_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Yalnız dostlarla yazışa bilərsiniz."
        )
    messages = auth_db.get_conversation(session, user.id, friend_id)
    auth_db.mark_conversation_read(session, user.id, friend_id)
    await hub.send_to_user(friend_id, {"type": "chat:read", "by": user.id})
    return {"messages": [m.public_dict() for m in messages]}


@social_router.post("/messages")
async def api_send_message(
    payload: MessageIn, user=Depends(require_user), session=Depends(get_db)
):
    """REST fallback for sending. The socket path is preferred when connected."""
    if not auth_db.are_friends(session, user.id, payload.to):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Yalnız dostlarla yazışa bilərsiniz."
        )
    try:
        msg = auth_db.save_message(session, user.id, payload.to, payload.body)
    except auth_db.FriendshipError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    await hub.send_to_user(
        payload.to, {"type": "chat:message", "peer_id": user.id, "message": msg.public_dict()}
    )
    return {"message": msg.public_dict()}


# --------------------------------------------------------------------------- #
# WebSocket: presence, live chat, and call signalling
# --------------------------------------------------------------------------- #
async def _notify_friends_presence(user_id: int, online: bool) -> None:
    friends = await _in_db(auth_db.list_friends, user_id)
    payload = {"type": "presence", "user_id": user_id, "online": online}
    for friend in friends:
        await hub.send_to_user(friend.id, payload)


async def _end_call(call: Call, ended_by: int, reason: str) -> None:
    """Tear a call down and tell the other side exactly once."""
    calls.drop(call.call_id)
    peer_id = call.peer_of(ended_by)
    if peer_id is None:
        return
    payload = {"type": "call:ended", "call_id": call.call_id, "reason": reason}
    peer_ws = call.socket_for(peer_id)
    if peer_ws is not None:
        await hub.send_to_socket(peer_ws, payload)
    else:
        # Callee never answered, so no socket is pinned yet: stop every tab ringing.
        await hub.send_to_user(peer_id, payload)


async def _handle_call_message(
    msg_type: str, message: dict, user, ws: WebSocket
) -> None:
    """Invite / answer / hang-up plus the SDP and ICE relay.

    Every branch re-checks that the sender is actually a party to the call it
    names, so a crafted call_id cannot be used to push SDP at a stranger.
    """
    if msg_type == "call:invite":
        target_id = int(message.get("to", 0))
        media = "audio" if message.get("media") == "audio" else "video"

        if not await _in_db(auth_db.are_friends, user.id, target_id):
            print(f"[call] REFUSED uid={user.id} -> uid={target_id}: not friends", flush=True)
            await hub.send_to_socket(
                ws, {"type": "error", "detail": "Yalnız dostlarınıza zəng edə bilərsiniz."}
            )
            return
        if not hub.is_online(target_id):
            await hub.send_to_socket(
                ws, {"type": "call:rejected", "call_id": None, "reason": "offline"}
            )
            return
        if calls.has_active_call(target_id):
            await hub.send_to_socket(
                ws, {"type": "call:rejected", "call_id": None, "reason": "busy"}
            )
            return

        print(
            f"[call] invite uid={user.id} -> uid={target_id} media={media} "
            f"callee_sockets=({hub.describe(target_id)}) "
            f"caller_sockets=({hub.describe(user.id)})",
            flush=True,
        )

        call = calls.create(user.id, target_id, media, ws)
        delivered = await hub.send_to_user(
            target_id,
            {
                "type": "call:incoming",
                "call_id": call.call_id,
                "media": media,
                "from": {"id": user.id, "full_name": user.full_name},
            },
        )
        # is_online() only says a socket is registered, not that anything is
        # listening. A phone that loses signal or gets its tab frozen leaves a
        # half-open connection that we keep for up to a ping timeout, so the
        # invite above can vanish into a dead pipe while the caller waits
        # forever. Trust the delivery count, not the registry.
        print(f"[call] invite {call.call_id[:8]} delivered to {delivered} socket(s)", flush=True)
        if delivered == 0:
            calls.drop(call.call_id)
            await hub.send_to_socket(
                ws, {"type": "call:rejected", "call_id": None, "reason": "offline"}
            )
            return
        await hub.send_to_socket(
            ws, {"type": "call:ringing", "call_id": call.call_id, "to": target_id}
        )
        return

    call = calls.get(str(message.get("call_id", "")))
    if call is None or not call.involves(user.id):
        return

    if msg_type == "call:accept":
        if user.id != call.callee_id or call.state != "ringing":
            return
        call.state = "active"
        call.callee_ws = ws
        # Silence this user's other tabs, which are still ringing.
        await hub.send_to_user(
            user.id,
            {"type": "call:ended", "call_id": call.call_id, "reason": "answered_elsewhere"},
            exclude=ws,
        )
        # The caller creates the SDP offer once it hears this.
        await hub.send_to_socket(
            call.caller_ws, {"type": "call:accepted", "call_id": call.call_id}
        )
        return

    if msg_type == "call:reject":
        await _end_call(call, user.id, "rejected")
        return

    if msg_type == "call:end":
        await _end_call(call, user.id, "ended")
        return

    if msg_type == "call:caption":
        # A line of text one participant produced by signing or speaking, for
        # the other to read. This is the accessibility path: it lets a Deaf and
        # a hearing person hold a conversation without either of them typing.
        peer_id = call.peer_of(user.id)
        if peer_id is None:
            return
        text = str(message.get("text", "")).strip()
        if not text:
            return
        source = message.get("source")
        source = source if source in ("sign", "speech") else "sign"

        # Persisted as an ordinary message, so the conversation survives the
        # call. Someone who relies on captions should be able to scroll back
        # through what was said rather than having it disappear on hang-up.
        stored = None
        try:
            stored = await _in_db(auth_db.save_message, user.id, peer_id, text)
        except auth_db.FriendshipError:
            # Too long or empty after trimming: still worth showing live.
            pass

        payload = {
            "type": "call:caption",
            "call_id": call.call_id,
            "from": user.id,
            "text": text,
            "source": source,
        }
        if stored is not None:
            payload["message"] = stored.public_dict()

        # To the peer's call socket if they have one, otherwise every socket
        # they hold — they may have the chat open on another device.
        peer_ws = call.socket_for(peer_id)
        if peer_ws is not None:
            await hub.send_to_socket(peer_ws, payload)
        else:
            await hub.send_to_user(peer_id, payload)
        # Echo to the sender's other tabs so the transcript stays consistent.
        await hub.send_to_user(user.id, payload, exclude=ws)
        return

    if msg_type in ("call:offer", "call:answer", "call:ice"):
        peer_id = call.peer_of(user.id)
        peer_ws = call.socket_for(peer_id) if peer_id is not None else None
        if peer_ws is None:
            return
        relay = {"type": msg_type, "call_id": call.call_id}
        if msg_type == "call:ice":
            relay["candidate"] = message.get("candidate")
        else:
            relay["sdp"] = message.get("sdp")
        await hub.send_to_socket(peer_ws, relay)


async def social_websocket(websocket: WebSocket) -> None:
    """``/ws/social`` — one socket per open tab, authenticated like ``/ws``."""
    uid = websocket.session.get("uid")
    if uid is None:
        uid = verify_ws_token(websocket.query_params.get("token", ""))
    if uid is None:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    user = await _in_db(auth_db.get_user_by_id, int(uid))
    if user is None:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.accept()
    became_online = await hub.add(user.id, websocket)
    print(
        f"[social] connect uid={user.id} ({user.full_name}) "
        f"sockets=({hub.describe(user.id)}) first={became_online}",
        flush=True,
    )

    friends = await _in_db(auth_db.list_friends, user.id)
    await hub.send_to_socket(
        websocket,
        {
            "type": "hello",
            "user": {"id": user.id, "full_name": user.full_name},
            "online": sorted(f.id for f in friends if hub.is_online(f.id)),
        },
    )
    if became_online:
        await _notify_friends_presence(user.id, True)

    try:
        while True:
            raw = await websocket.receive_text()
            # Anything arriving on this socket -- including the client's 10 s
            # heartbeat -- is what keeps it counted as live. See Hub's docstring.
            hub.touch(websocket)
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg_type = message.get("type", "")

            if msg_type == "ping":
                await hub.send_to_socket(websocket, {"type": "pong"})
                continue

            if msg_type.startswith("call:"):
                await _handle_call_message(msg_type, message, user, websocket)
                continue

            if msg_type == "chat:send":
                target_id = int(message.get("to", 0))
                body = str(message.get("body", ""))
                if not await _in_db(auth_db.are_friends, user.id, target_id):
                    await hub.send_to_socket(
                        websocket,
                        {"type": "error", "detail": "Yalnız dostlarla yazışa bilərsiniz."},
                    )
                    continue
                try:
                    msg = await _in_db(auth_db.save_message, user.id, target_id, body)
                except auth_db.FriendshipError as exc:
                    await hub.send_to_socket(websocket, {"type": "error", "detail": str(exc)})
                    continue

                payload = msg.public_dict()
                # Ack the sender (so their optimistic bubble gets a real id)...
                await hub.send_to_socket(
                    websocket,
                    {
                        "type": "chat:sent",
                        "client_id": message.get("client_id"),
                        "peer_id": target_id,
                        "message": payload,
                    },
                )
                # ...mirror to the sender's other tabs...
                await hub.send_to_user(
                    user.id,
                    {"type": "chat:message", "peer_id": target_id, "message": payload},
                    exclude=websocket,
                )
                # ...and deliver.
                await hub.send_to_user(
                    target_id,
                    {"type": "chat:message", "peer_id": user.id, "message": payload},
                )
                continue

            if msg_type == "chat:read":
                peer_id = int(message.get("from", 0))
                await _in_db(auth_db.mark_conversation_read, user.id, peer_id)
                await hub.send_to_user(peer_id, {"type": "chat:read", "by": user.id})
                continue

            if msg_type == "chat:typing":
                peer_id = int(message.get("to", 0))
                if await _in_db(auth_db.are_friends, user.id, peer_id):
                    await hub.send_to_user(
                        peer_id,
                        {
                            "type": "chat:typing",
                            "from": user.id,
                            "state": bool(message.get("state")),
                        },
                    )
                continue

    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001 - never take the process down
        print(f"[social ws] {type(exc).__name__}: {exc}", flush=True)
    finally:
        # A dropped tab must not leave the other side ringing forever.
        for call in calls.calls_for_user(user.id):
            if call.socket_for(user.id) is websocket or call.state == "ringing":
                await _end_call(call, user.id, "disconnected")
        went_offline = await hub.remove(user.id, websocket)
        print(
            f"[social] disconnect uid={user.id} sockets=({hub.describe(user.id)}) "
            f"now_offline={went_offline}",
            flush=True,
        )
        if went_offline:
            await _notify_friends_presence(user.id, False)


def register_social_ws(app: FastAPI) -> None:
    """Attach ``/ws/social``. Only called on hosts that can hold a socket open."""
    app.add_api_websocket_route("/ws/social", social_websocket)
