"""
Profiles, interest-based groups, and image attachments.

Split out of ``social.py`` — which is already carrying friends, direct
messages and the WebRTC signalling — because none of this needs the socket to
work. Every route here is plain request/response, so the whole feature runs on
the serverless deployment as well as the container one. The socket only adds
live delivery, and ``social.py`` owns that: this module reaches into its
:data:`~src.web_demo.social.hub` to push events out, and never the other way
round, which keeps the import one-directional.

The three things that are worth knowing before changing anything here:

* **Group access is a role, not a flag.** ``group_role()`` returns ``admin``,
  ``member`` or ``None``, and every mutating helper in ``db.py`` re-derives it
  from the acting user rather than trusting the request. A route that forgets
  a check therefore still cannot escalate — but check anyway.

* **Images live in the database.** There is no object storage, on purpose: see
  ``Attachment`` in ``db.py``. The browser downscales before uploading and
  :func:`api_upload_image` refuses anything whose *bytes* are not a known
  raster format, regardless of the Content-Type it was sent with.

* **System messages are ordinary rows** with ``kind="system"``. Joining,
  leaving and renaming write one, so opening a group tells you its recent
  history instead of presenting an unexplained set of strangers.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, field_validator

from src.web_demo import db as auth_db
from src.web_demo.deps import get_db, require_user
from src.web_demo.social import hub

groups_router = APIRouter(prefix="/api", tags=["groups"])


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #
class ProfileIn(BaseModel):
    """Every field optional: this is a PATCH, and the profile page sends only
    what the person actually edited."""

    full_name: Optional[str] = None
    bio: Optional[str] = None
    city: Optional[str] = None
    interests: Optional[list[str]] = None

    @field_validator("interests")
    @classmethod
    def _cap_interests(cls, v):
        if v is not None and len(v) > auth_db.MAX_INTERESTS_PER_USER:
            raise ValueError(
                f"Ən çoxu {auth_db.MAX_INTERESTS_PER_USER} maraq seçilə bilər."
            )
        return v


class GroupCreateIn(BaseModel):
    name: str
    description: str = ""
    interests: list[str] = []
    is_open: bool = True


class GroupUpdateIn(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    interests: Optional[list[str]] = None
    is_open: Optional[bool] = None


class GroupMemberIn(BaseModel):
    user_id: int


class GroupRoleIn(BaseModel):
    user_id: int
    role: str


class GroupMessageIn(BaseModel):
    body: str = ""
    attachment_id: Optional[int] = None


# --------------------------------------------------------------------------- #
# Payload shapes
#
# One place each, because the same group appears in four different lists (mine,
# recommended, search results, the socket event) and the page renders all four
# with the same function.
# --------------------------------------------------------------------------- #
def group_payload(
    session,
    group: auth_db.Group,
    user_id: int,
    *,
    unread: int = 0,
    shared: Optional[int] = None,
    member_count: Optional[int] = None,
) -> dict:
    return {
        "id": group.id,
        "name": group.name,
        "description": group.description or "",
        "is_open": bool(group.is_open),
        "created_by": group.created_by,
        "role": auth_db.group_role(session, group.id, user_id),
        "member_count": (
            auth_db.group_member_count(session, group.id)
            if member_count is None
            else member_count
        ),
        "unread": unread,
        "interests": [i.public_dict() for i in auth_db.group_interests(session, group.id)],
        # How many of *your* interests this group is tagged with. Only set on the
        # recommendation list, where it is the reason the row is there at all.
        "shared_interests": shared,
    }


def group_detail(session, group: auth_db.Group, user_id: int) -> dict:
    payload = group_payload(session, group, user_id)
    payload["members"] = [
        {
            "id": u.id,
            "full_name": u.full_name,
            "role": m.role,
            "online": hub.is_online(u.id),
            "joined_at": m.joined_at.isoformat() if m.joined_at else None,
            # Lets the member list offer "add friend" inline, which is the whole
            # point of groups: meet here, become friends afterwards.
            "friend_state": (
                "self" if u.id == user_id
                else auth_db.relationship_state(session, user_id, u.id)
            ),
        }
        for m, u in auth_db.group_members(session, group.id)
    ]
    return payload


async def _broadcast(session, group_id: int, payload: dict, *, exclude: int = 0) -> None:
    """Send an event to every member of a group who has a live socket."""
    for uid in auth_db.group_member_ids(session, group_id):
        if uid == exclude:
            continue
        await hub.send_to_user(uid, payload)


async def _system_message(session, group_id: int, actor_id: int, text: str) -> dict:
    """Record a join/leave/rename notice and push it to everyone present."""
    msg = auth_db.save_group_message(
        session, group_id, actor_id, text, kind="system"
    )
    payload = msg.public_dict("")
    await _broadcast(
        session, group_id, {"type": "group:message", "group_id": group_id, "message": payload}
    )
    return payload


# --------------------------------------------------------------------------- #
# Interests + profiles
# --------------------------------------------------------------------------- #
@groups_router.get("/interests")
async def api_interests(user=Depends(require_user), session=Depends(get_db)):
    """The catalogue, grouped into the picker's three sections."""
    return {"categories": auth_db.interests_grouped(session)}


@groups_router.get("/profile")
async def api_my_profile(user=Depends(require_user), session=Depends(get_db)):
    return {"profile": auth_db.profile_dict(session, user)}


@groups_router.put("/profile")
async def api_update_profile(
    payload: ProfileIn, user=Depends(require_user), session=Depends(get_db)
):
    try:
        updated = auth_db.update_profile(
            session,
            user.id,
            full_name=payload.full_name,
            bio=payload.bio,
            city=payload.city,
            interests=payload.interests,
        )
    except auth_db.SocialError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    return {"profile": auth_db.profile_dict(session, updated)}


@groups_router.get("/profile/{user_id}")
async def api_public_profile(
    user_id: int, user=Depends(require_user), session=Depends(get_db)
):
    """Someone else's profile.

    E-mail is stripped: it is the login identifier and the search key, and
    nothing on the page needs it. Everything else here the person chose to put
    on a profile, so showing it to another signed-in user is the point.
    """
    other = auth_db.get_user_by_id(session, user_id)
    if other is None:
        raise HTTPException(status_code=404, detail="İstifadəçi tapılmadı.")
    profile = auth_db.profile_dict(session, other)
    profile.pop("email", None)
    return {
        "profile": profile,
        "friend_state": (
            "self" if other.id == user.id
            else auth_db.relationship_state(session, user.id, other.id)
        ),
        "shared_interests": auth_db.shared_interest_labels(session, user.id, other.id),
    }


# --------------------------------------------------------------------------- #
# Groups: discovery
# --------------------------------------------------------------------------- #
@groups_router.get("/groups")
async def api_my_groups(user=Depends(require_user), session=Depends(get_db)):
    unread = auth_db.group_unread_counts(session, user.id)
    groups = auth_db.list_user_groups(session, user.id)
    return {
        "groups": [
            group_payload(session, g, user.id, unread=unread.get(g.id, 0)) for g in groups
        ]
    }


@groups_router.get("/groups/recommended")
async def api_recommended_groups(user=Depends(require_user), session=Depends(get_db)):
    rows = auth_db.recommend_groups(session, user.id)
    return {
        "groups": [
            group_payload(session, g, user.id, shared=shared, member_count=members)
            for g, shared, members in rows
        ],
        # The page says "pick some interests" instead of "no groups match you"
        # when the reason for a generic list is an empty profile.
        "has_interests": bool(auth_db.get_user_interests(session, user.id)),
    }


@groups_router.get("/groups/search")
async def api_search_groups(
    q: str = "", user=Depends(require_user), session=Depends(get_db)
):
    found = auth_db.search_groups(session, q)
    return {"groups": [group_payload(session, g, user.id) for g in found]}


# --------------------------------------------------------------------------- #
# Groups: lifecycle
# --------------------------------------------------------------------------- #
@groups_router.post("/groups")
async def api_create_group(
    payload: GroupCreateIn, user=Depends(require_user), session=Depends(get_db)
):
    try:
        group = auth_db.create_group(
            session,
            user.id,
            name=payload.name,
            description=payload.description,
            interests=payload.interests,
            is_open=payload.is_open,
        )
    except auth_db.SocialError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    return {"group": group_detail(session, group, user.id)}


@groups_router.get("/groups/{group_id}")
async def api_group_detail(
    group_id: int, user=Depends(require_user), session=Depends(get_db)
):
    group = auth_db.get_group(session, group_id)
    if group is None:
        raise HTTPException(status_code=404, detail="Qrup tapılmadı.")
    # Readable by anyone signed in: the member list is how you decide whether to
    # join. The messages are not — see api_group_messages.
    return {"group": group_detail(session, group, user.id)}


@groups_router.patch("/groups/{group_id}")
async def api_update_group(
    group_id: int,
    payload: GroupUpdateIn,
    user=Depends(require_user),
    session=Depends(get_db),
):
    before = auth_db.get_group(session, group_id)
    old_name = before.name if before else ""
    try:
        group = auth_db.update_group(
            session,
            user.id,
            group_id,
            name=payload.name,
            description=payload.description,
            interests=payload.interests,
            is_open=payload.is_open,
        )
    except auth_db.GroupError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))

    if payload.name is not None and group.name != old_name:
        await _system_message(
            session, group_id, user.id, f"Qrupun adı dəyişdirildi: {group.name}"
        )
    detail = group_detail(session, group, user.id)
    await _broadcast(
        session, group_id, {"type": "group:updated", "group": {
            "id": group.id, "name": group.name,
            "description": group.description or "", "is_open": bool(group.is_open),
        }}
    )
    return {"group": detail}


@groups_router.delete("/groups/{group_id}")
async def api_delete_group(
    group_id: int, user=Depends(require_user), session=Depends(get_db)
):
    member_ids = auth_db.group_member_ids(session, group_id)
    try:
        auth_db.delete_group(session, user.id, group_id)
    except auth_db.GroupError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))
    for uid in member_ids:
        await hub.send_to_user(uid, {"type": "group:deleted", "group_id": group_id})
    return {"status": "deleted"}


@groups_router.post("/groups/{group_id}/join")
async def api_join_group(
    group_id: int, user=Depends(require_user), session=Depends(get_db)
):
    try:
        auth_db.join_group(session, user.id, group_id)
    except auth_db.GroupError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))
    group = auth_db.get_group(session, group_id)
    await _system_message(session, group_id, user.id, f"{user.full_name} qrupa qoşuldu")
    return {"group": group_detail(session, group, user.id)}


@groups_router.post("/groups/{group_id}/leave")
async def api_leave_group(
    group_id: int, user=Depends(require_user), session=Depends(get_db)
):
    try:
        result = auth_db.leave_group(session, user.id, group_id)
    except auth_db.GroupError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    if not result["deleted"]:
        await _system_message(
            session, group_id, user.id, f"{user.full_name} qrupdan ayrıldı"
        )
        if result["promoted"]:
            promoted = auth_db.get_user_by_id(session, result["promoted"])
            if promoted is not None:
                await _system_message(
                    session, group_id, user.id,
                    f"{promoted.full_name} admin təyin edildi",
                )
        await _broadcast(
            session, group_id, {"type": "group:members-changed", "group_id": group_id}
        )
    return {"status": "left", **result}


# --------------------------------------------------------------------------- #
# Groups: membership administration
# --------------------------------------------------------------------------- #
@groups_router.post("/groups/{group_id}/members")
async def api_add_member(
    group_id: int,
    payload: GroupMemberIn,
    user=Depends(require_user),
    session=Depends(get_db),
):
    try:
        auth_db.add_group_member(session, user.id, group_id, payload.user_id)
    except auth_db.GroupError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))

    added = auth_db.get_user_by_id(session, payload.user_id)
    await _system_message(
        session, group_id, user.id, f"{added.full_name} qrupa əlavə edildi"
    )
    group = auth_db.get_group(session, group_id)
    await hub.send_to_user(
        payload.user_id,
        {"type": "group:joined", "group": group_payload(session, group, payload.user_id)},
    )
    await _broadcast(
        session, group_id, {"type": "group:members-changed", "group_id": group_id}
    )
    return {"group": group_detail(session, group, user.id)}


@groups_router.delete("/groups/{group_id}/members/{member_id}")
async def api_remove_member(
    group_id: int,
    member_id: int,
    user=Depends(require_user),
    session=Depends(get_db),
):
    removed = auth_db.get_user_by_id(session, member_id)
    try:
        auth_db.remove_group_member(session, user.id, group_id, member_id)
    except auth_db.GroupError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))

    if removed is not None:
        await _system_message(
            session, group_id, user.id, f"{removed.full_name} qrupdan çıxarıldı"
        )
    await hub.send_to_user(member_id, {"type": "group:removed", "group_id": group_id})
    await _broadcast(
        session, group_id, {"type": "group:members-changed", "group_id": group_id}
    )
    group = auth_db.get_group(session, group_id)
    return {"group": group_detail(session, group, user.id)}


@groups_router.post("/groups/{group_id}/members/{member_id}/role")
async def api_set_role(
    group_id: int,
    member_id: int,
    payload: GroupRoleIn,
    user=Depends(require_user),
    session=Depends(get_db),
):
    try:
        auth_db.set_group_role(session, user.id, group_id, member_id, payload.role)
    except auth_db.GroupError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))

    changed = auth_db.get_user_by_id(session, member_id)
    if changed is not None:
        await _system_message(
            session,
            group_id,
            user.id,
            f"{changed.full_name} admin təyin edildi"
            if payload.role == "admin"
            else f"{changed.full_name} adminlikdən çıxarıldı",
        )
    await _broadcast(
        session, group_id, {"type": "group:members-changed", "group_id": group_id}
    )
    group = auth_db.get_group(session, group_id)
    return {"group": group_detail(session, group, user.id)}


# --------------------------------------------------------------------------- #
# Groups: messages
# --------------------------------------------------------------------------- #
def _require_member(session, group_id: int, user_id: int) -> auth_db.Group:
    group = auth_db.get_group(session, group_id)
    if group is None:
        raise HTTPException(status_code=404, detail="Qrup tapılmadı.")
    if not auth_db.is_group_member(session, group_id, user_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Bu qrupun üzvü deyilsiniz.",
        )
    return group


@groups_router.get("/groups/{group_id}/messages")
async def api_group_messages(
    group_id: int, user=Depends(require_user), session=Depends(get_db)
):
    _require_member(session, group_id, user.id)
    rows = auth_db.get_group_messages(session, group_id)
    auth_db.mark_group_read(session, group_id, user.id)
    return {"messages": [m.public_dict(name) for m, name in rows]}


@groups_router.post("/groups/{group_id}/messages")
async def api_post_group_message(
    group_id: int,
    payload: GroupMessageIn,
    user=Depends(require_user),
    session=Depends(get_db),
):
    """REST fallback for sending; the socket path is preferred when connected."""
    _require_member(session, group_id, user.id)
    kind = "image" if payload.attachment_id else "text"
    if kind == "image" and not _owns_attachment(session, payload.attachment_id, user.id):
        raise HTTPException(status_code=400, detail="Şəkil tapılmadı.")
    try:
        msg = auth_db.save_group_message(
            session, group_id, user.id, payload.body,
            kind=kind, attachment_id=payload.attachment_id,
        )
    except auth_db.SocialError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    stored = msg.public_dict(user.full_name)
    await _broadcast(
        session,
        group_id,
        {"type": "group:message", "group_id": group_id, "message": stored},
        exclude=user.id,
    )
    return {"message": stored}


# --------------------------------------------------------------------------- #
# Image attachments
# --------------------------------------------------------------------------- #
def _owns_attachment(session, attachment_id: Optional[int], user_id: int) -> bool:
    """Only the uploader may attach an image to a message.

    Without this, anyone could put someone else's attachment id on their own
    message — which both steals the picture and grants their conversation
    partner permission to fetch it, since visibility is derived from the
    messages an attachment appears in.
    """
    if not attachment_id:
        return False
    return auth_db.attachment_owner(session, int(attachment_id)) == user_id


@groups_router.post("/uploads/image")
async def api_upload_image(
    request: Request, user=Depends(require_user), session=Depends(get_db)
):
    """Take an image as the raw request body and return its id.

    Raw body rather than multipart/form-data so the deployment does not need
    python-multipart — the browser can POST a Blob directly, and the only thing
    lost is a filename nobody displays.

    Dimensions come from the client as query parameters purely so the bubble can
    reserve the right space before the picture loads; they are never trusted for
    anything else, and there is no image library here to verify them with.
    """
    limit = auth_db.MAX_ATTACHMENT_BYTES
    mb = limit // (1024 * 1024)

    declared = request.headers.get("content-length")
    if declared:
        try:
            if int(declared) > limit:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Şəkil çox böyükdür (maks. {mb} MB).",
                )
        except ValueError:
            pass  # unparseable header; the streaming cap below still applies

    # Content-Length can lie, so cap while reading rather than trusting it.
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > limit:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Şəkil çox böyükdür (maks. {mb} MB).",
            )
        chunks.append(chunk)
    data = b"".join(chunks)

    def _dimension(key: str) -> int:
        try:
            return max(0, min(20000, int(request.query_params.get(key, 0))))
        except (TypeError, ValueError):
            return 0

    try:
        row = auth_db.save_attachment(
            session, user.id, data, width=_dimension("w"), height=_dimension("h")
        )
    except auth_db.SocialError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    return {
        "attachment": {
            "id": row.id,
            "mime": row.mime,
            "width": row.width,
            "height": row.height,
            "byte_size": row.byte_size,
            "url": f"/api/attachments/{row.id}",
        }
    }


@groups_router.get("/attachments/{attachment_id}")
async def api_attachment(
    attachment_id: int, user=Depends(require_user), session=Depends(get_db)
):
    """Serve an image to someone entitled to see it.

    A 404 rather than a 403 when they are not: an attachment id is a small
    integer, and distinguishing "exists but not yours" from "does not exist"
    would turn the id space into a directory of everyone's photos.
    """
    if not auth_db.attachment_visible_to(session, attachment_id, user.id):
        raise HTTPException(status_code=404, detail="Şəkil tapılmadı.")
    row = auth_db.get_attachment(session, attachment_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Şəkil tapılmadı.")
    return Response(
        content=row.data,
        media_type=row.mime,
        headers={
            # Content is immutable and the id is unguessable enough for a demo;
            # `private` keeps it out of any shared proxy cache.
            "Cache-Control": "private, max-age=31536000, immutable",
            # The mime was decided by sniffing the bytes, so tell the browser not
            # to second-guess it.
            "X-Content-Type-Options": "nosniff",
            "Content-Disposition": "inline",
            "Content-Security-Policy": "default-src 'none'; sandbox",
        },
    )
