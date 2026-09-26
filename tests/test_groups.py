"""
Interest-based groups, profiles, and image attachments.

Like test_social.py these run against throwaway SQLite rather than PostgreSQL,
so the suite needs no services up. The invariants under test are the ones that
would be security or correctness bugs in production:

  * a group's role model — only admins may change anything, and a group can
    never be left without one
  * the unread watermark, which was wrong the first time in a way a clock made
    invisible (see test_unread_is_counted_by_message_id_not_by_clock)
  * attachment visibility, since an attachment id is a small integer and the
    only thing standing between it and a stranger is the check under test
  * the additive column migration, which is what stops a deploy of this change
    from taking an existing database down
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")

from sqlalchemy import create_engine, inspect, select  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from src.web_demo import db as auth_db  # noqa: E402
from src.web_demo import social  # noqa: E402

# A minimal but genuine JPEG header, so the magic-byte sniffer accepts it.
JPEG = bytes.fromhex("ffd8ffe000104a464946000101") + b"\x00" * 64
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


@pytest.fixture()
def session():
    """A fresh in-memory schema, with the interest catalogue seeded."""
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    auth_db.Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    s = Session()
    auth_db.sync_interest_catalogue(s)
    try:
        yield s
    finally:
        s.close()
        engine.dispose()


def make_user(session, name, email):
    return auth_db.create_user(session, full_name=name, email=email, password="hunter2!")


@pytest.fixture()
def people(session):
    return (
        make_user(session, "Aysel Məmmədova", "aysel@example.com"),
        make_user(session, "Babək Quliyev", "babek@example.com"),
        make_user(session, "Cavid Əliyev", "cavid@example.com"),
    )


# --------------------------------------------------------------------------- #
# The interest catalogue
# --------------------------------------------------------------------------- #
def test_catalogue_sync_is_idempotent_and_relabels(session):
    """Runs on every startup, so a second run must add nothing."""
    first = auth_db.list_interests(session)
    added = auth_db.sync_interest_catalogue(session)
    assert added == 0
    assert len(auth_db.list_interests(session)) == len(first)

    # A slug keeps its identity while its label is free to change: user picks and
    # group tags both point at the slug.
    row = session.scalar(select(auth_db.Interest).where(auth_db.Interest.slug == "football"))
    row.label = "stale label"
    session.commit()
    auth_db.sync_interest_catalogue(session)
    session.refresh(row)
    assert row.label == "Futbol"


def test_catalogue_never_drops_an_entry_someone_picked(session, people):
    """A slug removed from INTEREST_CATALOGUE must not empty existing profiles.

    sync_interest_catalogue() only inserts and updates. If it deleted, dropping
    one line from the catalogue would silently wipe it from every profile and
    group that had chosen it.
    """
    a = people[0]
    auth_db.set_user_interests(session, a.id, ["football"])
    before = len(auth_db.list_interests(session))

    auth_db.sync_interest_catalogue(session)

    assert len(auth_db.list_interests(session)) == before
    assert [i.slug for i in auth_db.get_user_interests(session, a.id)] == ["football"]


def test_unknown_slugs_are_dropped_without_losing_the_valid_ones(session, people):
    a = people[0]
    chosen = auth_db.set_user_interests(session, a.id, ["football", "not-a-real-slug", "chess"])
    assert [c.slug for c in chosen] == ["football", "chess"]


def test_too_many_interests_is_refused(session, people):
    a = people[0]
    everything = [slug for slug, _, _, _ in auth_db.INTEREST_CATALOGUE]
    assert len(everything) > auth_db.MAX_INTERESTS_PER_USER
    with pytest.raises(auth_db.SocialError):
        auth_db.set_user_interests(session, a.id, everything)


def test_profile_patch_only_touches_what_was_passed(session, people):
    a = people[0]
    auth_db.update_profile(session, a.id, bio="ilk", city="Bakı", interests=["chess"])
    # No interests argument at all: the existing picks must survive.
    auth_db.update_profile(session, a.id, bio="ikinci")

    refreshed = auth_db.get_user_by_id(session, a.id)
    assert refreshed.bio == "ikinci"
    assert refreshed.city == "Bakı"
    assert [i.slug for i in auth_db.get_user_interests(session, a.id)] == ["chess"]


def test_shared_interests_are_the_intersection(session, people):
    a, b, _ = people
    auth_db.set_user_interests(session, a.id, ["football", "chess", "music"])
    auth_db.set_user_interests(session, b.id, ["chess", "yoga", "music"])
    assert sorted(auth_db.shared_interest_labels(session, a.id, b.id)) == ["Musiqi", "Şahmat"]


# --------------------------------------------------------------------------- #
# Group roles
# --------------------------------------------------------------------------- #
def test_creator_is_admin_and_the_first_member(session, people):
    a = people[0]
    group = auth_db.create_group(session, a.id, name="Şahmat klubu", interests=["chess"])
    assert auth_db.group_role(session, group.id, a.id) == "admin"
    assert auth_db.group_member_count(session, group.id) == 1
    assert [i.slug for i in auth_db.group_interests(session, group.id)] == ["chess"]


def test_only_an_admin_can_change_a_group(session, people):
    a, b, _ = people
    group = auth_db.create_group(session, a.id, name="Şahmat klubu")
    auth_db.join_group(session, b.id, group.id)

    for call in (
        lambda: auth_db.update_group(session, b.id, group.id, name="ələ keçirildi"),
        lambda: auth_db.delete_group(session, b.id, group.id),
        lambda: auth_db.add_group_member(session, b.id, group.id, people[2].id),
        lambda: auth_db.remove_group_member(session, b.id, group.id, a.id),
        lambda: auth_db.set_group_role(session, b.id, group.id, b.id, "admin"),
    ):
        with pytest.raises(auth_db.GroupError):
            call()

    # ...and nothing actually changed.
    assert auth_db.get_group(session, group.id).name == "Şahmat klubu"
    assert auth_db.group_role(session, group.id, b.id) == "member"


def test_a_non_member_cannot_be_a_role(session, people):
    a, b, _ = people
    group = auth_db.create_group(session, a.id, name="Şahmat klubu")
    assert auth_db.group_role(session, group.id, b.id) is None
    assert auth_db.is_group_member(session, group.id, b.id) is False


def test_closed_group_refuses_a_self_join_but_an_admin_can_add(session, people):
    a, b, _ = people
    group = auth_db.create_group(session, a.id, name="Qapalı qrup", is_open=False)

    with pytest.raises(auth_db.GroupError):
        auth_db.join_group(session, b.id, group.id)

    auth_db.add_group_member(session, a.id, group.id, b.id)
    assert auth_db.group_role(session, group.id, b.id) == "member"


def test_joining_twice_is_not_an_error(session, people):
    """The join button can be pressed twice, and two tabs can both press it."""
    a, b, _ = people
    group = auth_db.create_group(session, a.id, name="Açıq qrup")
    auth_db.join_group(session, b.id, group.id)
    auth_db.join_group(session, b.id, group.id)
    assert auth_db.group_member_count(session, group.id) == 2


def test_the_last_admin_cannot_demote_themselves(session, people):
    a, b, _ = people
    group = auth_db.create_group(session, a.id, name="Şahmat klubu")
    auth_db.join_group(session, b.id, group.id)

    with pytest.raises(auth_db.GroupError):
        auth_db.set_group_role(session, a.id, group.id, a.id, "member")

    # With a second admin in place it is allowed.
    auth_db.set_group_role(session, a.id, group.id, b.id, "admin")
    auth_db.set_group_role(session, a.id, group.id, a.id, "member")
    assert auth_db.group_role(session, group.id, a.id) == "member"


def test_last_admin_leaving_promotes_the_longest_standing_member(session, people):
    """Otherwise the group is frozen: nobody left who can change anything.

    There is no route back from a group with no admin — it could not be renamed,
    nobody could be added, and a troublemaker could not be removed.
    """
    a, b, c = people
    group = auth_db.create_group(session, a.id, name="Şahmat klubu")
    auth_db.join_group(session, b.id, group.id)
    auth_db.join_group(session, c.id, group.id)

    result = auth_db.leave_group(session, a.id, group.id)

    assert result["deleted"] is False
    assert result["promoted"] == b.id
    assert auth_db.group_role(session, group.id, b.id) == "admin"
    assert auth_db.group_role(session, group.id, c.id) == "member"


def test_leaving_does_not_promote_while_another_admin_remains(session, people):
    a, b, c = people
    group = auth_db.create_group(session, a.id, name="Şahmat klubu")
    auth_db.join_group(session, b.id, group.id)
    auth_db.join_group(session, c.id, group.id)
    auth_db.set_group_role(session, a.id, group.id, b.id, "admin")

    result = auth_db.leave_group(session, a.id, group.id)

    assert result["promoted"] is None
    assert auth_db.group_role(session, group.id, c.id) == "member"


def test_the_last_person_leaving_closes_the_group(session, people):
    a = people[0]
    group = auth_db.create_group(session, a.id, name="Tək qrup")
    auth_db.save_group_message(session, group.id, a.id, "salam")

    result = auth_db.leave_group(session, a.id, group.id)

    assert result["deleted"] is True
    assert auth_db.get_group(session, group.id) is None
    # Its messages go with it rather than lingering unreachable.
    assert auth_db.get_group_messages(session, group.id) == []


def test_deleting_a_group_removes_its_messages_and_membership(session, people):
    a, b, _ = people
    group = auth_db.create_group(session, a.id, name="Şahmat klubu", interests=["chess"])
    auth_db.join_group(session, b.id, group.id)
    auth_db.save_group_message(session, group.id, a.id, "salam")

    auth_db.delete_group(session, a.id, group.id)

    assert auth_db.get_group(session, group.id) is None
    assert auth_db.group_member_ids(session, group.id) == []
    assert auth_db.list_user_groups(session, b.id) == []


def test_an_admin_removes_others_but_leaves_by_leaving(session, people):
    a, b, _ = people
    group = auth_db.create_group(session, a.id, name="Şahmat klubu")
    auth_db.join_group(session, b.id, group.id)

    with pytest.raises(auth_db.GroupError):
        auth_db.remove_group_member(session, a.id, group.id, a.id)

    auth_db.remove_group_member(session, a.id, group.id, b.id)
    assert auth_db.group_member_ids(session, group.id) == [a.id]


# --------------------------------------------------------------------------- #
# Group messages and unread counting
# --------------------------------------------------------------------------- #
def test_unread_is_counted_by_message_id_not_by_clock(session, people):
    """Regression: a message sent in the same second as a read was invisible.

    The watermark used to be a timestamp set from the application clock, while
    ``created_at`` comes from the database clock and is truncated to whole
    seconds on SQLite. A message written in the same second as the read compared
    as not-newer, so the badge stayed at zero. Ids have no such ambiguity — this
    test sends and reads with no delay at all, which is exactly the case that
    used to fail.
    """
    a, b, _ = people
    group = auth_db.create_group(session, a.id, name="Şahmat klubu")
    auth_db.join_group(session, b.id, group.id)

    auth_db.mark_group_read(session, group.id, a.id)
    auth_db.save_group_message(session, group.id, b.id, "eyni saniyədə")

    assert auth_db.group_unread_counts(session, a.id) == {group.id: 1}

    auth_db.mark_group_read(session, group.id, a.id)
    assert auth_db.group_unread_counts(session, a.id) == {}


def test_your_own_messages_and_system_notices_never_badge(session, people):
    """A badge means "somebody said something to you", not "something happened"."""
    a, b, _ = people
    group = auth_db.create_group(session, a.id, name="Şahmat klubu")
    auth_db.join_group(session, b.id, group.id)
    auth_db.mark_group_read(session, group.id, a.id)

    auth_db.save_group_message(session, group.id, a.id, "mənim mesajım")
    auth_db.save_group_message(session, group.id, b.id, "qrupa qoşuldu", kind="system")

    assert auth_db.group_unread_counts(session, a.id) == {}

    auth_db.save_group_message(session, group.id, b.id, "əsl mesaj")
    assert auth_db.group_unread_counts(session, a.id) == {group.id: 1}


def test_a_new_member_does_not_inherit_the_whole_backlog_as_unread(session, people):
    """Joining should not show 200 unread from before you were there."""
    a, b, _ = people
    group = auth_db.create_group(session, a.id, name="Şahmat klubu")
    for i in range(5):
        auth_db.save_group_message(session, group.id, a.id, f"köhnə {i}")

    auth_db.join_group(session, b.id, group.id)
    # Opening the group is what marks it read; the API does this on first fetch.
    auth_db.mark_group_read(session, group.id, b.id)
    assert auth_db.group_unread_counts(session, b.id) == {}


def test_messages_come_back_oldest_first_with_sender_names(session, people):
    a, b, _ = people
    group = auth_db.create_group(session, a.id, name="Şahmat klubu")
    auth_db.join_group(session, b.id, group.id)
    auth_db.save_group_message(session, group.id, a.id, "birinci")
    auth_db.save_group_message(session, group.id, b.id, "ikinci")

    rows = auth_db.get_group_messages(session, group.id)
    assert [m.body for m, _ in rows] == ["birinci", "ikinci"]
    assert [name for _, name in rows] == ["Aysel Məmmədova", "Babək Quliyev"]


def test_an_image_message_may_have_no_caption_but_text_may_not_be_empty(session, people):
    a = people[0]
    group = auth_db.create_group(session, a.id, name="Şahmat klubu")
    attachment = auth_db.save_attachment(session, a.id, JPEG)

    msg = auth_db.save_group_message(
        session, group.id, a.id, "", kind="image", attachment_id=attachment.id
    )
    assert msg.kind == "image" and msg.body == ""

    with pytest.raises(auth_db.GroupError):
        auth_db.save_group_message(session, group.id, a.id, "   ")
    with pytest.raises(auth_db.GroupError):
        auth_db.save_group_message(session, group.id, a.id, "", kind="image")


# --------------------------------------------------------------------------- #
# Recommendation
# --------------------------------------------------------------------------- #
def test_recommendation_ranks_by_shared_interest_count(session, people):
    a, b, _ = people
    auth_db.set_user_interests(session, a.id, ["football", "chess", "music"])

    none = auth_db.create_group(session, b.id, name="Bağçılıq", interests=["gardening"])
    one = auth_db.create_group(session, b.id, name="Şahmat", interests=["chess"])
    two = auth_db.create_group(session, b.id, name="İdman və şahmat",
                               interests=["football", "chess"])

    ranked = auth_db.recommend_groups(session, a.id)
    ids = [g.id for g, _, _ in ranked]
    assert ids.index(two.id) < ids.index(one.id) < ids.index(none.id)
    shared = {g.id: n for g, n, _ in ranked}
    assert shared[two.id] == 2 and shared[one.id] == 1 and shared[none.id] == 0


def test_recommendation_still_returns_something_with_no_interests(session, people):
    """An empty discovery page guarantees the person never joins anything."""
    a, b, _ = people
    auth_db.create_group(session, b.id, name="Hər hansı qrup", interests=["chess"])

    ranked = auth_db.recommend_groups(session, a.id)
    assert len(ranked) == 1
    assert ranked[0][1] == 0  # no overlap claimed


def test_recommendation_excludes_groups_you_are_already_in(session, people):
    a, b, _ = people
    mine = auth_db.create_group(session, a.id, name="Mənim qrupum")
    theirs = auth_db.create_group(session, b.id, name="Başqa qrup")

    ids = [g.id for g, _, _ in auth_db.recommend_groups(session, a.id)]
    assert ids == [theirs.id]
    assert mine.id not in ids


def test_recommendation_prefers_the_busier_group_on_a_tie(session, people):
    a, b, c = people
    auth_db.set_user_interests(session, a.id, ["chess"])
    quiet = auth_db.create_group(session, b.id, name="Sakit şahmat", interests=["chess"])
    busy = auth_db.create_group(session, b.id, name="Canlı şahmat", interests=["chess"])
    auth_db.join_group(session, c.id, busy.id)

    ids = [g.id for g, _, _ in auth_db.recommend_groups(session, a.id)]
    assert ids.index(busy.id) < ids.index(quiet.id)


def test_group_search_treats_like_wildcards_literally(session, people):
    """Same trap as user search: "%" must not match every group."""
    a = people[0]
    auth_db.create_group(session, a.id, name="Şahmat klubu")
    auth_db.create_group(session, a.id, name="100% futbol")

    assert [g.name for g in auth_db.search_groups(session, "%%")] == []
    assert [g.name for g in auth_db.search_groups(session, "100%")] == ["100% futbol"]


def test_group_search_needs_two_characters(session, people):
    """One character is not a search, it is a table scan with a badge on.

    Deliberately an ASCII query: SQLite's ``lower()`` only folds ASCII, so a
    lower-cased "ş" would not match a stored "Ş" here even though PostgreSQL
    folds it correctly. Same behaviour as search_users(), which is the point —
    the matching rule is the database's, not ours.
    """
    a = people[0]
    auth_db.create_group(session, a.id, name="Sahmat klubu")
    assert auth_db.search_groups(session, "S") == []
    assert len(auth_db.search_groups(session, "sah")) == 1
    assert len(auth_db.search_groups(session, "KLUBU")) == 1


# --------------------------------------------------------------------------- #
# Attachments
# --------------------------------------------------------------------------- #
def test_image_type_comes_from_the_bytes_not_the_caller(session, people):
    a = people[0]
    assert auth_db.save_attachment(session, a.id, JPEG).mime == "image/jpeg"
    assert auth_db.save_attachment(session, a.id, PNG).mime == "image/png"
    assert auth_db.sniff_image_mime(b"RIFF\x00\x00\x00\x00WEBPxx") == "image/webp"
    assert auth_db.sniff_image_mime(b"GIF89a....") == "image/gif"

    # An executable, a script and an SVG are all "not an image" here.
    for payload in (b"MZ\x90\x00", b"<?php evil ?>", b"<svg onload=alert(1)>"):
        assert auth_db.sniff_image_mime(payload) is None
        with pytest.raises(auth_db.SocialError):
            auth_db.save_attachment(session, a.id, payload)


def test_oversized_and_empty_uploads_are_refused(session, people):
    a = people[0]
    with pytest.raises(auth_db.SocialError):
        auth_db.save_attachment(session, a.id, b"")
    too_big = JPEG + b"\x00" * auth_db.MAX_ATTACHMENT_BYTES
    with pytest.raises(auth_db.SocialError):
        auth_db.save_attachment(session, a.id, too_big)


def test_an_attachment_is_visible_only_inside_its_conversation(session, people):
    """An attachment id is a small integer, so this check is the whole fence.

    Without it the id space would be a directory of everyone's photos: increment
    until something comes back.
    """
    a, b, c = people
    attachment = auth_db.save_attachment(session, a.id, JPEG)

    # Uploaded but not yet sent anywhere: only the uploader.
    assert auth_db.attachment_visible_to(session, attachment.id, a.id) is True
    assert auth_db.attachment_visible_to(session, attachment.id, b.id) is False
    assert auth_db.attachment_visible_to(session, attachment.id, c.id) is False

    auth_db.send_friend_request(session, a.id, b.id)
    link = auth_db.friendship_between(session, a.id, b.id)
    auth_db.respond_to_request(session, b.id, link.id, accept=True)
    auth_db.save_message(session, a.id, b.id, "", kind="image", attachment_id=attachment.id)

    # The recipient can see it now; an unrelated third party still cannot.
    assert auth_db.attachment_visible_to(session, attachment.id, b.id) is True
    assert auth_db.attachment_visible_to(session, attachment.id, c.id) is False


def test_a_group_attachment_is_visible_to_members_only(session, people):
    a, b, c = people
    group = auth_db.create_group(session, a.id, name="Foto qrupu")
    auth_db.join_group(session, b.id, group.id)
    attachment = auth_db.save_attachment(session, a.id, JPEG)
    auth_db.save_group_message(
        session, group.id, a.id, "", kind="image", attachment_id=attachment.id
    )

    assert auth_db.attachment_visible_to(session, attachment.id, b.id) is True
    assert auth_db.attachment_visible_to(session, attachment.id, c.id) is False

    # Removed from the group, removed from the photo.
    auth_db.remove_group_member(session, a.id, group.id, b.id)
    assert auth_db.attachment_visible_to(session, attachment.id, b.id) is False


def test_a_missing_attachment_is_not_visible_to_anyone(session, people):
    a = people[0]
    assert auth_db.attachment_visible_to(session, 999999, a.id) is False


def test_attachment_owner_reads_the_owner_without_the_bytes(session, people):
    a, b, _ = people
    attachment = auth_db.save_attachment(session, a.id, JPEG)
    assert auth_db.attachment_owner(session, attachment.id) == a.id
    assert auth_db.attachment_owner(session, attachment.id) != b.id
    assert auth_db.attachment_owner(session, 999999) is None


# --------------------------------------------------------------------------- #
# The additive column migration
# --------------------------------------------------------------------------- #
def test_added_columns_land_on_a_database_that_predates_them(monkeypatch):
    """``create_all`` never alters an existing table.

    Without ``_ensure_columns`` the first request after deploying this change
    would fail with "column users.bio does not exist" on the live database,
    because ``users`` and ``messages`` already exist there. This builds the old
    shape by hand and checks the patch-up, including that rows written before
    the deploy get a value for the new NOT NULL column rather than NULL.
    """
    path = Path(tempfile.mkdtemp()) / "legacy.sqlite3"
    engine = create_engine(f"sqlite+pysqlite:///{path.as_posix()}", future=True)
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE users (id INTEGER PRIMARY KEY, full_name VARCHAR(120) NOT NULL,"
            " email VARCHAR(255) NOT NULL UNIQUE, password_hash VARCHAR(255) NOT NULL,"
            " created_at TIMESTAMP)"
        )
        conn.exec_driver_sql(
            "CREATE TABLE messages (id INTEGER PRIMARY KEY, sender_id INTEGER NOT NULL,"
            " recipient_id INTEGER NOT NULL, body TEXT NOT NULL,"
            " created_at TIMESTAMP, read_at TIMESTAMP)"
        )
        conn.exec_driver_sql(
            "INSERT INTO messages (sender_id, recipient_id, body) VALUES (1, 2, 'köhnə')"
        )

    monkeypatch.setattr(auth_db, "_engine", engine)
    auth_db._ensure_columns()

    inspector = inspect(engine)
    assert {"bio", "city"} <= {c["name"] for c in inspector.get_columns("users")}
    assert {"kind", "attachment_id"} <= {c["name"] for c in inspector.get_columns("messages")}

    with engine.connect() as conn:
        kind = conn.exec_driver_sql("SELECT kind FROM messages").scalar()
    assert kind == "text", "a row written before the deploy must not be left NULL"

    # Running it again is a no-op rather than a duplicate-column error.
    auth_db._ensure_columns()
    engine.dispose()


def test_every_added_column_is_declared_on_its_model():
    """A column in _ADDED_COLUMNS that no model declares would be dead DDL, and
    a model column missing from the list would break the next deploy."""
    for table, column, _ddl in auth_db._ADDED_COLUMNS:
        model = {"users": auth_db.User, "messages": auth_db.Message}[table]
        assert column in model.__table__.columns, f"{table}.{column} not on the model"


# --------------------------------------------------------------------------- #
# The group socket handler
#
# These drive social._handle_group_message directly. It reads membership from
# the database through _in_db, so they use the module's own SessionLocal (the
# file-backed SQLite conftest.py points DATABASE_URL at) rather than the
# in-memory fixture above, which _in_db could not see.
# --------------------------------------------------------------------------- #
class FakeWS:
    def __init__(self):
        self.sent = []

    async def send_text(self, text):
        self.sent.append(text)


@pytest.fixture()
def live_db():
    auth_db.init_db()
    s = auth_db.SessionLocal()
    try:
        yield s
    finally:
        s.close()


def _unique(prefix):
    import uuid

    return f"{prefix}-{uuid.uuid4().hex[:8]}@example.com"


def test_socket_refuses_to_post_into_a_group_you_left(live_db):
    """Membership is re-read per frame, not cached on the socket.

    A removed member keeps their socket open. Cached membership would let them
    carry on posting into the group until they happened to reload the page.
    """
    admin = make_user(live_db, "Qrup Admini", _unique("admin"))
    guest = make_user(live_db, "Keçmiş Üzv", _unique("guest"))
    group = auth_db.create_group(live_db, admin.id, name="Sosket qrupu")
    auth_db.join_group(live_db, guest.id, group.id)

    ws = FakeWS()
    hub_backup = social.hub
    social.hub = social.Hub()
    try:
        asyncio.run(
            social._handle_group_message(
                "group:send",
                {"group_id": group.id, "body": "üzv kimi"},
                guest,
                ws,
            )
        )
        assert [m.body for m, _ in auth_db.get_group_messages(live_db, group.id)][-1] == "üzv kimi"

        auth_db.remove_group_member(live_db, admin.id, group.id, guest.id)
        ws.sent.clear()

        asyncio.run(
            social._handle_group_message(
                "group:send",
                {"group_id": group.id, "body": "çıxarıldıqdan sonra"},
                guest,
                ws,
            )
        )
    finally:
        social.hub = hub_backup

    bodies = [m.body for m, _ in auth_db.get_group_messages(live_db, group.id)]
    assert "çıxarıldıqdan sonra" not in bodies
    assert any("üzvü deyilsiniz" in text for text in ws.sent), ws.sent


def test_socket_refuses_someone_elses_attachment(live_db):
    a = make_user(live_db, "Yükləyən", _unique("owner"))
    b = make_user(live_db, "Oğurlayan", _unique("thief"))
    group = auth_db.create_group(live_db, a.id, name="Foto sosket qrupu")
    auth_db.join_group(live_db, b.id, group.id)
    attachment = auth_db.save_attachment(live_db, a.id, JPEG)

    ws = FakeWS()
    hub_backup = social.hub
    social.hub = social.Hub()
    try:
        asyncio.run(
            social._handle_group_message(
                "group:send",
                {"group_id": group.id, "body": "", "attachment_id": attachment.id},
                b,
                ws,
            )
        )
    finally:
        social.hub = hub_backup

    assert [m.kind for m, _ in auth_db.get_group_messages(live_db, group.id)] == []
    assert any("Şəkil tapılmadı" in text for text in ws.sent), ws.sent


def test_socket_read_receipt_moves_the_watermark(live_db):
    a = make_user(live_db, "Oxuyan", _unique("reader"))
    b = make_user(live_db, "Yazan", _unique("writer"))
    group = auth_db.create_group(live_db, a.id, name="Oxu qrupu")
    auth_db.join_group(live_db, b.id, group.id)
    auth_db.save_group_message(live_db, group.id, b.id, "oxunmamış")
    assert auth_db.group_unread_counts(live_db, a.id).get(group.id) == 1

    hub_backup = social.hub
    social.hub = social.Hub()
    try:
        asyncio.run(
            social._handle_group_message(
                "group:read", {"group_id": group.id}, a, FakeWS()
            )
        )
    finally:
        social.hub = hub_backup

    assert auth_db.group_unread_counts(live_db, a.id).get(group.id) is None


def test_deleting_a_group_reclaims_the_photos_sent_to_it(session, people):
    """Otherwise every photo ever sent to a deleted group sits there forever.

    Nobody can see it - visibility is derived from the messages an attachment
    appears in, and those are gone - so it is pure dead weight on a free
    Postgres tier, which is the one kind of leak that actually runs out.
    """
    a, b, _ = people
    group = auth_db.create_group(session, a.id, name="Foto qrupu")
    auth_db.join_group(session, b.id, group.id)
    kept = auth_db.save_attachment(session, a.id, JPEG)
    doomed = auth_db.save_attachment(session, a.id, PNG)

    # `doomed` only ever appears in the group; `kept` is also in a direct message.
    auth_db.save_group_message(
        session, group.id, a.id, "", kind="image", attachment_id=doomed.id
    )
    auth_db.send_friend_request(session, a.id, b.id)
    link = auth_db.friendship_between(session, a.id, b.id)
    auth_db.respond_to_request(session, b.id, link.id, accept=True)
    auth_db.save_message(session, a.id, b.id, "", kind="image", attachment_id=kept.id)
    auth_db.save_group_message(
        session, group.id, a.id, "", kind="image", attachment_id=kept.id
    )

    auth_db.delete_group(session, a.id, group.id)

    assert auth_db.get_attachment(session, doomed.id) is None
    # Still referenced by the direct message, so it must survive.
    assert auth_db.get_attachment(session, kept.id) is not None


def test_the_last_member_leaving_also_reclaims_its_photos(session, people):
    a = people[0]
    group = auth_db.create_group(session, a.id, name="Tək qrup")
    attachment = auth_db.save_attachment(session, a.id, JPEG)
    auth_db.save_group_message(
        session, group.id, a.id, "", kind="image", attachment_id=attachment.id
    )

    result = auth_db.leave_group(session, a.id, group.id)

    assert result["deleted"] is True
    assert auth_db.get_attachment(session, attachment.id) is None


def test_reclamation_leaves_an_unsent_upload_alone(session, people):
    """An image uploaded but not yet sent belongs to its uploader, not to a
    group, so an unrelated group deletion must not take it."""
    a = people[0]
    group = auth_db.create_group(session, a.id, name="Boş qrup")
    pending = auth_db.save_attachment(session, a.id, JPEG)

    auth_db.delete_group(session, a.id, group.id)

    assert auth_db.get_attachment(session, pending.id) is not None
    assert auth_db.attachment_visible_to(session, pending.id, a.id) is True
