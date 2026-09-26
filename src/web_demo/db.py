"""
PostgreSQL-backed user store for the AzSL web demo (registration + login).

Everything auth-related lives here:
  * SQLAlchemy engine / session factory built from DATABASE_URL (.env)
  * the ``users`` table model
  * ``init_db()`` — create the database + tables on first run
  * password hashing (argon2) and the small query helpers the routes need

The rest of the app (backend.py) only ever calls the module-level functions;
it never touches SQLAlchemy directly.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, InvalidHashError
from dotenv import load_dotenv
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    delete,
    func,
    inspect,
    literal,
    or_,
    select,
    update,
)
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session, sessionmaker
from sqlalchemy.pool import NullPool

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://postgres:postgres@localhost:5432/azsl_demo",
)

# argon2id with library defaults — fine for an interactive login form.
_ph = PasswordHasher()

# On Vercel (or any serverless host) the process is frozen/thawed between
# requests, so a pooled connection goes stale — use NullPool and open a fresh
# connection per request. A normal long-lived server keeps the pool.
_serverless = bool(os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"))
_engine_kwargs = dict(pool_pre_ping=True, future=True)
if _serverless:
    _engine_kwargs = dict(poolclass=NullPool, future=True)
    # psycopg3 auto-prepares statements after a few uses; a transaction-mode
    # pooler (Neon/Supabase "-pooler" URL, PgBouncer) can't carry those across
    # connections. Disabling it lets either the pooled or the direct URL work.
    if DATABASE_URL.startswith(("postgresql+psycopg:", "postgresql:")):
        _engine_kwargs["connect_args"] = {"prepare_threshold": None}

_engine = create_engine(DATABASE_URL, **_engine_kwargs)
SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False, future=True)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    full_name: Mapped[str] = mapped_column(String(120), nullable=False)
    # Stored lower-cased; unique so a duplicate registration is a clean 409.
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    # Profile, all optional. Filled in on /profile (or right after registering)
    # and used to recommend groups - see recommend_groups().
    bio: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    city: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def public_dict(self) -> dict:
        return {
            "id": self.id,
            "full_name": self.full_name,
            "email": self.email,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Friendship(Base):
    """
    One row per relationship, not per direction.

    ``requester_id`` is whoever sent the invite and ``addressee_id`` whoever
    received it; that asymmetry is kept after acceptance only so the UI can say
    who asked. Every "are these two friends?" query therefore has to look at
    both column orders — see :func:`friendship_between`.

    A declined request deletes its row rather than storing a "declined" state,
    so the pair can try again later. Blocking is deliberately not modelled.
    """

    __tablename__ = "friendships"
    __table_args__ = (
        # One relationship per ordered pair. The reverse pair is prevented in
        # send_friend_request(), which upgrades a crossing invite to a mutual
        # accept instead of inserting a second row.
        UniqueConstraint("requester_id", "addressee_id", name="uq_friendship_pair"),
        CheckConstraint("requester_id <> addressee_id", name="ck_friendship_not_self"),
        Index("ix_friendship_addressee_status", "addressee_id", "status"),
        Index("ix_friendship_requester_status", "requester_id", "status"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    requester_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    addressee_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    # "pending" | "accepted"
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    responded_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class Message(Base):
    """A single direct message between two users."""

    __tablename__ = "messages"
    __table_args__ = (
        CheckConstraint("sender_id <> recipient_id", name="ck_message_not_self"),
        # Conversation lookups are always "these two people, newest first".
        Index("ix_message_pair_created", "sender_id", "recipient_id", "created_at"),
        Index("ix_message_unread", "recipient_id", "read_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    sender_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    recipient_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    body: Mapped[str] = mapped_column(Text, nullable=False)
    # "text" | "image". An image message carries a caption in body (possibly
    # empty) and the picture in attachment_id.
    kind: Mapped[str] = mapped_column(
        String(16), nullable=False, default="text", server_default="text"
    )
    # Soft reference to attachments.id - see Attachment's docstring for why it
    # is not a ForeignKey.
    attachment_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    read_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    def public_dict(self) -> dict:
        return {
            "id": self.id,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "body": self.body,
            "kind": self.kind or "text",
            "attachment_id": self.attachment_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "read_at": self.read_at.isoformat() if self.read_at else None,
        }


class Interest(Base):
    """One entry from the fixed interest catalogue below.

    A table rather than free text on the user row, because group
    recommendation is an *overlap* query: "how many of this group's tags does
    this person share?". Free-text hobbies would make that a fuzzy string
    problem and the recommendations meaningless ("futbol" vs "Futbol " vs
    "fudbol"). The trade-off is that people can only pick from a list, which
    for a demo of this size is the right way round.
    """

    __tablename__ = "interests"

    id: Mapped[int] = mapped_column(primary_key=True)
    slug: Mapped[str] = mapped_column(String(48), unique=True, index=True, nullable=False)
    label: Mapped[str] = mapped_column(String(64), nullable=False)
    # "hobby" | "activity" | "topic" — only used to group the picker's chips.
    category: Mapped[str] = mapped_column(String(16), nullable=False, default="topic")
    emoji: Mapped[str] = mapped_column(String(8), nullable=False, default="")
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    def public_dict(self) -> dict:
        return {
            "slug": self.slug,
            "label": self.label,
            "category": self.category,
            "emoji": self.emoji,
        }


class UserInterest(Base):
    """Which catalogue entries a person picked. Composite PK = no duplicates."""

    __tablename__ = "user_interests"

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    interest_id: Mapped[int] = mapped_column(
        ForeignKey("interests.id", ondelete="CASCADE"), primary_key=True
    )


class Group(Base):
    """An interest-based room where people who are not yet friends can talk.

    ``is_open`` is the one access control: open groups are joinable by anyone
    who finds them, closed groups only by someone an admin adds. Both are
    discoverable — hiding a group from search would defeat the point, which is
    meeting people.
    """

    __tablename__ = "groups"
    __table_args__ = (
        Index("ix_group_name", "name"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_by: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    is_open: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class GroupInterest(Base):
    """The group's tags — the other half of the recommendation overlap."""

    __tablename__ = "group_interests"

    group_id: Mapped[int] = mapped_column(
        ForeignKey("groups.id", ondelete="CASCADE"), primary_key=True
    )
    interest_id: Mapped[int] = mapped_column(
        ForeignKey("interests.id", ondelete="CASCADE"), primary_key=True
    )


class GroupMember(Base):
    """Membership plus role, plus the read watermark that drives the badge.

    Per-message read receipts would need a row per member per message; in a
    30-person group that is 30x the write volume for a badge. One watermark per
    member gives the same badge from one row.

    The watermark is a **message id, not a timestamp**. Timestamps looked
    natural and were wrong: ``created_at`` comes from the database clock
    (``func.now()``) while a Python-side ``datetime.now()`` watermark comes from
    the application clock, and on SQLite ``CURRENT_TIMESTAMP`` is truncated to
    whole seconds. A message written in the same second as a read therefore
    compared as not-newer and was never counted as unread. Ids are a single
    monotonic sequence, so ``id > last_read_message_id`` is exact regardless of
    clock skew or precision.
    """

    __tablename__ = "group_members"
    __table_args__ = (
        Index("ix_group_member_user", "user_id"),
        CheckConstraint("role in ('admin', 'member')", name="ck_group_role"),
    )

    group_id: Mapped[int] = mapped_column(
        ForeignKey("groups.id", ondelete="CASCADE"), primary_key=True
    )
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False, default="member")
    joined_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    # Highest group_messages.id this member has seen. 0 = has read nothing.
    last_read_message_id: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )


class GroupMessage(Base):
    """A message in a group. ``kind`` mirrors :class:`Message`.

    ``system`` rows are the joined/left/renamed notices; they have no sender
    the UI should attribute, so ``sender_id`` is still recorded (who caused it)
    but the bubble is rendered centred and plain.
    """

    __tablename__ = "group_messages"
    __table_args__ = (
        Index("ix_group_message_group_created", "group_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    group_id: Mapped[int] = mapped_column(
        ForeignKey("groups.id", ondelete="CASCADE"), nullable=False
    )
    sender_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    body: Mapped[str] = mapped_column(Text, nullable=False, default="")
    # "text" | "image" | "system"
    kind: Mapped[str] = mapped_column(String(16), nullable=False, default="text")
    # Soft reference to attachments.id — see Attachment's docstring.
    attachment_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def public_dict(self, sender_name: str = "") -> dict:
        return {
            "id": self.id,
            "group_id": self.group_id,
            "sender_id": self.sender_id,
            "sender_name": sender_name,
            "body": self.body,
            "kind": self.kind or "text",
            "attachment_id": self.attachment_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Attachment(Base):
    """An uploaded image, stored as bytes in the database.

    Object storage (S3, Cloudinary) would be the usual answer, and swapping to
    it means changing only :func:`save_attachment` and the serving route. It is
    not the answer *here*: every such service wants an account and a card, and
    this project deliberately has no paid dependency. A free Postgres tier has
    room for a few thousand chat photos once the browser has downscaled them,
    which is the honest limit of this approach — see MAX_ATTACHMENT_BYTES.

    ``attachment_id`` on the two message tables is a plain integer, not a
    foreign key, so that adding the column to an already-deployed table is a
    single ALTER with no constraint to validate. Integrity is enforced in
    :func:`attachment_visible_to` and a dangling id renders as a missing image
    rather than an error.
    """

    __tablename__ = "attachments"
    __table_args__ = (
        Index("ix_attachment_owner", "owner_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    owner_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    mime: Mapped[str] = mapped_column(String(40), nullable=False)
    byte_size: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    width: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    height: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    data: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


# --------------------------------------------------------------------------- #
# Schema / database bootstrap
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# The interest catalogue
#
# Order matters: it is the order the chips appear in the picker, and the
# sort_order column is derived from the position here. Slugs are the stable
# identity — a label can be reworded freely, a slug never changes, because a
# user's picks and a group's tags both point at it.
#
# Entries are only ever added or relabelled, never removed: sync_interest_
# catalogue() does not delete, so dropping one from this list leaves existing
# profiles and groups untouched rather than silently emptying them.
# --------------------------------------------------------------------------- #
INTEREST_CATALOGUE: tuple[tuple[str, str, str, str], ...] = (
    # (slug, label, category, emoji)
    ("music", "Musiqi", "hobby", "\U0001F3B5"),
    ("instruments", "Musiqi alətləri", "hobby", "\U0001F3B8"),
    ("film", "Film və serial", "hobby", "\U0001F3AC"),
    ("books", "Kitab və oxu", "hobby", "\U0001F4DA"),
    ("games", "Video oyunlar", "hobby", "\U0001F3AE"),
    ("cooking", "Yemək bişirmə", "hobby", "\U0001F373"),
    ("photography", "Fotoqrafiya", "hobby", "\U0001F4F7"),
    ("drawing", "Rəsm və dizayn", "hobby", "\U0001F3A8"),
    ("handcraft", "Əl işləri", "hobby", "\U0001F9F5"),
    ("gardening", "Bağçılıq", "hobby", "\U0001F331"),
    ("pets", "Heyvanlar", "hobby", "\U0001F43E"),
    ("coding", "Proqramlaşdırma", "hobby", "\U0001F4BB"),
    ("cars", "Avtomobillər", "hobby", "\U0001F697"),

    ("football", "Futbol", "activity", "\u26BD"),
    ("basketball", "Basketbol", "activity", "\U0001F3C0"),
    ("volleyball", "Voleybol", "activity", "\U0001F3D0"),
    ("table_tennis", "Stolüstü tennis", "activity", "\U0001F3D3"),
    ("swimming", "Üzgüçülük", "activity", "\U0001F3CA"),
    ("fitness", "Fitness və zal", "activity", "\U0001F3CB"),
    ("running", "Qaçış", "activity", "\U0001F3C3"),
    ("cycling", "Velosiped", "activity", "\U0001F6B4"),
    ("hiking", "Təbiət gəzintisi", "activity", "\U0001F3DE"),
    ("chess", "Şahmat", "activity", "\u265F"),
    ("dance", "Rəqs", "activity", "\U0001F483"),
    ("yoga", "Yoqa", "activity", "\U0001F9D8"),
    ("martial_arts", "Döyüş idmanları", "activity", "\U0001F94B"),

    ("sign_language", "İşarə dili", "topic", "\U0001F91F"),
    ("deaf_culture", "Karlar icması", "topic", "\U0001F450"),
    ("education", "Təhsil", "topic", "\U0001F393"),
    ("languages", "Dil öyrənmə", "topic", "\U0001F5E3"),
    ("science", "Elm", "topic", "\U0001F52C"),
    ("technology", "Texnologiya", "topic", "\U0001F4F1"),
    ("travel", "Səyahət", "topic", "\u2708"),
    ("history", "Tarix", "topic", "\U0001F3DB"),
    ("business", "Biznes və startap", "topic", "\U0001F4C8"),
    ("health", "Sağlamlıq", "topic", "\U0001F49A"),
    ("psychology", "Psixologiya", "topic", "\U0001F9E0"),
    ("volunteering", "Könüllülük", "topic", "\U0001F91D"),
    ("theatre", "Teatr və səhnə", "topic", "\U0001F3AD"),
    ("environment", "Ekologiya", "topic", "\u267B"),
)

INTEREST_CATEGORIES: tuple[tuple[str, str], ...] = (
    ("hobby", "Hobbi"),
    ("activity", "İdman və fəaliyyət"),
    ("topic", "Mövzular"),
)

MAX_INTERESTS_PER_USER = 15
MAX_INTERESTS_PER_GROUP = 6
MAX_GROUP_NAME = 64
MAX_GROUP_DESCRIPTION = 400
MAX_BIO_LENGTH = 400
MAX_CITY_LENGTH = 64
# How many groups one account may create. Not a security boundary — just a
# brake on someone filling the discovery list with empty rooms.
MAX_GROUPS_CREATED = 25

# 3 MB. The browser downscales to 1600px/JPEG before upload, which puts a
# phone photo at 150-400 KB, so this only catches the pathological cases. Each
# byte lands in the database, so it is also the per-image cost of the
# no-object-storage decision in Attachment's docstring.
MAX_ATTACHMENT_BYTES = 3 * 1024 * 1024

# Magic bytes, checked instead of trusting Content-Type. A file that claims to
# be an image but is not would be served back with an image mime type, and
# some browsers still sniff — so the declared type is not evidence.
_IMAGE_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)


def sniff_image_mime(data: bytes) -> Optional[str]:
    """The real image type of ``data``, or None if it is not a known image."""
    for signature, mime in _IMAGE_SIGNATURES:
        if data.startswith(signature):
            return mime
    # WEBP is "RIFF" + 4 size bytes + "WEBP".
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def sync_interest_catalogue(session: Session) -> int:
    """Insert missing catalogue entries and refresh the labels of existing ones.

    Idempotent, so it can run on every startup. Returns how many rows it added.
    """
    existing = {i.slug: i for i in session.scalars(select(Interest)).all()}
    added = 0
    for position, (slug, label, category, emoji) in enumerate(INTEREST_CATALOGUE):
        row = existing.get(slug)
        if row is None:
            session.add(
                Interest(
                    slug=slug, label=label, category=category,
                    emoji=emoji, sort_order=position,
                )
            )
            added += 1
        else:
            row.label = label
            row.category = category
            row.emoji = emoji
            row.sort_order = position
    session.commit()
    return added


# Columns added to tables that already exist on deployed databases.
# ``Base.metadata.create_all`` creates missing *tables* but never alters an
# existing one, so without this the first request after a deploy fails with
# "column users.bio does not exist". Each entry is (table, column, DDL). The
# DDL carries its own DEFAULT so rows written before the deploy get a value
# instead of NULL.
_ADDED_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("users", "bio", "TEXT"),
    ("users", "city", "VARCHAR(64)"),
    ("messages", "kind", "VARCHAR(16) NOT NULL DEFAULT 'text'"),
    ("messages", "attachment_id", "INTEGER"),
)


def _ensure_columns() -> None:
    """Add any missing column from :data:`_ADDED_COLUMNS`. Idempotent.

    Written by hand rather than with Alembic: four columns do not justify a
    migration tool and its version table, and this runs on a host where the
    only deploy step is "start the process".
    """
    inspector = inspect(_engine)
    try:
        tables = set(inspector.get_table_names())
    except (OperationalError, ProgrammingError):
        return

    for table, column, ddl in _ADDED_COLUMNS:
        if table not in tables:
            continue  # create_all will build it complete
        present = {c["name"] for c in inspector.get_columns(table)}
        if column in present:
            continue
        with _engine.begin() as conn:
            conn.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
        print(f"[db] added column {table}.{column}", flush=True)



class DatabaseUnavailable(RuntimeError):
    """Raised when the auth database can't be reached or prepared."""


def _create_database_if_missing() -> None:
    """
    Connect to the server's ``postgres`` maintenance DB and ``CREATE DATABASE``
    the target if it doesn't exist yet. A no-op when it already exists.
    """
    url = make_url(DATABASE_URL)
    target = url.database
    admin_url = url.set(database="postgres")
    admin_engine = create_engine(admin_url, isolation_level="AUTOCOMMIT", future=True)
    try:
        with admin_engine.connect() as conn:
            exists = conn.exec_driver_sql(
                "SELECT 1 FROM pg_database WHERE datname = %s", (target,)
            ).first()
            if not exists:
                # identifier can't be parameterised; validate then interpolate.
                if not re.fullmatch(r"[A-Za-z0-9_]+", target or ""):
                    raise DatabaseUnavailable(
                        f"Refusing to create database with unsafe name {target!r}"
                    )
                conn.exec_driver_sql(f'CREATE DATABASE "{target}"')
    finally:
        admin_engine.dispose()


def init_db() -> None:
    """
    Prepare the store: create the database if needed, create any missing table,
    add any column a previous deploy did not have, and seed the interest
    catalogue. Raises :class:`DatabaseUnavailable` with an actionable message on
    any connection/permission problem.

    Safe to call on every startup - each step is idempotent.
    """
    try:
        try:
            Base.metadata.create_all(_engine)
        except OperationalError as exc:
            # Most common first-run case: database doesn't exist yet.
            if 'does not exist' in str(exc).lower():
                _create_database_if_missing()
                Base.metadata.create_all(_engine)
            else:
                raise
        # create_all never alters a table that already exists, so a deploy that
        # adds a column to `users` or `messages` needs this to follow it.
        _ensure_columns()
        session = SessionLocal()
        try:
            sync_interest_catalogue(session)
        finally:
            session.close()
    except (OperationalError, ProgrammingError) as exc:
        raise DatabaseUnavailable(_friendly_db_error(exc)) from exc


def _friendly_db_error(exc: Exception) -> str:
    raw = str(getattr(exc, "orig", exc)).strip().splitlines()[0]
    return (
        "Cannot reach the auth database.\n"
        f"    DATABASE_URL = {_redacted_url()}\n"
        f"    postgres said: {raw}\n"
        "    Fix the role/password (or the whole URL) in the project's .env file, "
        "then restart the server."
    )


def _redacted_url() -> str:
    try:
        return make_url(DATABASE_URL).render_as_string(hide_password=True)
    except Exception:
        return "<unparseable DATABASE_URL>"


# --------------------------------------------------------------------------- #
# Password helpers
# --------------------------------------------------------------------------- #
def hash_password(plain: str) -> str:
    return _ph.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return _ph.verify(hashed, plain)
    except (VerifyMismatchError, InvalidHashError, TypeError):
        return False


# --------------------------------------------------------------------------- #
# Query helpers used by the routes
# --------------------------------------------------------------------------- #
def normalize_email(email: str) -> str:
    return email.strip().lower()


def get_user_by_email(session: Session, email: str) -> Optional[User]:
    return session.scalar(select(User).where(User.email == normalize_email(email)))


def get_user_by_id(session: Session, user_id: int) -> Optional[User]:
    return session.get(User, user_id)


def create_user(session: Session, *, full_name: str, email: str, password: str) -> User:
    user = User(
        full_name=full_name.strip(),
        email=normalize_email(email),
        password_hash=hash_password(password),
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


# --------------------------------------------------------------------------- #
# Friendships
#
# Every helper here takes the acting user's id and enforces that they are a
# party to the row they're touching, so a route can't be tricked into mutating
# someone else's relationship by passing an arbitrary id.
# --------------------------------------------------------------------------- #
class SocialError(ValueError):
    """A rejected social operation. The message is written for the end user, in
    Azerbaijani, and is safe to put straight into an HTTP response body."""


class FriendshipError(SocialError):
    """Rejected friendship operation; the message is safe to show the user."""


class GroupError(SocialError):
    """Rejected group operation; the message is safe to show the user."""


def friendship_between(session: Session, a_id: int, b_id: int) -> Optional[Friendship]:
    """The row linking these two users, in whichever direction it was created."""
    return session.scalar(
        select(Friendship).where(
            or_(
                (Friendship.requester_id == a_id) & (Friendship.addressee_id == b_id),
                (Friendship.requester_id == b_id) & (Friendship.addressee_id == a_id),
            )
        )
    )


def are_friends(session: Session, a_id: int, b_id: int) -> bool:
    """True only for an accepted relationship - a pending invite is not access."""
    link = friendship_between(session, a_id, b_id)
    return link is not None and link.status == "accepted"


def send_friend_request(session: Session, requester_id: int, addressee_id: int) -> Friendship:
    if requester_id == addressee_id:
        raise FriendshipError("Özünüzə dostluq sorğusu göndərə bilməzsiniz.")
    if session.get(User, addressee_id) is None:
        raise FriendshipError("İstifadəçi tapılmadı.")

    existing = friendship_between(session, requester_id, addressee_id)
    if existing is not None:
        if existing.status == "accepted":
            raise FriendshipError("Siz artıq dostsunuz.")
        if existing.requester_id == requester_id:
            raise FriendshipError("Sorğu artıq göndərilib.")
        # They invited us first and we have just invited them back - that is a yes.
        existing.status = "accepted"
        existing.responded_at = datetime.now(timezone.utc)
        session.commit()
        session.refresh(existing)
        return existing

    link = Friendship(
        requester_id=requester_id, addressee_id=addressee_id, status="pending"
    )
    session.add(link)
    session.commit()
    session.refresh(link)
    return link


def respond_to_request(
    session: Session, user_id: int, request_id: int, accept: bool
) -> Optional[Friendship]:
    """Accept or decline a pending invite addressed to ``user_id``.

    Returns the accepted row, or None when declined (the row is deleted so the
    pair can try again).
    """
    link = session.get(Friendship, request_id)
    # Only the addressee may answer, and only while it is still pending.
    if link is None or link.addressee_id != user_id or link.status != "pending":
        raise FriendshipError("Sorğu tapılmadı.")

    if not accept:
        session.delete(link)
        session.commit()
        return None

    link.status = "accepted"
    link.responded_at = datetime.now(timezone.utc)
    session.commit()
    session.refresh(link)
    return link


def cancel_request(session: Session, user_id: int, request_id: int) -> None:
    """Withdraw an invite that ``user_id`` sent and that is still pending."""
    link = session.get(Friendship, request_id)
    if link is None or link.requester_id != user_id or link.status != "pending":
        raise FriendshipError("Sorğu tapılmadı.")
    session.delete(link)
    session.commit()


def remove_friend(session: Session, user_id: int, other_id: int) -> None:
    link = friendship_between(session, user_id, other_id)
    if link is None or link.status != "accepted":
        raise FriendshipError("Bu istifadəçi dostlarınız arasında deyil.")
    session.delete(link)
    session.commit()


def list_friends(session: Session, user_id: int) -> list[User]:
    """Accepted friends, alphabetically. Reads both directions of the pair."""
    rows = session.scalars(
        select(Friendship).where(
            Friendship.status == "accepted",
            or_(
                Friendship.requester_id == user_id,
                Friendship.addressee_id == user_id,
            ),
        )
    ).all()
    friend_ids = [
        r.addressee_id if r.requester_id == user_id else r.requester_id for r in rows
    ]
    if not friend_ids:
        return []
    return list(
        session.scalars(
            select(User).where(User.id.in_(friend_ids)).order_by(User.full_name)
        ).all()
    )


def list_incoming_requests(session: Session, user_id: int) -> list[tuple[Friendship, User]]:
    rows = session.execute(
        select(Friendship, User)
        .join(User, User.id == Friendship.requester_id)
        .where(Friendship.addressee_id == user_id, Friendship.status == "pending")
        .order_by(Friendship.created_at.desc())
    ).all()
    return [(link, user) for link, user in rows]


def list_outgoing_requests(session: Session, user_id: int) -> list[tuple[Friendship, User]]:
    rows = session.execute(
        select(Friendship, User)
        .join(User, User.id == Friendship.addressee_id)
        .where(Friendship.requester_id == user_id, Friendship.status == "pending")
        .order_by(Friendship.created_at.desc())
    ).all()
    return [(link, user) for link, user in rows]


def _escape_like(term: str) -> str:
    r"""Escape LIKE wildcards so they are matched literally.

    Without this, searching for "%" builds the pattern "%%%", which matches
    every row — turning the search box into a user directory dump. Backslash
    first, or it would double-escape the escapes we add after it.
    """
    return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def search_users(session: Session, user_id: int, query: str, limit: int = 12) -> list[User]:
    """Find people to befriend by name or e-mail. Never returns the searcher."""
    q = (query or "").strip().lower()
    if len(q) < 2:
        return []
    like = f"%{_escape_like(q)}%"
    return list(
        session.scalars(
            select(User)
            .where(
                User.id != user_id,
                or_(
                    func.lower(User.full_name).like(like, escape="\\"),
                    User.email.like(like, escape="\\"),
                ),
            )
            .order_by(User.full_name)
            .limit(limit)
        ).all()
    )


def relationship_state(session: Session, user_id: int, other_id: int) -> str:
    """none | friends | request_sent | request_received - for search result rows."""
    link = friendship_between(session, user_id, other_id)
    if link is None:
        return "none"
    if link.status == "accepted":
        return "friends"
    return "request_sent" if link.requester_id == user_id else "request_received"


# --------------------------------------------------------------------------- #
# Direct messages
# --------------------------------------------------------------------------- #
MAX_MESSAGE_LENGTH = 2000


def save_message(
    session: Session,
    sender_id: int,
    recipient_id: int,
    body: str,
    *,
    kind: str = "text",
    attachment_id: Optional[int] = None,
) -> Message:
    """Persist a message. Callers must check friendship first.

    An image message may have an empty body - the picture is the content, the
    body only a caption - so the emptiness check applies to text only.
    """
    if kind not in ("text", "image"):
        raise FriendshipError("Naməlum mesaj növü.")
    body = (body or "").strip()
    if kind == "text" and not body:
        raise FriendshipError("Mesaj boş ola bilməz.")
    if len(body) > MAX_MESSAGE_LENGTH:
        raise FriendshipError(f"Mesaj çox uzundur (maks. {MAX_MESSAGE_LENGTH} simvol).")
    if kind == "image" and attachment_id is None:
        raise FriendshipError("Şəkil tapılmadı.")

    msg = Message(
        sender_id=sender_id,
        recipient_id=recipient_id,
        body=body,
        kind=kind,
        attachment_id=attachment_id,
    )
    session.add(msg)
    session.commit()
    session.refresh(msg)
    return msg


def get_conversation(
    session: Session, user_id: int, other_id: int, limit: int = 100
) -> list[Message]:
    """The newest ``limit`` messages between two users, returned oldest-first."""
    rows = session.scalars(
        select(Message)
        .where(
            or_(
                (Message.sender_id == user_id) & (Message.recipient_id == other_id),
                (Message.sender_id == other_id) & (Message.recipient_id == user_id),
            )
        )
        .order_by(Message.created_at.desc(), Message.id.desc())
        .limit(limit)
    ).all()
    return list(reversed(rows))


def mark_conversation_read(session: Session, user_id: int, other_id: int) -> int:
    """Mark everything ``other_id`` sent ``user_id`` as read. Returns the count."""
    result = session.execute(
        update(Message)
        .where(
            Message.recipient_id == user_id,
            Message.sender_id == other_id,
            Message.read_at.is_(None),
        )
        .values(read_at=datetime.now(timezone.utc))
    )
    session.commit()
    return int(result.rowcount or 0)


def unread_counts(session: Session, user_id: int) -> dict[int, int]:
    """Map of sender id -> unread message count, for everyone who wrote to us."""
    rows = session.execute(
        select(Message.sender_id, func.count(Message.id))
        .where(Message.recipient_id == user_id, Message.read_at.is_(None))
        .group_by(Message.sender_id)
    ).all()
    return {int(sender_id): int(count) for sender_id, count in rows}


# --------------------------------------------------------------------------- #
# Profiles: bio, city, and the interests that drive group recommendation
# --------------------------------------------------------------------------- #
def list_interests(session: Session) -> list[Interest]:
    return list(session.scalars(select(Interest).order_by(Interest.sort_order)).all())


def interests_grouped(session: Session) -> list[dict]:
    """The catalogue as the picker wants it: categories, each with its chips."""
    rows = list_interests(session)
    out = []
    for key, label in INTEREST_CATEGORIES:
        items = [r.public_dict() for r in rows if r.category == key]
        if items:
            out.append({"key": key, "label": label, "interests": items})
    return out


def get_user_interests(session: Session, user_id: int) -> list[Interest]:
    return list(
        session.scalars(
            select(Interest)
            .join(UserInterest, UserInterest.interest_id == Interest.id)
            .where(UserInterest.user_id == user_id)
            .order_by(Interest.sort_order)
        ).all()
    )


def _resolve_slugs(session: Session, slugs, limit: int) -> list[Interest]:
    """Catalogue rows for these slugs. Unknown slugs are ignored, not an error.

    An unknown slug means the catalogue moved on, not that the client is
    misbehaving, so dropping it silently is kinder than rejecting the whole
    save and losing the picks that *are* valid.
    """
    wanted = []
    seen = set()
    for raw in (slugs or []):
        slug = str(raw).strip().lower()
        if slug and slug not in seen:
            seen.add(slug)
            wanted.append(slug)
    if not wanted:
        return []
    if len(wanted) > limit:
        raise SocialError(f"Ən çoxu {limit} maraq seçilə bilər.")
    rows = session.scalars(select(Interest).where(Interest.slug.in_(wanted))).all()
    order = {slug: i for i, slug in enumerate(wanted)}
    return sorted(rows, key=lambda r: order.get(r.slug, 999))


def set_user_interests(session: Session, user_id: int, slugs) -> list[Interest]:
    chosen = _resolve_slugs(session, slugs, MAX_INTERESTS_PER_USER)
    session.execute(delete(UserInterest).where(UserInterest.user_id == user_id))
    for row in chosen:
        session.add(UserInterest(user_id=user_id, interest_id=row.id))
    session.commit()
    return chosen


def update_profile(
    session: Session,
    user_id: int,
    *,
    full_name: Optional[str] = None,
    bio: Optional[str] = None,
    city: Optional[str] = None,
    interests=None,
) -> User:
    """Patch semantics: only the fields that were passed are touched."""
    user = session.get(User, user_id)
    if user is None:
        raise SocialError("İstifadəçi tapılmadı.")

    if full_name is not None:
        name = full_name.strip()
        if len(name) < 2:
            raise SocialError("Ad və Soyad ən azı 2 simvol olmalıdır.")
        user.full_name = name[:120]
    if bio is not None:
        text = bio.strip()
        if len(text) > MAX_BIO_LENGTH:
            raise SocialError(
                f"Haqqınızda bölməsi çox uzundur (maks. {MAX_BIO_LENGTH} simvol)."
            )
        user.bio = text
    if city is not None:
        user.city = city.strip()[:MAX_CITY_LENGTH]

    session.commit()
    if interests is not None:
        set_user_interests(session, user_id, interests)
    session.refresh(user)
    return user


def profile_dict(session: Session, user: User) -> dict:
    return {
        **user.public_dict(),
        "bio": user.bio or "",
        "city": user.city or "",
        "interests": [i.public_dict() for i in get_user_interests(session, user.id)],
    }


def shared_interest_labels(session: Session, a_id: int, b_id: int) -> list[str]:
    """Labels both people picked - the "why you might get on" line in the UI."""
    mine = {i.slug: i.label for i in get_user_interests(session, a_id)}
    theirs = {i.slug for i in get_user_interests(session, b_id)}
    return [label for slug, label in mine.items() if slug in theirs]


# --------------------------------------------------------------------------- #
# Groups
#
# Same rule as the friendship helpers: every mutating function takes the acting
# user and checks their role itself, so a route cannot be talked into editing a
# group the caller is not an admin of by passing an arbitrary id.
# --------------------------------------------------------------------------- #
def get_group(session: Session, group_id: int) -> Optional[Group]:
    return session.get(Group, group_id)


def group_role(session: Session, group_id: int, user_id: int) -> Optional[str]:
    """'admin' | 'member' | None (not a member)."""
    row = session.get(GroupMember, {"group_id": group_id, "user_id": user_id})
    return row.role if row else None


def is_group_member(session: Session, group_id: int, user_id: int) -> bool:
    return group_role(session, group_id, user_id) is not None


def _require_admin(session: Session, group_id: int, user_id: int) -> Group:
    group = get_group(session, group_id)
    if group is None:
        raise GroupError("Qrup tapılmadı.")
    if group_role(session, group_id, user_id) != "admin":
        raise GroupError("Bunun üçün qrupun admini olmalısınız.")
    return group


def group_member_ids(session: Session, group_id: int) -> list[int]:
    return [
        int(uid)
        for uid in session.scalars(
            select(GroupMember.user_id).where(GroupMember.group_id == group_id)
        ).all()
    ]


def group_members(session: Session, group_id: int) -> list[tuple[GroupMember, User]]:
    """Members with their user row, admins first then by join order."""
    rows = session.execute(
        select(GroupMember, User)
        .join(User, User.id == GroupMember.user_id)
        .where(GroupMember.group_id == group_id)
        .order_by(GroupMember.role, GroupMember.joined_at)
    ).all()
    return [(m, u) for m, u in rows]


def group_member_count(session: Session, group_id: int) -> int:
    return int(
        session.scalar(
            select(func.count(GroupMember.user_id)).where(
                GroupMember.group_id == group_id
            )
        )
        or 0
    )


def group_interests(session: Session, group_id: int) -> list[Interest]:
    return list(
        session.scalars(
            select(Interest)
            .join(GroupInterest, GroupInterest.interest_id == Interest.id)
            .where(GroupInterest.group_id == group_id)
            .order_by(Interest.sort_order)
        ).all()
    )


def create_group(
    session: Session,
    user_id: int,
    *,
    name: str,
    description: str = "",
    interests=None,
    is_open: bool = True,
) -> Group:
    name = (name or "").strip()
    if len(name) < 3:
        raise GroupError("Qrup adı ən azı 3 simvol olmalıdır.")
    if len(name) > MAX_GROUP_NAME:
        raise GroupError(f"Qrup adı çox uzundur (maks. {MAX_GROUP_NAME} simvol).")
    description = (description or "").strip()
    if len(description) > MAX_GROUP_DESCRIPTION:
        raise GroupError(f"Təsvir çox uzundur (maks. {MAX_GROUP_DESCRIPTION} simvol).")

    owned = int(
        session.scalar(
            select(func.count(Group.id)).where(Group.created_by == user_id)
        )
        or 0
    )
    if owned >= MAX_GROUPS_CREATED:
        raise GroupError("Çox sayda qrup yaratmışsınız.")

    chosen = _resolve_slugs(session, interests, MAX_INTERESTS_PER_GROUP)

    group = Group(
        name=name, description=description, created_by=user_id, is_open=bool(is_open)
    )
    session.add(group)
    session.commit()
    session.refresh(group)

    session.add(GroupMember(group_id=group.id, user_id=user_id, role="admin"))
    for row in chosen:
        session.add(GroupInterest(group_id=group.id, interest_id=row.id))
    session.commit()
    return group


def update_group(
    session: Session,
    actor_id: int,
    group_id: int,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    interests=None,
    is_open: Optional[bool] = None,
) -> Group:
    group = _require_admin(session, group_id, actor_id)

    if name is not None:
        new_name = name.strip()
        if len(new_name) < 3:
            raise GroupError("Qrup adı ən azı 3 simvol olmalıdır.")
        group.name = new_name[:MAX_GROUP_NAME]
    if description is not None:
        text = description.strip()
        if len(text) > MAX_GROUP_DESCRIPTION:
            raise GroupError(
                f"Təsvir çox uzundur (maks. {MAX_GROUP_DESCRIPTION} simvol)."
            )
        group.description = text
    if is_open is not None:
        group.is_open = bool(is_open)

    if interests is not None:
        chosen = _resolve_slugs(session, interests, MAX_INTERESTS_PER_GROUP)
        session.execute(delete(GroupInterest).where(GroupInterest.group_id == group_id))
        for row in chosen:
            session.add(GroupInterest(group_id=group_id, interest_id=row.id))

    session.commit()
    session.refresh(group)
    return group


def _attachment_ids_in_group(session: Session, group_id: int) -> list[int]:
    return [
        int(a)
        for a in session.scalars(
            select(GroupMessage.attachment_id).where(
                GroupMessage.group_id == group_id,
                GroupMessage.attachment_id.is_not(None),
            )
        ).all()
    ]


def _reclaim_orphan_attachments(session: Session, attachment_ids) -> int:
    """Delete image bytes that no message points at any more.

    Called after a group and its messages are removed. Without it every photo
    ever sent to a deleted group would sit in the database forever, visible to
    nobody — which on a free Postgres tier is the one kind of leak that actually
    runs out. An id is only reclaimed when *neither* message table references it,
    since the same uploader may have sent the same picture in a direct message
    too.
    """
    removed = 0
    for attachment_id in set(int(a) for a in attachment_ids if a):
        in_dm = session.scalar(
            select(func.count(Message.id)).where(Message.attachment_id == attachment_id)
        )
        in_group = session.scalar(
            select(func.count(GroupMessage.id)).where(
                GroupMessage.attachment_id == attachment_id
            )
        )
        if not in_dm and not in_group:
            session.execute(delete(Attachment).where(Attachment.id == attachment_id))
            removed += 1
    if removed:
        session.commit()
    return removed


def delete_group(session: Session, actor_id: int, group_id: int) -> None:
    group = _require_admin(session, group_id, actor_id)
    doomed_attachments = _attachment_ids_in_group(session, group_id)
    # ON DELETE CASCADE only fires on SQLite with the foreign_keys pragma on, so
    # clear the children explicitly. Cheap, and it makes the behaviour identical
    # on both databases.
    session.execute(delete(GroupMessage).where(GroupMessage.group_id == group_id))
    session.execute(delete(GroupMember).where(GroupMember.group_id == group_id))
    session.execute(delete(GroupInterest).where(GroupInterest.group_id == group_id))
    session.delete(group)
    session.commit()
    _reclaim_orphan_attachments(session, doomed_attachments)


def join_group(session: Session, user_id: int, group_id: int) -> GroupMember:
    group = get_group(session, group_id)
    if group is None:
        raise GroupError("Qrup tapılmadı.")
    existing = session.get(GroupMember, {"group_id": group_id, "user_id": user_id})
    if existing is not None:
        return existing
    if not group.is_open:
        raise GroupError("Bu qrupa yalnız admin əlavə edə bilər.")

    member = GroupMember(group_id=group_id, user_id=user_id, role="member")
    session.add(member)
    session.commit()
    return member


def add_group_member(
    session: Session, actor_id: int, group_id: int, user_id: int
) -> GroupMember:
    _require_admin(session, group_id, actor_id)
    if session.get(User, user_id) is None:
        raise GroupError("İstifadəçi tapılmadı.")
    existing = session.get(GroupMember, {"group_id": group_id, "user_id": user_id})
    if existing is not None:
        raise GroupError("Bu istifadəçi artıq qrupdadır.")
    member = GroupMember(group_id=group_id, user_id=user_id, role="member")
    session.add(member)
    session.commit()
    return member


def _promote_longest_standing(session: Session, group_id: int) -> Optional[int]:
    """Make the earliest-joined remaining member an admin. Returns their id.

    Called when the last admin leaves. Without it the group would be frozen:
    nobody could rename it, add anyone, or remove a troublemaker, and there
    would be no route back to having an admin.
    """
    candidate = session.scalars(
        select(GroupMember)
        .where(GroupMember.group_id == group_id)
        .order_by(GroupMember.joined_at, GroupMember.user_id)
        .limit(1)
    ).first()
    if candidate is None:
        return None
    candidate.role = "admin"
    session.commit()
    return candidate.user_id


def leave_group(session: Session, user_id: int, group_id: int) -> dict:
    """Remove yourself. Returns what else happened as a result.

    ``{"promoted": <user_id|None>, "deleted": bool}`` - leaving can hand the
    group to someone else, or close it entirely if you were the last person in.
    """
    member = session.get(GroupMember, {"group_id": group_id, "user_id": user_id})
    if member is None:
        raise GroupError("Bu qrupun üzvü deyilsiniz.")
    was_admin = member.role == "admin"
    session.delete(member)
    session.commit()

    remaining = group_member_count(session, group_id)
    if remaining == 0:
        group = get_group(session, group_id)
        if group is not None:
            doomed_attachments = _attachment_ids_in_group(session, group_id)
            session.execute(
                delete(GroupMessage).where(GroupMessage.group_id == group_id)
            )
            session.execute(
                delete(GroupInterest).where(GroupInterest.group_id == group_id)
            )
            session.delete(group)
            session.commit()
            _reclaim_orphan_attachments(session, doomed_attachments)
        return {"promoted": None, "deleted": True}

    promoted = None
    if was_admin:
        admins = int(
            session.scalar(
                select(func.count(GroupMember.user_id)).where(
                    GroupMember.group_id == group_id, GroupMember.role == "admin"
                )
            )
            or 0
        )
        if admins == 0:
            promoted = _promote_longest_standing(session, group_id)
    return {"promoted": promoted, "deleted": False}


def remove_group_member(
    session: Session, actor_id: int, group_id: int, user_id: int
) -> None:
    _require_admin(session, group_id, actor_id)
    if user_id == actor_id:
        raise GroupError("Özünüzü çıxarmaq üçün qrupdan ayrılın.")
    member = session.get(GroupMember, {"group_id": group_id, "user_id": user_id})
    if member is None:
        raise GroupError("Bu istifadəçi qrupda deyil.")
    session.delete(member)
    session.commit()


def set_group_role(
    session: Session, actor_id: int, group_id: int, user_id: int, role: str
) -> GroupMember:
    _require_admin(session, group_id, actor_id)
    if role not in ("admin", "member"):
        raise GroupError("Naməlum rol.")
    member = session.get(GroupMember, {"group_id": group_id, "user_id": user_id})
    if member is None:
        raise GroupError("Bu istifadəçi qrupda deyil.")

    if role == "member" and member.role == "admin":
        admins = int(
            session.scalar(
                select(func.count(GroupMember.user_id)).where(
                    GroupMember.group_id == group_id, GroupMember.role == "admin"
                )
            )
            or 0
        )
        if admins <= 1:
            raise GroupError("Qrupda ən azı bir admin qalmalıdır.")

    member.role = role
    session.commit()
    return member


def list_user_groups(session: Session, user_id: int) -> list[Group]:
    return list(
        session.scalars(
            select(Group)
            .join(GroupMember, GroupMember.group_id == Group.id)
            .where(GroupMember.user_id == user_id)
            .order_by(Group.name)
        ).all()
    )


def search_groups(session: Session, query: str, limit: int = 20) -> list[Group]:
    q = (query or "").strip().lower()
    if len(q) < 2:
        return []
    like = f"%{_escape_like(q)}%"
    return list(
        session.scalars(
            select(Group)
            .where(
                or_(
                    func.lower(Group.name).like(like, escape="\\"),
                    func.lower(Group.description).like(like, escape="\\"),
                )
            )
            .order_by(Group.name)
            .limit(limit)
        ).all()
    )


def recommend_groups(
    session: Session, user_id: int, limit: int = 10
) -> list[tuple[Group, int, int]]:
    """Groups worth joining, as ``(group, shared_interest_count, members)``.

    Ranked by how many of the person's interests the group is tagged with, then
    by size. Someone who has picked nothing still gets a list - the busiest
    groups - because an empty discovery page is the one outcome that guarantees
    they never join anything.
    """
    my_interest_ids = [
        int(i)
        for i in session.scalars(
            select(UserInterest.interest_id).where(UserInterest.user_id == user_id)
        ).all()
    ]
    my_group_ids = [
        int(g)
        for g in session.scalars(
            select(GroupMember.group_id).where(GroupMember.user_id == user_id)
        ).all()
    ]

    members = (
        select(func.count(GroupMember.user_id))
        .where(GroupMember.group_id == Group.id)
        .scalar_subquery()
    )
    if my_interest_ids:
        shared = (
            select(func.count(GroupInterest.interest_id))
            .where(
                GroupInterest.group_id == Group.id,
                GroupInterest.interest_id.in_(my_interest_ids),
            )
            .scalar_subquery()
        )
    else:
        shared = literal(0)

    stmt = select(Group, shared.label("shared"), members.label("members"))
    if my_group_ids:
        stmt = stmt.where(Group.id.notin_(my_group_ids))
    stmt = stmt.order_by(
        shared.desc(), members.desc(), Group.created_at.desc(), Group.id.desc()
    ).limit(limit)

    return [(g, int(s or 0), int(m or 0)) for g, s, m in session.execute(stmt).all()]


# --------------------------------------------------------------------------- #
# Group messages
# --------------------------------------------------------------------------- #
def save_group_message(
    session: Session,
    group_id: int,
    sender_id: int,
    body: str = "",
    *,
    kind: str = "text",
    attachment_id: Optional[int] = None,
) -> GroupMessage:
    """Persist a group message. Callers must check membership first."""
    if kind not in ("text", "image", "system"):
        raise GroupError("Naməlum mesaj növü.")
    body = (body or "").strip()
    if kind == "text":
        if not body:
            raise GroupError("Mesaj boş ola bilməz.")
        if len(body) > MAX_MESSAGE_LENGTH:
            raise GroupError(f"Mesaj çox uzundur (maks. {MAX_MESSAGE_LENGTH} simvol).")
    if kind == "image" and attachment_id is None:
        raise GroupError("Şəkil tapılmadı.")

    msg = GroupMessage(
        group_id=group_id,
        sender_id=sender_id,
        body=body,
        kind=kind,
        attachment_id=attachment_id,
    )
    session.add(msg)
    session.commit()
    session.refresh(msg)
    return msg


def get_group_messages(
    session: Session, group_id: int, limit: int = 120
) -> list[tuple[GroupMessage, str]]:
    """Newest ``limit`` messages, oldest-first, each with its sender's name."""
    rows = session.execute(
        select(GroupMessage, User.full_name)
        .join(User, User.id == GroupMessage.sender_id)
        .where(GroupMessage.group_id == group_id)
        .order_by(GroupMessage.created_at.desc(), GroupMessage.id.desc())
        .limit(limit)
    ).all()
    return [(m, name) for m, name in reversed(rows)]


def mark_group_read(session: Session, group_id: int, user_id: int) -> None:
    """Move this member's watermark to the newest message in the group."""
    member = session.get(GroupMember, {"group_id": group_id, "user_id": user_id})
    if member is None:
        return
    newest = session.scalar(
        select(func.max(GroupMessage.id)).where(GroupMessage.group_id == group_id)
    )
    member.last_read_message_id = int(newest or 0)
    session.commit()


def group_unread_counts(session: Session, user_id: int) -> dict[int, int]:
    """group id -> unread count, from the member's read watermark.

    Own messages and the join/leave notices are excluded: a badge should mean
    "somebody said something to you", not "something happened".
    """
    rows = session.execute(
        select(GroupMessage.group_id, func.count(GroupMessage.id))
        .join(GroupMember, GroupMember.group_id == GroupMessage.group_id)
        .where(
            GroupMember.user_id == user_id,
            GroupMessage.sender_id != user_id,
            GroupMessage.kind != "system",
            GroupMessage.id > GroupMember.last_read_message_id,
        )
        .group_by(GroupMessage.group_id)
    ).all()
    return {int(gid): int(count) for gid, count in rows}


# --------------------------------------------------------------------------- #
# Image attachments
# --------------------------------------------------------------------------- #
def save_attachment(
    session: Session, owner_id: int, data: bytes, *, width: int = 0, height: int = 0
) -> Attachment:
    if not data:
        raise SocialError("Şəkil boşdur.")
    if len(data) > MAX_ATTACHMENT_BYTES:
        mb = MAX_ATTACHMENT_BYTES // (1024 * 1024)
        raise SocialError(f"Şəkil çox böyükdür (maks. {mb} MB).")
    mime = sniff_image_mime(data)
    if mime is None:
        raise SocialError("Yalnız JPEG, PNG, GIF və WEBP şəkillər qəbul edilir.")

    row = Attachment(
        owner_id=owner_id,
        mime=mime,
        byte_size=len(data),
        width=max(0, int(width or 0)),
        height=max(0, int(height or 0)),
        data=data,
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def get_attachment(session: Session, attachment_id: int) -> Optional[Attachment]:
    return session.get(Attachment, attachment_id)


def attachment_visible_to(session: Session, attachment_id: int, user_id: int) -> bool:
    """Can ``user_id`` see this image?

    Yes if they uploaded it, or if it is attached to a direct message they are a
    party to, or to a message in a group they belong to. Anything else is a no -
    an attachment id is a small integer, so without this check the whole store
    would be enumerable by anyone with an account.
    """
    row = session.get(Attachment, attachment_id)
    if row is None:
        return False
    if row.owner_id == user_id:
        return True

    in_dm = session.scalar(
        select(func.count(Message.id)).where(
            Message.attachment_id == attachment_id,
            or_(Message.sender_id == user_id, Message.recipient_id == user_id),
        )
    )
    if in_dm:
        return True

    in_group = session.scalar(
        select(func.count(GroupMessage.id))
        .join(GroupMember, GroupMember.group_id == GroupMessage.group_id)
        .where(
            GroupMessage.attachment_id == attachment_id,
            GroupMember.user_id == user_id,
        )
    )
    return bool(in_group)


def attachment_owner(session: Session, attachment_id: int) -> Optional[int]:
    """Who uploaded this attachment, without loading the image bytes.

    ``session.get(Attachment, id)`` would pull the whole picture into memory
    just to read one integer, and this is called on every image message sent.
    """
    return session.scalar(
        select(Attachment.owner_id).where(Attachment.id == attachment_id)
    )
