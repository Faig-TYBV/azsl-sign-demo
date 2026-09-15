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
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    func,
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
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "read_at": self.read_at.isoformat() if self.read_at else None,
        }


# --------------------------------------------------------------------------- #
# Schema / database bootstrap
# --------------------------------------------------------------------------- #
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
    Prepare the auth store: create the database if needed, then create the
    ``users`` table. Raises :class:`DatabaseUnavailable` with an actionable
    message on any connection/permission problem.
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
class FriendshipError(ValueError):
    """Rejected friendship operation; the message is safe to show the user."""


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


def save_message(session: Session, sender_id: int, recipient_id: int, body: str) -> Message:
    """Persist a message. Callers must check friendship first."""
    body = (body or "").strip()
    if not body:
        raise FriendshipError("Mesaj boş ola bilməz.")
    if len(body) > MAX_MESSAGE_LENGTH:
        raise FriendshipError(f"Mesaj çox uzundur (maks. {MAX_MESSAGE_LENGTH} simvol).")

    msg = Message(sender_id=sender_id, recipient_id=recipient_id, body=body)
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
