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
    String,
    DateTime,
    create_engine,
    func,
    select,
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
