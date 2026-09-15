"""
Shared, ML-free request plumbing: session config, the signed WebSocket token,
and the FastAPI dependencies that resolve the current user.

This module exists so ``webapp.py`` (pages + auth) and ``social.py`` (friends,
chat, calls) can both use these without importing each other. Nothing here
imports the ML stack, so it is safe in the Vercel function.
"""

from __future__ import annotations

import os

from fastapi import Depends, HTTPException, Request, status
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from src.web_demo import db as auth_db

_DEV_SECRET = "dev-insecure-secret-set-SESSION_SECRET"


def _secret_key() -> str:
    return os.getenv("SESSION_SECRET", _DEV_SECRET)


class InsecureConfiguration(RuntimeError):
    """Raised when a deployment would run with the development secret."""


def check_session_secret() -> None:
    """Refuse to serve HTTPS traffic with the fallback development secret.

    SESSION_SECRET signs both the session cookie and the ``/ws`` tokens. Left at
    its default, anyone who has read this repository could mint a cookie for any
    account. Locally that is only a convenience; on a real deployment it is a
    full authentication bypass, so we fail at startup instead of serving.

    SESSION_COOKIE_SECURE=1 is the signal that this is a real deployment — it is
    already required for the cookie to work over HTTPS.
    """
    secret = os.getenv("SESSION_SECRET", "").strip()
    is_production = os.getenv("SESSION_COOKIE_SECURE", "0") == "1"

    if secret and secret != _DEV_SECRET:
        return

    message = (
        "SESSION_SECRET is not set (or is still the example value).\n"
        "    It signs the login cookie and the WebSocket tokens, so leaving it\n"
        "    at the default lets anyone forge a session for any account.\n"
        "    Generate one with:\n"
        '        py -c "import secrets; print(secrets.token_hex(32))"\n'
        "    and set it in .env (or in your host's environment settings)."
    )

    if is_production:
        print("\n" + "=" * 70, flush=True)
        print(message, flush=True)
        print("=" * 70 + "\n", flush=True)
        raise InsecureConfiguration(
            "Refusing to start: SESSION_SECRET must be set when "
            "SESSION_COOKIE_SECURE=1 (see above)."
        )

    print(f"WARNING: {message.splitlines()[0]} Fine for local dev; never deploy like this.", flush=True)


# --------------------------------------------------------------------------- #
# Cross-origin WebSocket auth
#
# When the recognition backend lives on another host (Vercel pages + Render
# /ws), the browser will NOT send the azsl_session cookie to that origin — the
# two are different registrable domains, so no shared cookie is possible. The
# page therefore asks its own origin for a short-lived signed token and passes
# it on the socket URL; the recognition backend verifies it with the same
# SESSION_SECRET. Same-origin deployments keep using the cookie and never touch
# this path.
# --------------------------------------------------------------------------- #
WS_TOKEN_SALT = "azsl-ws-token"
WS_TOKEN_MAX_AGE = 120  # seconds — only has to survive page-load -> connect


def _ws_serializer() -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(_secret_key(), salt=WS_TOKEN_SALT)


def issue_ws_token(user_id: int) -> str:
    return _ws_serializer().dumps({"uid": int(user_id)})


def verify_ws_token(token: str):
    """Return the user id encoded in a valid, unexpired token, else None."""
    if not token:
        return None
    try:
        data = _ws_serializer().loads(token, max_age=WS_TOKEN_MAX_AGE)
        return int(data["uid"])
    except (BadSignature, SignatureExpired, KeyError, TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Session cookie config (identical in both entrypoints)
# --------------------------------------------------------------------------- #
def session_middleware_kwargs() -> dict:
    return dict(
        secret_key=_secret_key(),
        session_cookie="azsl_session",
        https_only=os.getenv("SESSION_COOKIE_SECURE", "0") == "1",
        same_site="lax",
    )


# --------------------------------------------------------------------------- #
# Dependencies
# --------------------------------------------------------------------------- #
def get_db():
    session = auth_db.SessionLocal()
    try:
        yield session
    finally:
        session.close()


def current_user(request: Request, session=Depends(get_db)):
    """The logged-in User, or None."""
    uid = request.session.get("uid")
    if uid is None:
        return None
    try:
        return auth_db.get_user_by_id(session, int(uid))
    except (TypeError, ValueError):
        return None


def require_user(user=Depends(current_user)):
    """For JSON APIs: 401 when not signed in."""
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated"
        )
    return user
