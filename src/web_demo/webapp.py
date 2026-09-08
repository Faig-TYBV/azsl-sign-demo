"""
ML-free web layer for the AzSL demo: session auth (register / login / logout)
and the static HTML pages.

Shared by two entrypoints:
  * ``backend.py``          — the full container backend (adds MediaPipe + the
                              GRU model + the ``/ws`` recognition socket)
  * ``api/index.py``        — the Vercel serverless entrypoint (this layer only;
                              torch / mediapipe / opencv never get imported, so
                              the function stays well under Vercel's size limit)

Nothing here imports the ML stack. The only project dependency is
``src.web_demo.db`` (SQLAlchemy + argon2 + psycopg — all small wheels).
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from pydantic import BaseModel, EmailStr, field_validator
from starlette.middleware.sessions import SessionMiddleware

from src.web_demo import db as auth_db

FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"

# Optional override: the absolute base URL of a separate recognition backend,
# e.g. "wss://azsl-api.fly.dev". Normally left unset:
#   * the container backend (backend.py) serves /ws itself -> same origin
#   * Vercel with no backend -> the /app page shows a "recognition offline"
#     notice instead of retrying a socket that can't exist there
RECOGNITION_WS_URL = os.getenv("RECOGNITION_WS_URL", "").strip()

# Set by build_web_layer(): the value injected into index.html as
# window.__AZSL_CONFIG__.recognitionWsUrl  ("" | "<url>" | None).
_injected_ws_url: str | None = None


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
    return URLSafeTimedSerializer(
        os.getenv("SESSION_SECRET", "dev-insecure-secret-set-SESSION_SECRET"),
        salt=WS_TOKEN_SALT,
    )


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
        secret_key=os.getenv("SESSION_SECRET", "dev-insecure-secret-set-SESSION_SECRET"),
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


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #
class RegisterIn(BaseModel):
    full_name: str
    email: EmailStr
    password: str
    confirm: str | None = None

    @field_validator("full_name")
    @classmethod
    def _name_len(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 2:
            raise ValueError("Ad və Soyad ən azı 2 simvol olmalıdır.")
        return v

    @field_validator("password")
    @classmethod
    def _pw_len(cls, v: str) -> str:
        if len(v) < 6:
            raise ValueError("Şifrə ən azı 6 simvoldan ibarət olmalıdır.")
        return v

    @field_validator("confirm")
    @classmethod
    def _pw_match(cls, v, info):
        if v is not None and v != info.data.get("password"):
            raise ValueError("Şifrələr uyğun gəlmir.")
        return v


class LoginIn(BaseModel):
    email: EmailStr
    password: str


# --------------------------------------------------------------------------- #
# Page rendering
# --------------------------------------------------------------------------- #
def _serve_page(filename: str) -> HTMLResponse:
    page_path = FRONTEND_DIR / filename
    return HTMLResponse(
        content=page_path.read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


def _serve_workspace() -> HTMLResponse:
    """
    index.html with a small runtime-config script injected before </head> so
    the frontend knows where (or whether) to open the recognition socket:
      ""    -> same origin (this process also serves /ws)
      <url> -> a separate backend, e.g. "wss://azsl-api.fly.dev"
      null  -> recognition unavailable (Vercel with no backend); the page
               shows a notice instead of retrying a socket that can't exist
    """
    import json

    html = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    cfg = (
        "<script>window.__AZSL_CONFIG__="
        f'{{"recognitionWsUrl": {json.dumps(_injected_ws_url)}}};</script>'
    )
    html = html.replace("</head>", cfg + "\n</head>", 1)
    return HTMLResponse(
        content=html,
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


# --------------------------------------------------------------------------- #
# Routers
# --------------------------------------------------------------------------- #
api_router = APIRouter(prefix="/api", tags=["auth"])


@api_router.post("/register")
async def api_register(payload: RegisterIn, request: Request, session=Depends(get_db)):
    if auth_db.get_user_by_email(session, payload.email):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Bu e-poçt artıq qeydiyyatdan keçib.",
        )
    user = auth_db.create_user(
        session,
        full_name=payload.full_name,
        email=payload.email,
        password=payload.password,
    )
    request.session["uid"] = user.id
    return {"user": user.public_dict()}


@api_router.post("/login")
async def api_login(payload: LoginIn, request: Request, session=Depends(get_db)):
    user = auth_db.get_user_by_email(session, payload.email)
    if user is None or not auth_db.verify_password(payload.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="E-poçt və ya şifrə yanlışdır.",
        )
    request.session["uid"] = user.id
    return {"user": user.public_dict()}


@api_router.post("/logout")
async def api_logout(request: Request):
    request.session.clear()
    return {"ok": True}


@api_router.get("/me")
async def api_me(user=Depends(require_user)):
    return {"user": user.public_dict()}


@api_router.get("/ws-token")
async def api_ws_token(user=Depends(require_user)):
    """Short-lived token so a cross-origin recognition backend can authenticate
    this signed-in user (cookies don't cross origins)."""
    return {"token": issue_ws_token(user.id), "expires_in": WS_TOKEN_MAX_AGE}


page_router = APIRouter(tags=["pages"])


@page_router.get("/health")
async def health():
    """
    Liveness only — deliberately touches no database.

    Free hosts sleep after ~15 minutes idle, so this is the endpoint an uptime
    pinger should hit. Pinging a page route instead would open a DB session on
    every ping and keep the (also free, also auto-suspending) Postgres awake,
    burning its compute-hour allowance for nothing.
    """
    return {"ok": True}


@page_router.get("/")
async def root():
    return _serve_page("landing.html")


@page_router.get("/login")
async def login_page(user=Depends(current_user)):
    if user is not None:
        return RedirectResponse(url="/app", status_code=302)
    return _serve_page("register.html")


@page_router.get("/register")
async def register_page(user=Depends(current_user)):
    if user is not None:
        return RedirectResponse(url="/app", status_code=302)
    return _serve_page("register.html")


@page_router.get("/app")
async def app_page(user=Depends(current_user)):
    if user is None:
        return RedirectResponse(url="/login", status_code=302)
    return _serve_workspace()


@page_router.get("/workspace")
async def workspace_alias():
    return RedirectResponse(url="/app")


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #
def build_web_layer(app: FastAPI, *, serves_ws: bool = False) -> None:
    """
    Attach the session middleware, static mount, and auth + page routes.

    serves_ws=True  -> this same process also serves /ws, so the workspace page
                       opens the socket on its own origin.
    serves_ws=False -> no local /ws (Vercel); use RECOGNITION_WS_URL if set,
                       otherwise tell the page recognition is unavailable.
    """
    global _injected_ws_url
    if RECOGNITION_WS_URL:
        _injected_ws_url = RECOGNITION_WS_URL
    else:
        _injected_ws_url = "" if serves_ws else None

    app.add_middleware(SessionMiddleware, **session_middleware_kwargs())
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
    app.include_router(api_router)
    app.include_router(page_router)
