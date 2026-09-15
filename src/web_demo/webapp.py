"""
ML-free web layer for the AzSL demo: session auth (register / login / logout)
and the static HTML pages.

Shared by two entrypoints:
  * ``backend.py``          — the full container backend (adds MediaPipe + the
                              GRU model + the ``/ws`` recognition socket)
  * ``api/index.py``        — the Vercel serverless entrypoint (this layer only;
                              torch / mediapipe / opencv never get imported, so
                              the function stays well under Vercel's size limit)

Nothing here imports the ML stack (torch / mediapipe / opencv). Project
dependencies are ``src.web_demo.db`` (SQLAlchemy + argon2 + psycopg — all
small wheels) and ``edge_tts`` (a small async client that talks to Microsoft's
free Edge neural-voice service over a websocket — no local model, no API key).
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import edge_tts
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr, field_validator
from starlette.middleware.sessions import SessionMiddleware

from src.web_demo import db as auth_db
from src.web_demo import social
from src.web_demo.deps import (  # re-exported: backend.py imports verify_ws_token from here
    WS_TOKEN_MAX_AGE,
    WS_TOKEN_SALT,
    check_session_secret,
    current_user,
    get_db,
    issue_ws_token,
    require_user,
    session_middleware_kwargs,
    verify_ws_token,
)

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

# Where the friends page should open /ws/social (live chat, presence, calls):
#   ""    -> same origin (this process serves it)
#   <url> -> the recognition host serves it too, so a Vercel page can use it
#   None  -> nowhere to connect; the page falls back to REST and hides calling
_social_ws_url: str | None = None


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


class TTSIn(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Mətn boş ola bilməz.")
        if len(v) > 500:
            raise ValueError("Mətn çox uzundur (maks. 500 simvol).")
        return v


# --------------------------------------------------------------------------- #
# Page rendering
# --------------------------------------------------------------------------- #
def _serve_page(filename: str) -> HTMLResponse:
    page_path = FRONTEND_DIR / filename
    return HTMLResponse(
        content=page_path.read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


def _serve_configured(filename: str) -> HTMLResponse:
    """
    A page with a small runtime-config script injected before </head>.

    ``recognitionWsUrl`` tells the workspace where (or whether) to open the
    recognition socket:
      ""    -> same origin (this process also serves /ws)
      <url> -> a separate backend, e.g. "wss://azsl-api.fly.dev"
      null  -> recognition unavailable (Vercel with no backend); the page
               shows a notice instead of retrying a socket that can't exist

    ``socialWsUrl`` is the same idea for /ws/social, which the friends page
    uses for live chat, presence and calls:
      ""    -> same origin (this process serves it)
      <url> -> a separate host serves it; the page authenticates with a signed
               token because cookies don't cross origins
      null  -> no socket anywhere; the page polls REST and hides calling
    """
    import json

    html = (FRONTEND_DIR / filename).read_text(encoding="utf-8")
    cfg = (
        "<script>window.__AZSL_CONFIG__="
        + json.dumps(
            {
                "recognitionWsUrl": _injected_ws_url,
                "socialWsUrl": _social_ws_url,
            }
        )
        + ";</script>"
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


@api_router.post("/tts")
async def api_tts(payload: TTSIn, user=Depends(require_user)):
    """Azerbaijani text-to-speech for the sentence builder's "Səsləndir" button.

    Uses edge-tts (Microsoft's free Edge neural-voice service — no API key,
    no quota) rather than the client-side Web Speech API, since most browsers
    / OSes don't ship an az-AZ voice at all.
    """
    communicate = edge_tts.Communicate(payload.text, voice="az-AZ-BanuNeural")
    audio_buffer = io.BytesIO()
    try:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_buffer.write(chunk["data"])
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Səsləndirmə xidməti əlçatan deyil.",
        ) from exc

    audio_bytes = audio_buffer.getvalue()
    if not audio_bytes:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Səsləndirmə xidməti əlçatan deyil.",
        )
    return Response(content=audio_bytes, media_type="audio/mpeg")


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
    return _serve_configured("index.html")


@page_router.get("/friends")
async def friends_page(user=Depends(current_user)):
    if user is None:
        return RedirectResponse(url="/login", status_code=302)
    return _serve_configured("friends.html")


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
                       opens the socket on its own origin. /ws/social (chat,
                       presence, calls) is registered here too.
    serves_ws=False -> no local /ws (Vercel); use RECOGNITION_WS_URL if set,
                       otherwise tell the page recognition is unavailable. The
                       friends page still works over REST, minus calling.
    """
    global _injected_ws_url, _social_ws_url
    if RECOGNITION_WS_URL:
        _injected_ws_url = RECOGNITION_WS_URL
    else:
        _injected_ws_url = "" if serves_ws else None

    # The social socket rides along with the recognition backend: any host that
    # can hold /ws open can hold /ws/social open too. So a Vercel page with
    # RECOGNITION_WS_URL set gets full live chat and calling against that host
    # (authenticated by the same signed token as /ws), rather than silently
    # degrading to REST polling.
    if serves_ws:
        _social_ws_url = ""
    elif RECOGNITION_WS_URL:
        _social_ws_url = RECOGNITION_WS_URL
    else:
        _social_ws_url = None

    # Fails fast on a deployment still using the fallback dev secret, which
    # would otherwise let anyone forge a session cookie.
    check_session_secret()

    app.add_middleware(SessionMiddleware, **session_middleware_kwargs())
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
    app.include_router(api_router)
    app.include_router(social.social_router)
    app.include_router(page_router)
    if serves_ws:
        social.register_social_ws(app)
