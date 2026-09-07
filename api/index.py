"""
Vercel serverless entrypoint.

Serves the landing / register / login / workspace pages and the PostgreSQL-backed
auth API (/api/register, /api/login, /api/logout, /api/me). It does NOT run the
sign-language recognition: that needs a persistent WebSocket process (MediaPipe +
PyTorch), which Vercel's serverless runtime cannot host. Point the workspace at a
separate always-on backend by setting the RECOGNITION_WS_URL env var
(e.g. "wss://azsl-api.fly.dev"); leave it unset and the workspace shows an
"recognition offline" notice while auth keeps working.

Required env vars on Vercel:
  DATABASE_URL           postgresql+psycopg://USER:PASS@HOST:5432/DBNAME   (e.g. Neon)
  SESSION_SECRET         64 hex chars — `python -c "import secrets;print(secrets.token_hex(32))"`
  SESSION_COOKIE_SECURE  1
Optional:
  RECOGNITION_WS_URL     wss://<your recognition backend>
"""

import os
import sys

# Make `src...` importable when this file runs as the Vercel function.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fastapi import FastAPI  # noqa: E402

from src.web_demo import db as auth_db  # noqa: E402
from src.web_demo.webapp import build_web_layer  # noqa: E402

app = FastAPI(title="AzSL Web (Vercel)")
build_web_layer(app, serves_ws=False)

# Create the users table on the first cold start. The database itself must
# already exist (managed Postgres won't let us CREATE DATABASE); init_db()
# handles "table already exists" fine and we swallow a transient DB error so a
# blip doesn't hard-fail every request in this function instance.
try:
    auth_db.init_db()
except Exception as exc:  # pragma: no cover - startup best-effort
    print(f"[api/index] init_db skipped: {exc}", flush=True)
