# Deploying to Vercel

**What runs on Vercel:** the landing / register / login / workspace pages and the
PostgreSQL-backed auth API (`/api/register`, `/api/login`, `/api/logout`,
`/api/me`).

**What does NOT run on Vercel:** the sign-language recognition (`/ws`). It needs a
persistent WebSocket process holding MediaPipe + PyTorch in memory — Vercel's
serverless runtime can't do that, and torch + mediapipe + opencv (~770 MB) blow
past the 250 MB function limit. The workspace page loads and shows a
"recognition offline" notice; auth works fully. To make recognition work too,
run `src/web_demo/backend.py` on a container host (Fly.io / Render / Railway) and
set `RECOGNITION_WS_URL` (below).

---

## 1. A cloud PostgreSQL database

`localhost` Postgres is unreachable from Vercel. Use a managed one:

- **Neon** (https://neon.tech) or **Supabase** — free tier is fine.
- Create a database (any name). Copy its connection string and convert it to the
  psycopg form:

  ```
  postgresql+psycopg://USER:PASSWORD@HOST/DBNAME?sslmode=require
  ```

  (Neon gives `postgresql://…` — just insert `+psycopg` after `postgresql`.)

The app creates the `users` **table** automatically on first request; it does
**not** create the database itself.

## 2. Import the repo into Vercel

Vercel → **Add New… → Project** → import
`github.com/murad-2007ML/Holberton_SignLanguage_Project`.
Framework preset: **Other**. Leave build/output settings empty — `vercel.json`
handles routing (everything → `api/index.py`).

## 3. Environment variables (Project → Settings → Environment Variables)

| Name | Value | Notes |
| --- | --- | --- |
| `DATABASE_URL` | `postgresql+psycopg://USER:PASS@HOST/DB?sslmode=require` | from step 1 |
| `SESSION_SECRET` | 64 hex chars | `python -c "import secrets; print(secrets.token_hex(32))"` |
| `SESSION_COOKIE_SECURE` | `1` | Vercel is HTTPS |
| `RECOGNITION_WS_URL` | `wss://your-backend.fly.dev` | **optional** — only if you host the recognition backend elsewhere |

## 4. Deploy

Push to `main` (or click **Deploy**). Then check:

- `https://<project>.vercel.app/` → landing page
- `/register` → create an account → redirected to `/app`
- `/app` → loads, header shows your name, "recognition offline" notice unless
  `RECOGNITION_WS_URL` is set
- `/login`, logout button, `/api/me` all work

---

## Files that make this work

| File | Purpose |
| --- | --- |
| `vercel.json` | routes every path to the one Python function |
| `api/index.py` | the serverless entrypoint — auth + pages, **no ML imports** |
| `api/requirements.txt` | slim deps (fastapi, sqlalchemy, psycopg, argon2, itsdangerous, dotenv, email-validator) |
| `.vercelignore` | hides `node_modules/`, `outputs/`, the ML source dirs, model binaries, and the full root `requirements.txt` so the function bundle stays small |
| `.python-version` | pins Python 3.12 |
| `src/web_demo/webapp.py` | the shared ML-free web layer (`backend.py` uses it too) |

## Recognition backend (optional, separate host)

`src/web_demo/backend.py` is the full app. On Fly.io / Render / Railway:

```
pip install -r requirements.txt            # full set incl. torch/mediapipe/opencv
python -m uvicorn src.web_demo.backend:app --host 0.0.0.0 --port $PORT
```

Set the same `DATABASE_URL` / `SESSION_SECRET` there, then put that host's URL in
Vercel's `RECOGNITION_WS_URL` as `wss://<host>` (no path). Both origins share the
session cookie only if they're the same site; otherwise the recognition socket
still connects but is treated as unauthenticated — for a single-origin setup,
host the whole `backend.py` on the container platform and skip Vercel.
