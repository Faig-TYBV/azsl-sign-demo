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
| `requirements.txt` | **slim, web-only** — this is what Vercel installs (fastapi, sqlalchemy, psycopg, argon2, itsdangerous, dotenv, email-validator) |
| `requirements-recognition.txt` | the ML stack (torch, mediapipe, opencv…) — container hosts only, never installed by Vercel |
| `api/requirements.txt` | same slim list, next to the entrypoint (belt & braces) |
| `.vercelignore` | hides `node_modules/`, `outputs/`, the ML source dirs and model binaries. **Note:** a pattern without a slash matches in *every* directory — that's why the root `requirements.txt` is no longer listed here (it was also hiding `api/requirements.txt`, so Vercel installed nothing and the function crashed with `FUNCTION_INVOCATION_FAILED`). |
| `.python-version` | pins Python 3.12 |
| `src/web_demo/webapp.py` | the shared ML-free web layer (`backend.py` uses it too) |

## Recognition backend (separate host)

To get camera recognition working too, run `src/web_demo/backend.py` on a
container host and set `RECOGNITION_WS_URL` here.

**See [DEPLOY_RENDER.md](DEPLOY_RENDER.md)** for the full Vercel + Render walkthrough.

Short version: the repo `Dockerfile` builds the recognition backend; give it the
same `DATABASE_URL` and the **identical** `SESSION_SECRET`, then set
`RECOGNITION_WS_URL=wss://<that-host>` on Vercel and redeploy.

The session cookie cannot cross origins, so the page fetches a short-lived
signed token from `GET /api/ws-token` and passes it on the socket URL; the
recognition backend verifies it with the shared `SESSION_SECRET`. Same-origin
deployments keep using the cookie and never touch that path.
