# Deploying to Vercel

> Run `py scripts/preflight_deploy.py` first — it checks the build context, dependencies, routes and environment for this target before you spend a build on it.


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

The app creates the `users`, `friendships` and `messages` **tables**
automatically on first request; it does **not** create the database itself.
Adding friends and chat to an existing deployment therefore needs no migration
step — the new tables appear on the next cold start.

## 2. Import the repo into Vercel

Vercel → **Add New… → Project** → import
`github.com/Faig-TYBV/azsl-sign-demo`.
Framework preset: **Other**. **Root Directory must be `./`** (if Vercel
auto-detects FastAPI it will suggest `api` — clear it, or `src/` never gets
deployed). Leave build/output settings empty — `vercel.json` handles routing
(everything → `api/index.py`).

## 3. Environment variables (Project → Settings → Environment Variables)

| Name | Value | Notes |
| --- | --- | --- |
| `DATABASE_URL` | `postgresql+psycopg://USER:PASS@HOST/DB?sslmode=require` | from step 1 |
| `SESSION_SECRET` | 64 hex chars | `python -c "import secrets; print(secrets.token_hex(32))"` |
| `SESSION_COOKIE_SECURE` | `1` | Vercel is HTTPS |
| `RECOGNITION_WS_URL` | `wss://your-backend.fly.dev` | **optional** — the container host running `backend.py`. Enables camera recognition **and** live chat + friend calls (that host serves `/ws/social` too). Without it, friends and messaging still work over REST; only live delivery and calling are hidden. |

## 4. Deploy

Push to `main` (or click **Deploy**). Then check:

- `https://<project>.vercel.app/` → landing page
- `/register` → create an account → redirected to `/app`
- `/app` → loads, header shows your name, "recognition offline" notice unless
  `RECOGNITION_WS_URL` is set
- `/friends` → search for a user, send an invite, accept it from a second
  account, exchange messages. Without `RECOGNITION_WS_URL` the page shows a
  "canlı rejim əlçatan deyil" banner and polls every 5 s instead of streaming;
  with it set, presence dots, typing indicators and the call buttons all work.
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

That one variable also turns on the live social features: the same backend
serves `/ws/social`, and the page authenticates to it with the short-lived
signed token from `/api/ws-token` (cookies can't cross origins). Both hosts
must share the **same `SESSION_SECRET`** — that is what verifies the token —
and the **same `DATABASE_URL`**, since accounts, friendships and messages all
live in one database.

Friend calls are peer-to-peer, so the media never touches either host.

If your users are on mobile data or corporate networks, also set `TURN_URL`,
`TURN_USERNAME` and `TURN_CREDENTIAL`. **In a split deploy these go on Vercel**,
because the browser reads them from `/api/rtc-config` on the origin serving the
page. Setting them on both hosts is harmless and saves you remembering which is
which. See the TURN section of [DEPLOY_RECOGNITION.md](DEPLOY_RECOGNITION.md).

**See [DEPLOY_RECOGNITION.md](DEPLOY_RECOGNITION.md)** for the full walkthrough (Fly.io or Render).

Short version: the repo `Dockerfile` builds the recognition backend; give it the
same `DATABASE_URL` and the **identical** `SESSION_SECRET`, then set
`RECOGNITION_WS_URL=wss://<that-host>` on Vercel and redeploy.

The session cookie cannot cross origins, so the page fetches a short-lived
signed token from `GET /api/ws-token` and passes it on the socket URL; the
recognition backend verifies it with the shared `SESSION_SECRET`. Same-origin
deployments keep using the cookie and never touch that path.
