# Recognition backend on Render (Vercel + Render split)

Vercel serves the pages and auth; Render runs the sign-language recognition
(`/ws`). They share one Neon database and one `SESSION_SECRET`.

```
browser ──HTTPS──> Vercel   (pages, /api/register, /api/login, /api/ws-token)
        └──WSS───> Render   (/ws  ← MediaPipe + GRU model)
                     │
        both ────────┴────> Neon Postgres
```

## Why a token, not the cookie

The session cookie is set on `*.vercel.app`; the browser will never send it to
`*.onrender.com`. So the page asks its own origin for a **short-lived signed
token** (`GET /api/ws-token`, 120 s) and appends it to the socket URL. Render
verifies it with the same `SESSION_SECRET`. Both hosts must have the **identical**
`SESSION_SECRET` or every connection is rejected.

---

## 1. Deploy on Render

**Dashboard → New → Web Service → connect `Faig-TYBV/azsl-sign-demo`.**

| Setting | Value |
| --- | --- |
| Language / Runtime | **Docker** (it picks up the repo `Dockerfile`) |
| Region | Frankfurt (closest) |
| Instance type | **Standard (2 GB)** — see the memory note below |
| Health check path | `/login` |

### Environment variables

| Key | Value |
| --- | --- |
| `DATABASE_URL` | the **same** Neon string used on Vercel (`postgresql+psycopg://…?sslmode=require`) |
| `SESSION_SECRET` | the **exact same** 64-char value as Vercel |
| `SESSION_COOKIE_SECURE` | `1` |

Deploy. First build takes ~10 min (torch + MediaPipe). Copy the resulting URL,
e.g. `https://azsl-recognition.onrender.com`.

> There's also a `render.yaml` blueprint in the repo if you'd rather use
> **New → Blueprint**.

### ⚠️ Memory and cold starts

torch + MediaPipe + the GRU model need roughly **700 MB–1 GB** resident.

- **Free (512 MB)** — will be OOM-killed. It also spins down after 15 min idle,
  so the first connection waits ~1 min for a cold start.
- **Starter (512 MB)** — same memory ceiling, likely killed.
- **Standard (2 GB)** — the first plan that comfortably fits.

## 2. Point Vercel at it

Vercel → project → **Settings → Environment Variables** → add:

| Key | Value |
| --- | --- |
| `RECOGNITION_WS_URL` | `wss://azsl-recognition.onrender.com` |

No trailing slash, no `/ws` — the frontend appends that. Use `wss://` (or
`https://`, which is rewritten to `wss://`).

Then **Deployments → ⋯ → Redeploy** so the new env var takes effect.

## 3. Verify

Open the Vercel URL and sign in:

- The yellow "Tanınma xidməti oflayndır" banner is **gone**
- **Kameranı Başlat** works, then **SINAĞA BAŞLA** → 3-2-1 countdown → `KADR 1/26 … 26/26` → a prediction
- Browser devtools → Network → WS shows a connection to `…onrender.com/ws?token=…`

### If the socket won't connect

| Symptom | Cause |
| --- | --- |
| WS closes immediately, code **1008** | `SESSION_SECRET` differs between Vercel and Render |
| "Sessiya doğrulanmadı" | `/api/ws-token` returned 401 — you're not logged in on Vercel |
| First connect hangs ~60 s | Render free/starter cold start |
| Render logs show `Killed` | out of memory — move to Standard |
| `libGL.so.1: cannot open shared object file` | not using the repo `Dockerfile` (it installs `libgl1`) |

## Local development is unchanged

One process, same origin, cookie auth — no token needed:

```bash
pip install -r requirements.txt -r requirements-recognition.txt
py -m uvicorn src.web_demo.backend:app --host 0.0.0.0 --port 8000
```
