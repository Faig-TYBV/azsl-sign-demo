# Running the recognition backend (your machine, Fly.io, or Render)

Vercel serves the pages and auth; something else has to run the sign-language
recognition (`/ws`), because it needs a persistent WebSocket process holding
MediaPipe + PyTorch in memory.

> **Don't want to pay for hosting?** Skip to **[Option C — your own machine](#option-c--your-own-machine-free)**.
> Running everything locally costs nothing and needs no configuration at all.

```
browser ──HTTPS──> Vercel        (pages, /api/register, /api/login, /api/ws-token)
        └──WSS───> Fly / Render  (/ws  ← MediaPipe + GRU model)
                     │
        both ────────┴─────────> Neon Postgres
```

## Why a token, not the cookie

The session cookie is set on `*.vercel.app`; the browser will never send it to
`*.fly.dev` or `*.onrender.com`. So the page asks its own origin for a
**short-lived signed token** (`GET /api/ws-token`, 120 s TTL) and appends it to
the socket URL. The recognition backend verifies it with the same
`SESSION_SECRET`. **Both hosts must have a byte-identical `SESSION_SECRET`** or
every connection is closed with code 1008.

## What this app actually needs

Measured on the real thing, not estimated:

| | RSS |
| --- | --- |
| Idle, model loaded | **298 MB** |
| + 1 concurrent viewer | 361 MB |
| + 2 viewers | 406 MB |
| + 3 viewers | 452 MB |

So roughly **300 MB + ~50 MB per concurrent viewer**. A 512 MB instance holds
3–4 people; 1 GB is comfortable.

**CPU is the real limit, not RAM.** Every frame is a JPEG decode plus a MediaPipe
landmark pass, so a fractional-CPU instance simply runs at fewer frames per
second — it still works, just slower.

## Which host?

A frame costs **12.3 ms** of CPU (JPEG decode + MediaPipe), measured. At 15 fps
that's ~18% of one core.

| | Free? | Card needed? | CPU | Verdict |
| --- | --- | --- | --- | --- |
| **Render free** | ✅ | **No** | 0.1 → **~8 fps** | **Start here.** Degraded but working, and the only option that needs no card. 512 MB fits 3–4 viewers. Sleeps after 15 min (~50 s cold start). |
| **Google Cloud Run** | ✅ within free tier | **Yes** (~$50 hold) | 1 vCPU | Better performance, but the card hold is a blocker for many. [DEPLOY_CLOUDRUN.md](DEPLOY_CLOUDRUN.md) |
| **Fly.io** | ❌ ~$4–7/mo | Yes | 1 shared vCPU | Most reliable; `suspend` resumes in seconds. |
| **Render Starter** | ❌ ~$7/mo | Yes | 0.5 CPU | Removes the fps limit without changing anything else. |
| **HF Spaces** | ❌ | Yes | — | Docker Spaces now require PRO. |

On a 0.1-CPU instance a 26-frame word trial takes ~3.3 s instead of ~1.7 s. The
frontend self-paces (one frame in flight at a time), so a slow backend degrades
the frame rate instead of building an ever-growing queue.

> **Want it fast on a free tier?** The real fix is to run MediaPipe in the
> browser and send landmarks (~500 bytes) instead of JPEG frames, leaving the
> server with just the 203 K-parameter GRU. That drops per-frame server cost
> from 12.3 ms to well under 1 ms and cuts bandwidth ~40×. It's a real
> refactor, not a config change — ask if you want it.

*(Prices and free-tier limits change — check current pricing before committing.)*

---

# Option A — Fly.io (cheapest paid host)

Install the CLI, then from the repo root:

```bash
# 1. install (PowerShell)
iwr https://fly.io/install.ps1 -useb | iex

# 2. sign in
fly auth signup      # or: fly auth login

# 3. create the app from the committed fly.toml (does not deploy yet)
fly launch --no-deploy --copy-config

# 4. secrets — SESSION_SECRET must match Vercel exactly
fly secrets set \
  DATABASE_URL='postgresql+psycopg://neondb_owner:...@ep-....neon.tech/neondb?sslmode=require' \
  SESSION_SECRET='<the same 64-char value as Vercel>' \
  SESSION_COOKIE_SECURE=1

# 5. deploy (first build ~10 min: torch + MediaPipe)
fly deploy
```

Your URL is `https://<app>.fly.dev`. Check it booted:

```bash
fly logs        # expect "Auth database ready" + "WebSocket backend started."
fly status
```

If `azsl-recognition` is taken, `fly launch` will offer another name — update
`app =` in `fly.toml` to match.

# Option B — Render (free tier works; no credit card)

**Dashboard → New → Web Service → connect `Faig-TYBV/azsl-sign-demo`.**

| Setting | Value |
| --- | --- |
| Runtime | **Docker** (picks up the repo `Dockerfile`) |
| Region | Frankfurt |
| Instance type | **Free** works (~8 fps, sleeps after 15 min idle). **Starter** removes the fps limit. |
| Health check path | `/login` |

Environment variables: `DATABASE_URL` (same Neon string), `SESSION_SECRET`
(identical to Vercel), `SESSION_COOKIE_SECURE=1`.

There's also a `render.yaml` blueprint if you prefer **New → Blueprint**.

---

# Option C — your own machine (free)

No hosting bill. Two ways to do it.

## C1. Everything local — simplest, nothing to configure

```bash
py -m uvicorn src.web_demo.backend:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000**. This one process serves the pages, the auth API
*and* `/ws` — same origin, cookie auth, no token, no tunnel. Everything works,
including camera recognition. Ideal for demoing on your own laptop or to anyone
on the same Wi-Fi (`http://<your-LAN-IP>:8000`).

## C2. Public Vercel page + your laptop doing the ML

Use a tunnel so the HTTPS page can reach your machine over `wss://`.
(Browsers block plain `ws://` from an `https://` page, so a tunnel is required —
it also gives you TLS for free.)

**1. Start the backend with Vercel's `SESSION_SECRET`.** Only that has to match —
`/ws` verifies the signed token and never touches the database, so your local
`DATABASE_URL` can stay as-is.

```powershell
$env:SESSION_SECRET="<the same 64-char value as Vercel>"
py -m uvicorn src.web_demo.backend:app --host 0.0.0.0 --port 8000
```

**2. Open a tunnel.** Cloudflare's quick tunnel needs no account:

```powershell
winget install --id Cloudflare.cloudflared
cloudflared tunnel --url http://localhost:8000
```

It prints something like `https://tidy-otter-yak.trycloudflare.com`.

*(ngrok works too and its free tier now includes one **static** domain, which is
worth it if you restart often: `ngrok http 8000 --domain=your-name.ngrok-free.app`.)*

**3. Point the deployed page at it — no redeploy needed.** Visit your Vercel
`/app` once with a `?ws=` parameter:

```
https://azsl-sign-demo.vercel.app/app?ws=wss://tidy-otter-yak.trycloudflare.com
```

The URL is saved in `localStorage`, so every later visit uses it automatically.
To go back to the default (or clear a dead tunnel):

```
https://azsl-sign-demo.vercel.app/app?ws=
```

**Caveats:** the quick-tunnel hostname changes on every restart (just re-visit
with the new `?ws=`), and recognition only works while your machine and the
tunnel are running. Anyone you share the link with is sending their camera
frames to *your* computer.

---

## Point Vercel at a hosted backend (Options A and B)

Vercel → project → **Settings → Environment Variables** → add:

| Key | Value |
| --- | --- |
| `RECOGNITION_WS_URL` | `wss://azsl-recognition.fly.dev` (or `wss://….onrender.com`) |

No trailing slash, no `/ws` — the frontend appends that. Then
**Deployments → ⋯ → Redeploy** so the env var takes effect.

## Verify

Open the Vercel URL and sign in:

- The yellow "Tanınma xidməti oflayndır" banner is **gone**
- **Kameranı Başlat**, then **SINAĞA BAŞLA** → 3-2-1 countdown → `KADR 1/26 … 26/26` → a prediction
- Devtools → Network → WS shows a connection to `…/ws?token=…`

### If the socket won't connect

| Symptom | Cause |
| --- | --- |
| WS closes immediately, code **1008** | `SESSION_SECRET` differs between the two hosts |
| "Sessiya doğrulanmadı" | `/api/ws-token` returned 401 — not logged in on Vercel |
| First connect hangs | cold start (Render ~60 s; Fly ~1–3 s suspended, ~20 s stopped) |
| Logs show `Killed` / OOM | instance too small — needs ~300 MB idle plus ~50 MB per viewer |
| `libGL.so.1: cannot open shared object file` | not using the repo `Dockerfile` (it installs `libgl1`) |

## Note: local development never uses any of this

One process, same origin, cookie auth — no token involved:

```bash
pip install -r requirements.txt -r requirements-recognition.txt
py -m uvicorn src.web_demo.backend:app --host 0.0.0.0 --port 8000
```
