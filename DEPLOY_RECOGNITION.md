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

## Which host?

torch + MediaPipe + the GRU model need roughly **700 MB–1 GB resident**, so the
512 MB tiers on either platform get OOM-killed.

| | Fly.io | Render |
| --- | --- | --- |
| 2 GB, always on | ~$11/mo | ~$25/mo (Standard) |
| Scale to zero when idle | ✅ `auto_stop_machines` | ❌ (free tier spins down, but 512 MB won't fit) |
| Cold start | ~1–3 s with `suspend` (RAM snapshot) | ~60 s |
| Config in repo | `fly.toml` | `render.yaml` |

If you do want a host, **Fly is the better fit** — cheaper, and `auto_stop_machines = "suspend"`
snapshots RAM so a resume skips the ~15 s torch/MediaPipe load. For a demo used
occasionally you pay for very little running time.

*(Prices are approximate — check current pricing before committing.)*

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

# Option B — Render

**Dashboard → New → Web Service → connect `Faig-TYBV/azsl-sign-demo`.**

| Setting | Value |
| --- | --- |
| Runtime | **Docker** (picks up the repo `Dockerfile`) |
| Region | Frankfurt |
| Instance type | **Standard (2 GB)** — Free/Starter are 512 MB and will be OOM-killed |
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
| Logs show `Killed` / OOM | instance too small — needs ~1 GB, use 2 GB |
| `libGL.so.1: cannot open shared object file` | not using the repo `Dockerfile` (it installs `libgl1`) |

## Note: local development never uses any of this

One process, same origin, cookie auth — no token involved:

```bash
pip install -r requirements.txt -r requirements-recognition.txt
py -m uvicorn src.web_demo.backend:app --host 0.0.0.0 --port 8000
```
