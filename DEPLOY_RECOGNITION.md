# Running the recognition backend (your machine, Fly.io, or Render)

> Run `py scripts/preflight_deploy.py` first — it checks the build context, dependencies, routes and environment for this target before you spend a build on it.


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
| **Render free** | ✅ | **No** | nominally 0.1, bursts higher | **Start here — verified working.** Measured ~21 fps round-trip end-to-end (empty frames; expect less with a hand actually being tracked, and less again with several viewers). 512 MB fits 3–4 viewers. Sleeps after 15 min (~50 s cold start). |
| **Google Cloud Run** | ✅ within free tier | **Yes** (~$50 hold) | 1 vCPU | Better performance, but the card hold is a blocker for many. [DEPLOY_CLOUDRUN.md](DEPLOY_CLOUDRUN.md) |
| **Fly.io** | ❌ ~$4–7/mo | Yes | 1 shared vCPU | Most reliable; `suspend` resumes in seconds. |
| **Render Starter** | ❌ ~$7/mo | Yes | 0.5 CPU | Removes the fps limit without changing anything else. |
| **HF Spaces** | ❌ | Yes | — | Docker Spaces now require PRO. |

The frontend self-paces — one frame in flight at a time — so a slow backend
degrades the frame rate instead of building an ever-growing queue. That means a
fractional-CPU instance stays usable; a word trial just takes a little longer
than the nominal 1.7 s.

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
| Health check path | `/health` (no DB access) |

Environment variables: `DATABASE_URL` (same Neon string), `SESSION_SECRET`
(identical to Vercel), `SESSION_COOKIE_SECURE=1`.

There's also a `render.yaml` blueprint if you prefer **New → Blueprint**.

## Stopping it from sleeping

A free service spins down after **15 minutes with no requests**. It isn't gone —
the next request wakes it in **~50 s**, and it stays up while people are using it
(an open WebSocket counts as traffic). So only the first visitor after a quiet
spell ever waits.

Two ways to avoid even that:

**a) Warm it up manually.** Open the URL yourself a minute before the demo. Free,
nothing to set up, and enough for a scheduled meeting.

**b) Keep it awake with an uptime pinger.** Free, no card:
[cron-job.org](https://cron-job.org) or [UptimeRobot](https://uptimerobot.com) →
add a monitor for

```
https://<your-service>.onrender.com/health
```

every **10 minutes**.

> **Ping `/health`, not `/` or `/login`.** `/health` deliberately touches no
> database. The page routes open a DB session, which would wake your Neon
> instance every 10 minutes and burn its free compute-hour allowance for nothing.

Render's free plan includes 750 instance-hours/month and a month is ~730 hours,
so keeping one service awake continuously does fit — but only one.

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
| `libEGL.so.1` / `libGL.so.1: cannot open shared object file` | MediaPipe needs a graphics stack even in CPU mode. The repo `Dockerfile` installs libgl1, libegl1, libgles2, libglib2.0-0, libsm6, libxext6, libxrender1 — make sure you are building from it. |

## Note: local development never uses any of this

One process, same origin, cookie auth — no token involved:

```bash
pip install -r requirements.txt -r requirements-recognition.txt
py -m uvicorn src.web_demo.backend:app --host 0.0.0.0 --port 8000
```

---

# Friend calls: do you need a TURN server?

Audio and video calls between friends are **WebRTC**. The two browsers send
media directly to each other; this backend only relays the handshake (SDP and
ICE), which is a few KB per call. Nothing about calling scales with your
instance size — a free Render box handles it as well as a paid one.

There are two ways a browser finds a path to its peer:

| | What it does | Cost | Credentials |
|---|---|---|---|
| **STUN** | Tells each browser its own public address so they can connect directly | Free (Google's public servers, built in) | None |
| **TURN** | Relays the media when a direct path is impossible | Paid — it carries every byte of the call | **Yes** |

**STUN alone is enough for most users.** TURN is only needed for *symmetric
NAT*, which in practice means some mobile-carrier networks and corporate
firewalls. The usual rule of thumb is that it affects roughly 10–20% of calls.

## Without TURN

Nothing to do — this is the default. Calls work for most pairs of users. The
ones that can't find a path fail with
*"Zəng qoşula bilmədi — şəbəkə məhdudiyyəti"* rather than hanging on a spinner,
and `/api/rtc-config` reports `"turn": false`.

## With TURN

Set three environment variables on the host that **serves the page**:

```bash
TURN_URL=turn:HOST:80,turn:HOST:443,turns:HOST:443?transport=tcp
TURN_USERNAME=<username>
TURN_CREDENTIAL=<password>
```

`TURN_URL` takes a **comma-separated list** sharing one credential. Providers
issue several on purpose and you want them all:

| URL | When it is the one that works |
| --- | --- |
| `turn:HOST:80` (UDP) | Normal networks. Lowest latency, so it is tried first. |
| `turn:HOST:443` (UDP) | UDP allowed but odd ports blocked. |
| `turns:HOST:443?transport=tcp` | UDP blocked entirely. Looks like ordinary HTTPS, so it survives strict corporate firewalls — the exact case that needed a relay. |

Listing only the UDP entry leaves the hardest networks broken, which defeats
the point of paying for a relay.

All three variables must be set; a URL without credentials is ignored, because
a half-configured relay fails every call at ICE time — worse than no relay.

### Getting credentials from Metered (free tier, fastest)

1. Sign up at https://dashboard.metered.ca. Check the current free
   allowance before relying on it — it is measured in hundreds of MB, not
   GB, and relayed video is expensive. See "Budgeting relay bandwidth"
   below.
2. Create an app; open its **TURN credentials** page. You get one username,
   one password, and a list of URLs.
3. Join the URLs with commas into `TURN_URL`, and set the other two.
4. Restart the host and verify:

```bash
py scripts/verify_turn.py --url https://<your-host> --email you@example.com --password ...
```

That signs in, reads the ICE config the server is actually serving browsers,
and sends a real Allocate request to each relay. It reports one of three
things: the relay allocated a port (works), the server rejected the
credentials (fix the password), or inconclusive — UDP is often blocked on the
machine you run it from, and some providers refuse raw allocations by policy,
neither of which means the relay is broken. The definitive test is still a
real call between two networks.

Where to get them:

- **Twilio Network Traversal Service** — pay per GB, no server to run. The
  usual choice if you want this working today.
- **Metered / Open Relay** — has a small free tier, good for testing.
- **Cloudflare Calls** — TURN included, generous free allowance.
- **Self-hosted `coturn`** — free software, but you supply the VM and the
  bandwidth, and it needs its own public IP and open UDP ports.

Verify it took effect:

```bash
curl -s https://<your-host>/api/rtc-config   # signed in
# {"iceServers":[{"urls":[...]},{"urls":"turn:...","username":"..."}],"turn":true}
```

## Budgeting relay bandwidth

Only **relayed** calls cost anything. ICE tries a direct path first, and a
direct call never touches the TURN server — so on most networks the quota is
untouched. You pay only for the calls that could not connect any other way,
which are also the ones that would otherwise have failed entirely.

When a call *is* relayed, every byte crosses the relay and is billed. Rough
figures per hour, per call:

| Mode | Approx. relay usage |
| --- | --- |
| Audio only | ~30–60 MB/hour |
| Video, capped (what this app sends on a relayed call) | ~200–250 MB/hour |
| Video, uncapped 720p | ~700 MB – 1.1 GB/hour |

Free allowances are small — typically hundreds of MB, not GB — so uncapped
video would exhaust one in well under an hour.

The app therefore adapts automatically: on connecting it inspects the selected
ICE candidate pair, and **only if the path is a relay** does it cap outgoing
video to 500 kbps at half resolution. Direct calls keep full quality, because
they are free. A relayed call shows `· ötürücü` next to the timer so the softer
picture is explained rather than mysterious.

To stretch a small allowance further while testing:

* **use audio calls** — roughly five times cheaper, and enough to prove the
  relay works at all
* remember only one side needs a bad network to force the relay; testing two
  devices on the same Wi-Fi goes direct and costs nothing
* watch the usage counter in your provider's dashboard after the first call to
  calibrate against these estimates

`RELAY_MAX_BITRATE` in `src/web_demo/frontend/friends.html` is the knob if you
want to trade quality for minutes differently.

## HTTPS is not optional for calls

Browsers only expose the camera and microphone on a **secure origin**. That
means `localhost` or HTTPS. A LAN IP over plain HTTP (`http://192.168.1.20:8000`)
will load the page and the chat fine, but the call buttons will be disabled
because `navigator.mediaDevices` does not exist there. If you are testing from
a phone against your laptop, use the Cloudflare tunnel from Option C2 — it
gives you an HTTPS URL.

## One instance only

Presence, live chat delivery and call signalling are held in memory in a single
process. Two users connected to *different* instances would not see each other
online and could not call. Keep the recognition/social host at one instance
(Fly: `min_machines_running = 0` with a single machine is fine; Render free is
a single instance by definition) until `Hub.send_to_user` in
`src/web_demo/social.py` is backed by Redis pub/sub.
