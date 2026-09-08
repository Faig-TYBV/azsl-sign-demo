# Deploy to Google Cloud Run (free tier)

Cloud Run runs the **whole app** — pages, auth *and* the recognition WebSocket —
on one origin. That means no Vercel, no cross-origin token, no tunnel: the
browser just uses the normal session cookie.

**Why it fits:** the free tier gives 2M requests, 360,000 GiB-seconds and 180,000
vCPU-seconds a month, with real vCPUs while serving and scale-to-zero when idle.
This app needs ~300 MB idle + ~50 MB per viewer and 1 vCPU, so a demo sits well
inside the allowance.

**The trade-off:** scaling to zero means the first request after an idle period
pays a **~30–60 s cold start** (the image carries torch + MediaPipe, and the
model load is ~15 s). Open the URL yourself a minute before your meeting to warm
it up.

---

## 1. One-time Google Cloud setup

1. Create a Google Cloud account: https://console.cloud.google.com
   A card is required for identity, but nothing is charged inside the free tier.
2. Create a project (any name), and note its **Project ID**.
3. Install the CLI: https://cloud.google.com/sdk/docs/install
4. Sign in and select the project:

```powershell
gcloud auth login
gcloud config set project <YOUR_PROJECT_ID>
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
```

## 2. Deploy

You already have the Neon `DATABASE_URL`. Generate a session secret if you don't
have one handy:

```powershell
py -c "import secrets; print(secrets.token_hex(32))"
```

Then, from the repo root:

```powershell
.\scripts\deploy-cloudrun.ps1 `
  -DatabaseUrl "postgresql+psycopg://neondb_owner:...@ep-....neon.tech/neondb?sslmode=require" `
  -SessionSecret "<your 64-char hex>"
```

Cloud Build builds the image in the cloud — **Docker is not needed locally**.
First build takes **~10 minutes** (torch + MediaPipe). When it finishes the
script prints your URL:

```
https://azsl-recognition-xxxxxxxx-ew.a.run.app
```

## 3. Use it

Open that URL. Register, log in, **Kameranı Başlat** → **SINAĞA BAŞLA** → sign.
Share the same link with anyone, anywhere — recognition runs on Google's
servers, so your laptop can be off.

## Redeploying

After committing changes:

```powershell
.\scripts\deploy-cloudrun.ps1 -SkipEnv
```

`-SkipEnv` reuses the environment variables already set on the service.

---

## What the script sets, and why

| Flag | Why |
| --- | --- |
| `--timeout 3600` | WebSockets are long-lived requests. The 5-minute default would cut off every session mid-demo; 3600 s is Cloud Run's maximum. |
| `--concurrency 8` | ~50 MB per concurrent viewer on top of ~300 MB idle — keeps a 1 GiB instance safely under its limit. |
| `--min-instances 0` | Scale to zero. This is what keeps it inside the free tier (and what causes the cold start). |
| `--max-instances 3` | Caps runaway scaling so a busy moment can't quietly leave the free tier. |
| `--cpu-boost` | Extra CPU during startup, so the ~15 s model load finishes sooner. |
| `--memory 1Gi` / `--cpu 1` | Measured need is ~300 MB + ~50 MB per viewer; 1 GiB is comfortable. |

The `Dockerfile` is shared with the other hosts. Cloud Run injects `PORT`
(8080) and the container's `CMD` honours it.

`.gcloudignore` keeps `node_modules/`, `data/`, `.env` and the training
artefacts out of the upload — without it, gcloud falls back to `.gitignore` and
would still upload the on-disk `node_modules`.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| Build fails: API not enabled | run the `gcloud services enable ...` line from step 1 |
| Container fails to start, logs mention the DB banner | `DATABASE_URL` is wrong — needs `+psycopg` and `?sslmode=require` |
| WebSocket drops after ~5 minutes | `--timeout` wasn't applied; redeploy with the script |
| First visit takes ~a minute | cold start — expected with `--min-instances 0`. Warm it up before the demo, or set `--min-instances 1` (leaves the free tier: ~$8-15/mo) |
| `Killed` / OOM in logs | lower `--concurrency`, or raise `--memory` to `2Gi` |
| Camera won't start | the page must be HTTPS — `*.run.app` already is |

## Keeping Vercel as the public URL (optional)

Not necessary — the Cloud Run URL serves the full app. But if you want the
prettier Vercel address, set on Vercel:

```
RECOGNITION_WS_URL = wss://azsl-recognition-xxxxxxxx-ew.a.run.app
```

and redeploy. That is the cross-origin path, so both must share a
**byte-identical `SESSION_SECRET`** — see
[DEPLOY_RECOGNITION.md](DEPLOY_RECOGNITION.md).
