# Deploy the whole app to Hugging Face Spaces (free)

> **⚠️ Out of date:** Hugging Face now requires a PRO subscription for Docker
> Spaces; only Static Spaces are free. This guide no longer describes a free
> deployment. See [DEPLOY_RECOGNITION.md](DEPLOY_RECOGNITION.md) for the current
> options.

This is the simplest working deployment: **one Space runs everything** — landing
page, registration, login, *and* the recognition WebSocket. Same origin, so the
session cookie just works and none of the cross-origin token machinery is
involved. You do **not** need Vercel for this.

## Why this works when Vercel/Render free tiers don't

| | HF Spaces (free CPU) | Vercel | Render free |
| --- | --- | --- | --- |
| RAM | **16 GB** | n/a (serverless, 250 MB bundle) | 512 MB |
| Persistent process / WebSockets | ✅ | ❌ | ✅ |
| Cost | **$0** | $0 | $0 |
| Sleeps after | 48 h idle | — | 15 min idle |

torch + MediaPipe + the GRU model need ~1 GB resident, which is why the 512 MB
tiers get OOM-killed. HF's free CPU tier has 16 GB.

---

## 1. Database (once)

Spaces have no database, so keep using **Neon** — the same
`DATABASE_URL` you already created for Vercel:

```
postgresql+psycopg://neondb_owner:...@ep-....neon.tech/neondb?sslmode=require
```

## 2. Create the Space

1. Sign in at **https://huggingface.co** (free, no card).
2. **New → Space**.
3. Fill in:
   - **Owner**: your username
   - **Space name**: `azsl-sign-demo`
   - **License**: whatever you like
   - **SDK**: **Docker** → *Blank*
   - **Hardware**: **CPU basic — 2 vCPU, 16 GB (FREE)**
   - **Visibility**: **Public** (a private Space isn't reachable by your audience)
4. **Create Space**.

## 3. Add the secrets

Space → **Settings** → **Variables and secrets** → **New secret** for each:

| Name | Value |
| --- | --- |
| `DATABASE_URL` | your Neon string (as above) |
| `SESSION_SECRET` | any 64 hex chars — `py -c "import secrets; print(secrets.token_hex(32))"` |
| `SESSION_COOKIE_SECURE` | `1` |

Use **Secrets** (not Variables) for the first two so they aren't shown publicly.

## 4. Push the code

Get a **Write** access token first:
https://huggingface.co/settings/tokens → **New token** → type **Write**.

Then, from the repo root:

```powershell
.\scripts\deploy-hf.ps1 -Space "<your-hf-username>/azsl-sign-demo"
```

When git asks for credentials: **username** = your HF username, **password** =
that access token (not your account password).

> **Why the script and not `git push hf main`?** This repo's history still holds
> the old committed `node_modules` — about **553 MB** of pack objects. The Space
> only needs the current files (~20 MB), so the script pushes a single-commit
> snapshot instead. Much faster, and the Space stays small. Your GitHub history
> is untouched.

The Space starts building immediately — watch the **Logs** tab. The first build
takes **~10 minutes** (torch + MediaPipe). It's ready when the logs show:

```
Auth database ready (PostgreSQL).
WebSocket backend started.
Application startup complete.
```

## 5. Use it

Your public URL is:

```
https://<your-hf-username>-azsl-sign-demo.hf.space
```

Share that link. Anyone, anywhere:

- registers / logs in (stored in your Neon database)
- **Kameranı Başlat** → **SINAĞA BAŞLA** → 3‑2‑1 → sign → prediction
- alphabet mode works too

The recognition runs on Hugging Face's servers, **not** on your laptop — you can
close it and the demo keeps working.

## Redeploying

Commit your changes as usual, then run the same script again — it force-pushes a
fresh snapshot and the Space rebuilds:

```powershell
git add -A && git commit -m "..."
git push origin main                 # GitHub
.\scripts\deploy-hf.ps1 -Space "<your-hf-username>/azsl-sign-demo"
```

The script refuses to run with a dirty working tree, so commit first.

---

## What makes this work

- **`README.md` frontmatter** — the YAML block at the top tells HF this is a
  Docker Space and that the app listens on port 8000. GitHub ignores it.
- **`Dockerfile`** — the same one Fly/Render would use: `python:3.12-slim`,
  `libgl1` + `libglib2.0-0` for MediaPipe/OpenCV, and the CPU-only torch wheel
  so the image isn't several GB.
- **Same-origin** — because the Space serves the pages *and* `/ws`,
  `build_web_layer(app, serves_ws=True)` injects `recognitionWsUrl: ""` and the
  browser authenticates with the normal session cookie. `RECOGNITION_WS_URL` and
  `/api/ws-token` are not used at all here.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| Build fails on `libGL.so.1` | you're not using the repo `Dockerfile` — check the Space SDK is **Docker** |
| App starts then exits, logs show the DB banner | `DATABASE_URL` secret is missing or wrong (needs `+psycopg` and `?sslmode=require`) |
| `git push hf` rejected — auth | use a **Write** access token as the password, not your account password |
| Space is "Sleeping" | free Spaces sleep after 48 h idle; opening the URL wakes it (~30 s) |
| Camera won't start | the page must be HTTPS — `*.hf.space` already is |

## Keeping Vercel too (optional)

You don't need to, but if you want the Vercel URL as the public face, set
`RECOGNITION_WS_URL` on Vercel to `wss://<your-hf-username>-azsl-sign-demo.hf.space`
and redeploy. That's the cross-origin path, so the two must share an identical
`SESSION_SECRET` — see [DEPLOY_RECOGNITION.md](DEPLOY_RECOGNITION.md). Simpler to
just use the Space URL.
