# Setting up after cloning

```bash
git clone https://github.com/Faig-TYBV/azsl-sign-demo.git
cd azsl-sign-demo
```

Everything the app needs is in the repo — including the trained models
(`gru_24_cap50_best.pt`, `hand_landmarker.task`, `azsl_hierarchical_model.json`).
**The only file you must create yourself is `.env`.**

---

## 1. Python 3.12+

```bash
py --version        # Windows
python3 --version   # macOS / Linux
```

## 2. Install dependencies

```bash
pip install -r requirements.txt -r requirements-recognition.txt
```

- `requirements.txt` — web layer + auth (small)
- `requirements-recognition.txt` — torch, MediaPipe, OpenCV (large, ~10 min)

## 3. A database

> **Sharing a `localhost` password does not work.** `localhost:5432` on your
> teammate's machine is *their* Postgres, not yours — your local database is not
> reachable from another computer. Share the **Neon** URL instead, or have each
> person run their own local Postgres with their own password.

Pick one:

**a) Shared Neon database — recommended, nothing to install.**
Ask Faiq for the `DATABASE_URL`. Everyone shares one set of accounts, and there
is no PostgreSQL to set up locally.

**b) Your own local Postgres.** Install PostgreSQL and use your own password.
The app creates the database and the `users` table on first start. Your accounts
are then separate from everyone else's.

## 4. Create `.env`

```bash
cp .env.example .env        # Windows: copy .env.example .env
```

Then edit it:

```ini
# (a) shared Neon:
DATABASE_URL=postgresql+psycopg://neondb_owner:PASSWORD@ep-xxxx.neon.tech/neondb?sslmode=require
# (b) local Postgres:
DATABASE_URL=postgresql+psycopg://postgres:YOURPASSWORD@localhost:5432/azsl_demo

SESSION_SECRET=<any 64 hex chars>
SESSION_COOKIE_SECURE=0
```

Generate a secret:

```bash
py -c "import secrets; print(secrets.token_hex(32))"
```

`.env` is gitignored — never commit it.

## 5. Run

**This is a Python project — there is no `npm run dev`.** (The old Vite frontend
was removed; the pages are static HTML served by FastAPI.)

### Full app — pages, auth *and* camera recognition

```bash
py -m uvicorn src.web_demo.backend:app --host 0.0.0.0 --port 8000
```

Takes ~15 s to start (loading torch + MediaPipe + the GRU model). You should see:

```
Auth database ready (PostgreSQL).
WebSocket backend started.
```

### Light mode — pages and auth only, no ML

If you are only working on the HTML/CSS/JS or the login flow, run the web layer
on its own:

```bash
py -m uvicorn api.index:app --host 0.0.0.0 --port 8000
```

Starts in ~2 s, and needs **only `requirements.txt`** — you can skip the ~10
minute `requirements-recognition.txt` install entirely. The workspace page loads
and shows a "recognition offline" notice; everything else works normally.

Either way, open **http://localhost:8000**.

### Or: don't run it locally at all

Pushing to `main` redeploys https://azsl-sign-demo.onrender.com automatically.
That's fine for an occasional check, but each rebuild takes ~5-10 minutes, so
it's a poor edit-test loop. Run locally while developing.

---

## Notes

- **Only `SESSION_SECRET` has to match another deployment** if you point a
  deployed page at your local backend; otherwise any value works.
- **Camera needs a secure context.** `localhost` counts as secure, so local dev
  is fine. Over a LAN IP the browser will block `getUserMedia` — use a tunnel
  (see [DEPLOY_RECOGNITION.md](DEPLOY_RECOGNITION.md) option C2) if you need that.
- **The live deployment** is https://azsl-sign-demo.onrender.com and redeploys
  automatically on every push to `main`.
- Deployment guides: [DEPLOY_RECOGNITION.md](DEPLOY_RECOGNITION.md) (Render / Fly /
  local + tunnel), [DEPLOY_VERCEL.md](DEPLOY_VERCEL.md) (pages + auth only),
  [DEPLOY_CLOUDRUN.md](DEPLOY_CLOUDRUN.md).

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `Startup aborted: auth database unavailable` | `DATABASE_URL` wrong — check the password, and that it starts `postgresql+psycopg://` |
| `libEGL.so.1: cannot open shared object file` (Linux) | `sudo apt install libegl1 libgl1 libglib2.0-0` |
| `ModuleNotFoundError: mediapipe` | you skipped `requirements-recognition.txt` |
| Camera button does nothing | not on `localhost`/HTTPS, or permission denied in the browser |
