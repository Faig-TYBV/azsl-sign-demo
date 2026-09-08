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

## 3. A PostgreSQL database

Pick one:

**a) Share the team's cloud database (easiest).** Ask Faiq for the Neon
`DATABASE_URL`. Everyone then shares one set of accounts.

**b) Your own local Postgres.** Install PostgreSQL, then the app creates the
database and the `users` table on first start — you only need the server running
and the password right.

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

```bash
py -m uvicorn src.web_demo.backend:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000**. Register, log in, start the camera. One process
serves the pages, the auth API and the `/ws` recognition socket on the same
origin — no tunnel or token needed locally.

You should see:

```
Auth database ready (PostgreSQL).
WebSocket backend started.
```

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
