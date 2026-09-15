# AzSL Web Demo — `src/web_demo/`

The web product around the AzSL recognition models: sign-language recognition
in the browser, plus accounts, friends, chat and calls.

Everything here is served by FastAPI. There is no build step and no npm — the
frontend is hand-written HTML/CSS/JS served as static pages.

---

## Modules

| File | Role | Imports the ML stack? |
| :--- | :--- | :---: |
| `deps.py` | Session config, signed `/ws` token, `get_db` / `current_user` / `require_user` | no |
| `db.py` | SQLAlchemy models + queries: users, friendships, messages | no |
| `webapp.py` | Pages, auth API, Azerbaijani TTS, and the wiring in `build_web_layer()` | no |
| `social.py` | Friends/chat REST, and the `/ws/social` socket (presence, live chat, call signalling) | no |
| `backend.py` | Everything above **plus** the `/ws` recognition socket (MediaPipe + the GRU) | **yes** |

The split exists so the Vercel function can import `webapp.py` without pulling
in torch, MediaPipe and OpenCV — which would blow past its 250 MB limit.

### Two entrypoints

```bash
# Full app: pages, auth, friends/chat/calls, AND camera recognition.
py -m uvicorn src.web_demo.backend:app --host 0.0.0.0 --port 8000

# Light: everything except recognition. Starts in ~2 s, needs only
# requirements.txt (no torch/MediaPipe).
py -m uvicorn api.index:app --host 0.0.0.0 --port 8000
```

---

## Pages

| Route | File | Notes |
| :--- | :--- | :--- |
| `/` | `frontend/landing.html` | Public marketing page |
| `/login`, `/register` | `frontend/register.html` | Redirects to `/app` when already signed in |
| `/app` | `frontend/index.html` | The recognition workspace — auth required |
| `/friends` | `frontend/friends.html` | Friends, chat and calls — auth required |

`/app` and `/friends` are served through `_serve_configured()`, which injects
`window.__AZSL_CONFIG__` before `</head>`:

```js
{
  "recognitionWsUrl": "" | "wss://host" | null,  // null => recognition offline
  "socialWs": true | false                       // false => no live chat/calls
}
```

That is how one set of HTML files serves both the container deploy (everything
on) and the Vercel deploy (pages + auth + REST chat, no sockets).

---

## Recognition (`/ws`)

Loads the Experiment 8 checkpoint (24 classes) and the feature normalizer once
at import, then runs a per-connection state machine.

**Word mode** is trial-based, not continuous:

```
READY ──START──> COUNTDOWN (3·2·1) ──> RECORDING (26 frames) ──> ANALYZING ──> RESULT
```

The 26 frames are normalized with the training statistics and classified in one
shot. A trial holding fewer than `MIN_VALID_FRAMES_FOR_INFERENCE` real hand
detections reports `ƏL AŞKARLANMADI` instead of running the model — the GRU has
no "idle" class, so it would otherwise return a confident-looking label for an
empty buffer.

**Alphabet mode** classifies one frame at a time through `AlphabetClassifier`,
behind three gates in `AlphabetStabilizer`: a confidence floor, a stable-hold
requirement (~0.7 s), and a motion gate so a hand travelling between poses
cannot commit a letter.

---

## Social (`/ws/social`)

Registered only when `build_web_layer(serves_ws=True)`. Authenticates from the
session cookie, or a signed `?token=` for a cross-origin page.

- **Presence** — broadcast to your friends when your first socket opens and
  your last one closes. A user may hold several sockets (phone + laptop + tabs).
- **Chat** — messages are written to Postgres first, then fanned out. The REST
  API is the source of truth; the socket only makes delivery instant.
- **Calls** — WebRTC. The browsers exchange an SDP offer/answer and ICE
  candidates *through* this server and then send audio/video **directly to each
  other**. Only signalling text crosses the backend, so call quality does not
  depend on the instance size.

Every relay re-checks that the two parties are actually friends, and that the
sender belongs to the call it names.

```
A: call:invite ──> server ──> B: call:incoming
                              B: call:accept ──> A: call:accepted
A: call:offer  ──> B          B: call:answer ──> A
A/B: call:ice  <──>           (media now flows peer-to-peer)
A/B: call:end  ──> the other side: call:ended
```

**State is per-process and in memory.** With more than one instance behind a
load balancer, two users on different instances would not see each other.
Scaling past one instance means putting Redis pub/sub behind `Hub.send_to_user`;
nothing else would change.

### TURN

STUN alone (the default) connects most pairs of users. Symmetric NAT — common
on mobile data and corporate networks — needs a TURN relay, which always costs
bandwidth and therefore always needs credentials:

```bash
TURN_URL=turn:turn.example.com:3478
TURN_USERNAME=...
TURN_CREDENTIAL=...
```

Without them calls simply fail to connect for the affected users; the UI says
so rather than hanging on a spinner. `/api/rtc-config` reports whether a relay
is configured.

> **Calls need HTTPS.** `getUserMedia` is only exposed on a secure origin, so
> camera and microphone work on `localhost` and on any HTTPS deployment, but
> never on plain HTTP over a LAN IP.

---

## API

| Method | Path | Purpose |
| :--- | :--- | :--- |
| POST | `/api/register`, `/api/login`, `/api/logout` | Session auth (argon2) |
| GET | `/api/me` | Current user |
| GET | `/api/ws-token` | Short-lived signed token for a cross-origin socket |
| POST | `/api/tts` | Azerbaijani speech (`edge-tts`, `az-AZ-BanuNeural`) |
| GET | `/api/friends` | Friends, with online flag and unread counts |
| GET | `/api/friends/requests` | Incoming + outgoing invites |
| GET | `/api/users/search?q=` | Find people (min. 2 characters) |
| POST | `/api/friends/request` \| `/respond` \| `/cancel` \| `/remove` | Manage relationships |
| GET | `/api/messages/{friend_id}` | Conversation (also marks it read) |
| POST | `/api/messages` | Send without a socket (REST fallback) |
| GET | `/api/rtc-config` | ICE servers + whether TURN is available |
| GET | `/health` | Liveness; deliberately touches no database |

---

## Data model

```
users ──┬─< friendships >─┬── users      one row per relationship, either
        │                 │              direction; status pending|accepted
        └─< messages    >─┘              sender/recipient, body, read_at
```

A declined invite deletes its row rather than storing a "declined" state, so
the pair can try again. Blocking is deliberately not modelled.

`init_db()` creates all three tables on first start, so adding friends and chat
to an existing deployment needs no migration step.

---

## Tests

```bash
py -m pytest tests/ -q          # 126 tests
```

- `tests/test_web_api.py` — auth, friends and chat over real HTTP with real
  session cookies (this is where a missing auth check would fail)
- `tests/test_social.py` — friendship/message rules, ICE config, call routing
- `tests/conftest.py` points `DATABASE_URL` at a temporary SQLite file, so the
  suite runs with no Postgres up

---

## Configuration

| Variable | Required | Purpose |
| :--- | :---: | :--- |
| `DATABASE_URL` | yes | `postgresql+psycopg://user:pass@host:5432/db` |
| `SESSION_SECRET` | yes | Signs the session cookie **and** the `/ws` tokens — must match across hosts in a split deploy |
| `SESSION_COOKIE_SECURE` | on HTTPS | `1` marks the cookie `Secure` |
| `RECOGNITION_WS_URL` | optional | Point the page at a separate recognition backend |
| `TURN_URL` / `TURN_USERNAME` / `TURN_CREDENTIAL` | optional | WebRTC relay for restrictive networks |
| `STUN_URLS` | optional | Comma-separated override of the default STUN servers |
