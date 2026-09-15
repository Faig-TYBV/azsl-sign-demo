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
  "socialWsUrl":      "" | "wss://host" | null   // null => no live chat/calls
}
```

`""` means same origin. A URL means a separate host, which the page
authenticates to with a signed token because cookies do not cross origins.
`null` means there is nowhere to connect, so the friends page polls REST and
hides calling.

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

### In-call conversation (sign -> text, speech -> text)

The accessibility feature the product exists for: a Deaf signer and a hearing
speaker hold a conversation, each in their own language, neither typing.

Each participant independently chooses how they *speak*:

| Mode | How it works |
| :--- | :--- |
| **İşarə** (sign) | The call's own camera stream is read directly — no second camera is opened — and MediaPipe runs in the browser (see *Where detection runs* below). Sub-modes: **Hərf** (continuous fingerspelling, any word, fully client-side) and **Söz** (trial-based, the 24-word vocabulary, landmarks sent to the GRU). |
| **Səs** (speech) | The browser's Web Speech API with `az-AZ`. Free, client-side, no API key. Chrome/Edge only. |

Recognised text lands in an **editable compose line** and is sent on Göndər.
Word recognition is ~85% accurate, so roughly one word in seven would
otherwise arrive wrong with no way to retract it. An **Avtomatik göndər**
switch is there for anyone who prefers speed over review.

Captions travel over `/ws/social` as `call:caption`, which verifies the sender
is a party to the call before relaying -- the same rule as SDP. They are also
**saved as ordinary messages**, so the conversation is still in the chat
history after hanging up; someone who depends on captions can scroll back
through what was said.

#### Where detection runs

MediaPipe runs **in the browser** (`@mediapipe/tasks-vision` from CDN, with
`/models/hand_landmarker.task` served by this app). That changes the economics
completely:

| | Before (JPEGs to the server) | Now (browser detection) |
| :--- | :--- | :--- |
| Upstream per signer | ~1.2–2.4 Mbps | **~0** (alphabet) / a few KB per trial (word) |
| Per hour | ~700 MB – 1 GB | negligible |
| Server work | MediaPipe per stream | none (alphabet) / just the GRU (word) |
| Latency | a network round trip per frame | none (alphabet) |

* **Alphabet** is classified entirely client-side by `js/azsl_alphabet.js`
  against `models/azsl_hierarchical_model.json`. Nothing touches the network.
* **Word** still needs the server — the GRU is PyTorch — but the browser
  extracts the 126-dim vectors with `js/azsl_features.js` and sends those
  instead of images. 26 frames ≈ a few KB rather than ~520 KB.

**Parity is enforced, not assumed.** `azsl_features.js` must produce exactly
what `extract_landmarks.normalize_frame()` produces, or the GRU silently
receives features it was never trained on — no error, just worse accuracy.
`tests/test_js_feature_parity.py` runs the JS under node against the Python on
shared inputs (including two-hand slot ordering, degenerate hands, and scale
and translation invariance) and requires agreement to 1e-5. Do not change
either implementation without it passing.

Vectors arriving from a browser are untrusted: the server enforces length 126
and rejects non-finite values, because a NaN would poison the normalizer and
yield a confident-looking prediction from nonsense.

If MediaPipe cannot load — an old browser, or a deployment that excludes the
`.task` bundle — `local.ready` stays false and the original JPEG path is used
unchanged.

Resource notes:

* Frames are processed at ~10 fps, below the workspace page's 15, because the
  same CPU is already encoding WebRTC video.
* In **Söz** mode frames are only sent during a trial. The backend ignores them
  otherwise, and the free instance is 0.1 CPU.
* Hanging up stops recognition and releases the microphone. A mode left on
  would otherwise keep the camera pipeline running for the rest of the session.
* Sign mode needs a video call; speech mode works on audio-only too.

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
