# Project Status — AzSL Word Recognition

**Status: ACTIVE — live web demo built; friends/chat/calls added (2026-09-15).**
Do not delete, overwrite, reset or refactor existing files without checking first —
the trained checkpoint, curated vocabulary, and extracted feature set are all
production inputs to the running demo, not intermediate scratch.

---

## 1. Current Project Goal

Word-level sign language recognition for Azerbaijani Sign Language (AzSL), served
as a real-time web demo: a browser opens a WebSocket to a backend that runs
MediaPipe hand-landmark extraction + a trained GRU classifier and streams back
predictions live. The ML pipeline (extraction → vocabulary curation → training →
evaluation) is complete for a 24-word vocabulary; current work is the web
product around it (auth, sentence builder, text-to-speech, fingerspelling mode,
friends/chat/calls, deployment).

## 2. Dataset Information

- Source: `data/raw/AzSLD_Words_200/<word_label>/<video_hash>.mp4`
- Total videos: **8,557**, across 200 word classes (hash-named `.mp4` files;
  originals untouched)
- Both one-hand and two-hand signs; `MAX_HANDS = 2`
- Typical length ~9–33 frames (P50/P75 ≈ 26 → chosen target sequence length)

## 3. Feature Extraction — COMPLETE

Full extraction finished on 2026-08-24 (see `outputs/reports/full_extraction_report.txt`):

| Metric | Value |
|---|---|
| Total videos | 8,557 |
| Successfully extracted | 8,557 (0 failed) |
| Feature shape | `(26, 126)` per video |
| Avg detection rate | 94.54% |
| Total processing time | 3.67 h (CPU) |

Pipeline files under `src/features/`: `extract_landmarks.py` (MediaPipe Tasks
`HandLandmarker`, VIDEO mode, ≤2 hands, wrist-relative + scale normalization,
deterministic left/right slot ordering), `preprocess_sequence.py` (uniform
resample/pad to 26 frames + validity mask), `run_full_extraction.py`
(resumable, per-video `.npz`, append-only `extraction_metadata.jsonl`).
Preprocessing version tag: `v1.0-wrist-scale-normalized`.

**GPU note (still applies):** the PyPI Windows MediaPipe wheel has no GPU
(OpenGL/EGL) support — extraction is CPU-only by design. Do not install
CUDA/cuDNN or change the MediaPipe version to chase this.

## 4. Vocabulary Curation & Training — COMPLETE (Experiment 8)

The full 200-class label space was pruned to a **24-word demo vocabulary**
(`outputs/vocabulary_analysis/final_vocabulary.json`) by dropping morphological
variants and confusion attractors that hurt a shared baseline model — e.g.
`MƏNİM`/`MƏNƏ`/`ONUN` around `MƏN`, `SİZİN` around `SİZ`, `İSTƏYİRƏM` around
`İSTƏMƏK`, plus low-sample or low-F1 classes (`GÖRMƏK`, `BİLMƏK`, `O`, `İŞ`).
Full rationale and rejected-class analysis are in that file and in
`outputs/vocabulary_analysis/final_vocabulary_audit.md`.

**Experiment 8 model** (`outputs/vocabulary_24_cap50/EXPERIMENT_8_REPORT.md`):

- 2-layer unidirectional GRU (hidden=128, dropout=0.3), mean+max temporal
  pooling → 256-dim → Linear(256→24); 203,544 params
- Trained on 866 sequences (capped 50/class), validated/tested on untouched
  676/676 splits identical to the 200-class production partition
- **Test accuracy 85.21%, macro-F1 74.21%, weighted-F1 86.03%**
  (vs. 67.31% acc / 65.10% macro-F1 for the old 200-class model on the same
  676 test sequences)
- Checkpoint: `outputs/vocabulary_24_cap50/checkpoints/gru_24_cap50_best.pt`
- Normalizer stats (fit on the 866 training sequences only):
  `outputs/vocabulary_24_cap50/metadata/feature_normalization_stats_24.json`

This checkpoint is what the live backend loads — see §6.

## 5. Web Demo — BUILT AND WIRED (`src/web_demo/`)

- **`webapp.py`** — ML-free layer: session cookies (`itsdangerous`-signed,
  `SessionMiddleware`), register/login/logout/me (`src/web_demo/db.py`,
  SQLAlchemy + argon2 + Postgres), Azerbaijani TTS via `edge-tts`
  (`az-AZ-BanuNeural`, no API key/local model), and page routing
  (`landing.html` → `register.html`/login → `app` workspace). Shared by both
  entrypoints below so torch/mediapipe/opencv never load in the Vercel path.
- **`backend.py`** — full container backend: everything in `webapp.py` plus
  the `/ws` WebSocket. Loads the Experiment 8 checkpoint + normalizer at
  startup, runs a per-connection state machine for **word mode**
  (READY → COUNTDOWN → RECORDING → ANALYZING → RESULT, 26-frame trial,
  hand-presence gate to suppress predictions on an empty/idle buffer) and a
  separate **alphabet (fingerspelling) mode** (`AlphabetClassifier` +
  `AlphabetStabilizer`: confidence floor, stable-frame hold, motion gate).
  *Note:* `src/inference/ambiguity_gate.py` is **not** wired into the live
  backend. It guards the MƏN/MƏNƏ/MƏNİM pronoun cluster, and MƏNƏ/MƏNİM were
  dropped during the 24-word curation — so at most one cluster member can ever
  be predicted and the gate can never fire. It is still used by
  `src/inference/predict.py` for offline analysis.
- **`api/index.py`** — Vercel serverless entrypoint; imports only
  `webapp.py`, so no ML deps ship to the function.
- **Cross-origin auth for `/ws`:** when pages (Vercel) and recognition
  (Fly/Render/Cloud Run) live on different origins, cookies can't cross, so
  the page fetches a short-lived signed token (`/api/ws-token`) and passes it
  on the socket URL instead.
- **Frontend** (`src/web_demo/frontend/`): `landing.html`, `register.html`,
  `index.html` (the workspace: webcam capture, word-mode trial UI, sentence
  builder, TTS playback, recognition-WS wiring driven by
  `window.__AZSL_CONFIG__.recognitionWsUrl`), `js/azsl_alphabet.js`
  (client-side fingerspelling helper), and `friends.html` (the social page —
  same design tokens and glassmorphism cards as `index.html`).

## 5b. Social Layer — friends, chat, calls (added 2026-09-15)

- **`deps.py`** (new) — session config, the signed `/ws` token, and the
  `get_db` / `current_user` / `require_user` dependencies. Extracted from
  `webapp.py` so `social.py` can share them without a circular import;
  `webapp.py` re-exports the names, so `backend.py`'s existing
  `from ...webapp import verify_ws_token` keeps working.
- **`db.py`** — two new tables. `friendships` stores **one row per
  relationship** (not per direction) with status `pending|accepted`; every
  "are these friends?" query checks both column orders. `messages` stores
  sender/recipient/body/`read_at`. Declining an invite deletes the row rather
  than recording a "declined" state, so a pair can try again. `init_db()`
  creates both, so an existing deployment needs no migration.
- **`social.py`** (new) — the REST API (`/api/friends*`, `/api/messages*`,
  `/api/users/search`, `/api/rtc-config`) plus the `/ws/social` socket:
  presence, live chat, typing, and WebRTC call signalling. REST is the source
  of truth; the socket only makes delivery instant, so the Vercel path still
  works by polling.
- **Calls are peer-to-peer (WebRTC).** The server relays SDP offer/answer and
  ICE candidates only — a few KB per call — and the audio/video go browser to
  browser. Call quality therefore does not depend on the instance size.
- **Authorization**: every relay and every message re-checks friendship, and
  call signalling re-checks that the sender is a party to the call it names.
  A pending invite grants nothing.

**Two constraints worth remembering:**

1. **Single process only.** `Hub` and `CallRegistry` are in-memory, so two
   users on different instances behind a load balancer would not see each
   other. Scaling out means putting Redis pub/sub behind `Hub.send_to_user`.
2. **TURN is unconfigured.** STUN alone connects most users; symmetric NAT
   (mobile data, corporate firewalls) needs a TURN relay, which costs
   bandwidth and needs credentials (`TURN_URL` / `TURN_USERNAME` /
   `TURN_CREDENTIAL`). Until those are set, a minority of calls fail —
   deliberately with a clear message rather than a hanging spinner.

## 6. Deployment — three documented paths, pick one as canonical

- `DEPLOY_VERCEL.md` — pages + auth API only (no `/ws`; needs a separate
  recognition backend or the workspace shows "recognition offline")
- `DEPLOY_RECOGNITION.md` — the recognition backend alone (local / Fly.io /
  Render), paired with the Vercel deploy above
- `DEPLOY_CLOUDRUN.md` — whole app on one origin (pages + auth + `/ws`),
  simplest to reason about but loses Vercel's free static/CDN tier
- `render.yaml`, `fly.toml`, `Dockerfile` back the container options

**Open decision:** which of these is the actual target for users, vs. which
are exploratory. Recommend picking one, verifying it end-to-end in a browser,
and treating the others as fallback options in the docs rather than three
equally-live paths.

## 7. Known Gaps / Next Steps

**Closed on 2026-09-15:**

- ~~No automated tests for `src/web_demo`~~ — `tests/test_web_api.py` drives
  auth, friends and chat over real HTTP with real session cookies, and
  `tests/test_social.py` covers the friendship/message rules, ICE config and
  call routing. Suite is now **126 tests** (was 91) and needs no Postgres:
  `tests/conftest.py` points `DATABASE_URL` at a temporary SQLite file.
  *Still untested:* the TTS endpoint (it calls out to Microsoft's service).
- ~~`src/web_demo/README.md` is stale~~ — rewritten to match the current
  backend, and extended to document the social layer.
- ~~Dead streaming-era code in `backend.py`~~ — the per-connection buffers and
  ~12 segmentation constants left over from the pre-trial design were never
  read; removed, and the live hand-presence threshold is now the named
  `MIN_VALID_FRAMES_FOR_INFERENCE` instead of a bare `5`.

**Open:**

1. **TURN server not configured** — calls between users on restrictive
   networks will fail until `TURN_URL` / `TURN_USERNAME` / `TURN_CREDENTIAL`
   are set. This is the one part of the feature that needs a paid or
   self-hosted service; everything else is self-contained. See §5b.
2. **Social state is single-process** — see §5b. Fine for the current demo
   scale; a blocker before running two instances.
3. **End-to-end deploy not yet verified in a browser** — the full stack was
   verified locally (126 unit/integration tests, plus a live two-client socket
   run covering presence, chat delivery and the complete call handshake), but
   the webcam → `/ws` → GRU → TTS round trip and a real camera-to-camera call
   still need confirming on an actual HTTPS deployment. **Calls cannot be
   tested over plain HTTP** — `getUserMedia` needs a secure origin.
4. **Vocabulary is 24 words** — expanding it (more classes, more per-class
   cap) is the natural next ML step if the demo needs a larger sentence
   vocabulary; re-run the same curation methodology
   (`scripts/vocabulary_audit_and_selection.py`,
   `scripts/prepare_vocabulary_24_cap50.py`) against a different class list.

## 8. Important Decisions and Reasons

| Decision | Reason |
|---|---|
| Target length 26 frames | Matches dataset P50/P75 frame counts |
| Uniform temporal resampling (not truncation) | Preserves the whole gesture |
| Keep no-detection frames + validity masks | Missing detections carry temporal info; loss can ignore them |
| Wrist-relative + scale normalization | Invariance to position and camera distance |
| Fixed left/right slot ordering | Consistent feature layout across frames/videos |
| Per-video `.npz` storage (no single huge file) | Resumability, avoids RAM issues, scalable (~0.14 GB total) |
| CPU-only extraction | Windows MediaPipe wheel compiled without GPU support (verified) |
| 24-class vocabulary, cap 50 train/class | Removing morphological variants + confusion attractors and balancing the head class (`MƏN` was 51% of training data) raised accuracy +17.9 pts on an identical test set |
| Normalizer fit on training split only | Avoids val/test leakage into feature scaling stats |
| `webapp.py` split from `backend.py` | Lets the Vercel function stay ML-free and under its size limit while the container backend reuses the same auth/page code |
| Signed short-lived `/api/ws-token` for cross-origin `/ws` | Session cookies don't cross registrable domains (Vercel pages vs. Fly/Render recognition host) |
| Hand-presence + ambiguity gates on top of the GRU | The model has no trained "idle/no-hand" class, so raw softmax always emits a confident-looking top-1 even on an empty buffer |
| `edge-tts` (`az-AZ-BanuNeural`) instead of Web Speech API | Most browsers/OSes ship no `az-AZ` voice locally; edge-tts is free and needs no API key |
| One `friendships` row per relationship, not per direction | Halves the rows and makes "are they friends?" a single query; the requester/addressee split is kept only so the UI can say who asked |
| Declined invite deletes the row | Lets a pair try again later without a "declined" state to reason about or expire |
| WebRTC (peer-to-peer) for calls, server relays signalling only | Media never touches the backend, so call quality is independent of the free-tier instance's 0.1 CPU and bandwidth |
| REST is the source of truth, `/ws/social` is an accelerator | The Vercel deploy can't hold a socket open; chat still works there by polling, with only calling hidden |
| `deps.py` split out of `webapp.py` | `social.py` needs the same auth dependencies; without the split the two modules would import each other |
| Social state in-memory rather than Redis | No extra service for a demo at this scale; the seam is `Hub.send_to_user` when it needs to scale |

---

## Reproducibility — How to Continue

```powershell
# ML pipeline is complete; only re-run these if you're touching the vocabulary or model.

# Re-run vocabulary curation / training for a different class list
python scripts/vocabulary_audit_and_selection.py
python scripts/prepare_vocabulary_24_cap50.py
python scripts/run_experiment_8_training.py
python scripts/verify_vocabulary_24_cap50.py   # audits the checkpoint before it's trusted in backend.py

# Run the web demo locally (full backend: pages + auth + /ws recognition)
python -m uvicorn src.web_demo.backend:app --host 0.0.0.0 --port 8000
# then open http://localhost:8000 , register/login, and go to /app

# Run the test suite (126 tests: ML modules + the web/social layer)
pytest
```
