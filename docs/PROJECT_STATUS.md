# Project Status — AzSL Word Recognition

**Status: ACTIVE — live web demo built, in frontend/voice polish stage (2026-09-12).**
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
deployment).

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
  (READY → COUNTDOWN → RECORDING → RESULT, 26-frame buffer, ambiguity gate,
  hand-presence gate to suppress predictions on an empty/idle buffer) and a
  separate **alphabet (fingerspelling) mode** (`AlphabetClassifier` +
  `AlphabetStabilizer`: confidence floor, stable-frame hold, motion gate).
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
  (client-side fingerspelling helper).

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

1. **No automated tests for `src/web_demo`** — `tests/` covers only the ML
   modules (features, models, inference gates). Auth (register/login/session),
   password hashing, and the TTS endpoint have no test coverage despite
   handling user credentials.
2. **`src/web_demo/README.md` is stale** — it still describes the old
   Experiment 2 / 200-class checkpoint (`outputs/checkpoints/gru_temporal_pool_best.pt`)
   and a no-auth local-Wi-Fi-only run flow. It should be rewritten to match
   `backend.py` (Experiment 8 checkpoint, auth-gated `/app`, both word +
   alphabet modes).
3. **End-to-end deploy not yet verified** — recent commits are frontend/voice
   iteration (`index.html`, TTS wiring); confirm the full webcam → `/ws` →
   GRU → TTS round trip on an actual deployment, not just locally.
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

# Run the test suite (ML modules only today — see §7.1)
pytest
```
