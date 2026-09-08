"""
WebSocket backend for iPhone camera real-time sign language detection.
Reuses existing preprocessing, normalization, model loading, and class mapping.
"""

import asyncio
import json
import base64
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict

import cv2
import mediapipe as mp
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, status

# Windows consoles default to a legacy codepage (cp1252 / cp1251) that cannot
# encode the Azerbaijani characters in our log lines (e.g. "BAŞLA!"). Without
# this, print() raises UnicodeEncodeError inside the countdown task and the
# trial never reaches the RECORDING state — the frame counter stays at 0/26.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# Make project-root imports work
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Absolute path to the MediaPipe hand-landmarker bundle. create_hand_landmarker()
# otherwise resolves "models/hand_landmarker.task" relative to the process CWD,
# which only exists if the server is launched from a directory that happens to
# have a models/ folder — so the /ws socket would accept then immediately close.
HAND_LANDMARKER_TASK = PROJECT_ROOT / "src" / "models" / "hand_landmarker.task"

from src.features.extract_landmarks import (
    create_hand_landmarker,
    normalize_frame,
    _mp_result_to_frame_result,
)
from src.features.preprocess_sequence import (
    TARGET_SEQ_LEN,
    preprocess_sequence,
)
from src.models.gru_classifier import GRUClassifier
from src.inference.ambiguity_gate import is_ambiguous_prediction
from src.inference.alphabet_classifier import AlphabetClassifier, AlphabetStabilizer
from src.data.normalization import FeatureNormalizer
from src.inference.predict import get_device
from src.web_demo import db as auth_db
from src.web_demo.webapp import build_web_layer, verify_ws_token

app = FastAPI(title="AzSLD Web Demo Backend")

# Session cookie + /api/* auth routes + static pages. Same layer the Vercel
# entrypoint (api/index.py) uses; this backend also serves /ws on the same
# origin, so pass serves_ws=True.
build_web_layer(app, serves_ws=True)

# Load the Experiment 8 (24-class Cap-50) word model and normalizer at startup
print("Loading Experiment 8 word model and normalizer...", flush=True)
device = get_device()
checkpoint_path = (
    PROJECT_ROOT
    / "outputs/vocabulary_24_cap50/checkpoints/gru_24_cap50_best.pt"
)
normalizer_path = (
    PROJECT_ROOT
    / "outputs/vocabulary_24_cap50/metadata/feature_normalization_stats_24.json"
)

try:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not normalizer_path.is_file():
        raise FileNotFoundError(f"Normalizer stats not found: {normalizer_path}")

    normalizer = FeatureNormalizer.load(normalizer_path)
    if normalizer.mean.shape != (126,) or normalizer.std.shape != (126,):
        raise ValueError(
            f"Normalizer shape mismatch: mean={normalizer.mean.shape}, std={normalizer.std.shape} (expected 126)"
        )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" not in ckpt or "model_config" not in ckpt:
        raise KeyError("Checkpoint missing 'model_state_dict' or 'model_config'.")

    cfg = ckpt["model_config"]
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = ckpt["idx_to_class"]

    if len(class_to_idx) != 24 or len(idx_to_class) != 24:
        raise ValueError(
            f"Expected 24 classes, got {len(class_to_idx)} in class_to_idx and {len(idx_to_class)} in idx_to_class"
        )

    model = GRUClassifier(**cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(
        f"Experiment 8 model (24 classes) and normalizer loaded successfully on {device}",
        flush=True,
    )
except Exception as e:
    print(f"Failed to load model or normalizer: {e}", flush=True)
    raise

print("Loading alphabet classifier...", flush=True)
alphabet_classifier = AlphabetClassifier()
print("Alphabet classifier loaded successfully.", flush=True)

# Constants for prediction smoothing
WINDOW_SIZE = 5
CONFIDENCE_THRESHOLD = 0.35

# --- Alphabet (fingerspelling) mode tuning ------------------------------------
# The alphabet MLP has no "not-a-letter" class, so its softmax saturates and the
# confidence gate alone can't stop garbage from accumulating ("FFAX" while the
# user isn't even signing). These three knobs do the real work:
#   * a long stable-hold requirement (~0.7 s at 15 fps, matches the UI hint)
#   * a motion gate: a hand travelling between poses can't commit anything
#   * a high confidence floor
ALPHA_MIN_CONF = 0.82           # per-frame letter confidence required
ALPHA_STABILITY_FRAMES = 10     # consecutive agreeing frames to commit (~0.66 s @ 15 fps)
ALPHA_MOTION_THRESH = 0.030     # wrist travel (normalised image units) per frame; above => "moving"
# MediaPipe reports handedness as if the image were mirrored (selfie view); the
# frontend sends a raw, un-mirrored frame, so the Left/Right label is inverted
# relative to reality. Flip it back before the classifier mirrors the hand.
# If letters come out consistently wrong, set this to False.
ALPHA_HANDEDNESS_IS_MIRRORED = True

# Server-side logging verbosity.
#   False (default): only operational / error / hand-presence events print
#                    to the console. Per-frame debug noise is silenced to
#                    keep the server console usable at 15-30 fps.
#   True:            every frame prints (JPEG decoded, MediaPipe processed,
#                    FRAME RECEIVED, INFERENCE, Prediction: ..., response
#                    sent) — useful when debugging the WebSocket frame
#                    pipeline itself.
DEBUG_LOG = False


def _dbg(msg: str) -> None:
    """Per-frame debug printer; silenced unless DEBUG_LOG=True."""
    if DEBUG_LOG:
        print(msg, flush=True)

# Temporal segmentation thresholds (additive layer; does not change inference).
PRESENCE_ON_FRAMES = 2      # consecutive hand-present frames to enter SIGNING
PRESENCE_OFF_FRAMES = 6     # consecutive end-condition frames to exit SIGNING
MOTION_DELTA_MIN = 0.02     # mean L2 feature delta required to start signing
MOTION_WINDOW = 5           # number of frames to average the motion delta over
MOTION_DELTA_FLOOR = 0.001  # below this motion we treat the hand as still
IDLE_CONFIDENCE_MAX = 0.10  # minimum smoothed confidence to consider a segment valid
STABLE_PRED_FRAMES = 2      # consecutive agreeing smoothed predictions required
MIN_SEGMENT_FRAMES = 15     # minimum frames between consecutive emits
COOLDOWN_FRAMES = 8         # minimum cooldown frames after an emit

# === Hand-presence gate (post-prediction, no retraining required) ============
# The GRU was trained without an explicit "idle/background" class, so softmax
# ALWAYS returns a top-1 even on all-zero feature buffers (confidence ~1/N).
# We compensate by NOT running inference when the buffer has no valid frames,
# and by clamping the displayed fields when the most recent frame is not a
# real hand detection. These thresholds are tuning knobs, not architecture.
SEGMENT_CONFIDENCE_FLOOR = 0.35   # smoothed conf must be >= this to emit
DISPLAY_CONFIDENCE_FLOOR = 0.35   # below this we display "-" instead of the label
MIN_VALID_FRAMES_IN_BUFFER = 1    # skip inference if buffer has 0 valid frames

class ConnectionState:
    """State per WebSocket connection."""

    def __init__(self):
        self.active_mode = "word"

        # --- Word Mode Trial State Machine ---
        self.word_state = "READY"  # "READY" | "COUNTDOWN" | "RECORDING" | "RESULT"
        self.countdown_task = None
        self.countdown_val = None
        self.countdown_text = ""
        self.countdown_title = ""
        self.trial_seq = 0
        self.recorded_features = []
        self.recorded_validity = []
        self.last_word_result = None

        # --- Alphabet Mode Stabilization Engine ---
        self.stabilizer = AlphabetStabilizer(
            min_confidence=ALPHA_MIN_CONF, stability_frames=ALPHA_STABILITY_FRAMES
        )
        # Previous-frame wrist (x, y) in normalised image space, for the motion gate.
        self.alpha_prev_wrist = None

        self.landmarker = None
        self.last_timestamp_ms = 0
        self.frame_count = 0

        # Legacy buffers for compatibility
        self.feature_buffer = deque(maxlen=TARGET_SEQ_LEN)
        self.validity_buffer = deque(maxlen=TARGET_SEQ_LEN)
        self.pred_history = deque(maxlen=WINDOW_SIZE)
        self.smoothed_prediction = None
        self.smoothed_confidence = 0.0

        # Temporal segmentation
        self.seg_state = "IDLE"
        self.presence_on_count = 0
        self.presence_off_count = 0
        self.motion_window = deque(maxlen=MOTION_WINDOW)
        self.last_segment_label = None
        self.frames_since_last_emit = 10**9
        self.stable_pred_count = 0
        self.prev_smoothed_prediction = None
        self.last_is_ambiguous = False


@app.on_event("startup")
async def startup_event():
    # Prepare the PostgreSQL auth store (creates the database + users table on
    # first run). A bad DATABASE_URL fails loudly here rather than at first login.
    try:
        auth_db.init_db()
        print("Auth database ready (PostgreSQL).", flush=True)
    except auth_db.DatabaseUnavailable as exc:
        print("\n" + "=" * 70, flush=True)
        print(str(exc), flush=True)
        print("=" * 70 + "\n", flush=True)
        # Terse re-raise — the actionable message is the banner above, not a
        # SQLAlchemy stack trace.
        raise SystemExit("Startup aborted: auth database unavailable (see above).")
    print("WebSocket backend started.", flush=True)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # Two ways in, so the socket is gated like the /app page either way:
    #   * same origin  -> SessionMiddleware populates websocket.session
    #   * cross origin -> ?token=... signed by the page's origin with the same
    #                     SESSION_SECRET (cookies can't cross domains)
    uid = websocket.session.get("uid")
    if uid is None:
        uid = verify_ws_token(websocket.query_params.get("token", ""))
    if uid is None:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.accept()
    print("Client connected", flush=True)

    state = ConnectionState()

    # Greet the client IMMEDIATELY, before doing any slow work. Building the
    # MediaPipe landmarker costs ~300 ms of CPU, which on a throttled instance
    # (Render free is 0.1 CPU) becomes seconds of wall time — and doing it first
    # left the socket silent long enough that the proxy in front of us closed it
    # as idle (observed: handshake at 0.3 s, killed at 20.3 s with code 1006).
    await websocket.send_text(json.dumps({
        "status": "CONNECTING",
        "message": "Model hazırlanır...",
    }))

    # ...then build it off the event loop, so the health check and other
    # sockets stay responsive while this one initialises.
    try:
        state.landmarker = await asyncio.to_thread(
            create_hand_landmarker, HAND_LANDMARKER_TASK
        )
        print("Landmarker ready for client", flush=True)
    except FileNotFoundError as e:
        await websocket.send_text(json.dumps({"error": str(e)}))
        await websocket.close()
        return
    except Exception as e:  # noqa: BLE001 - surface the reason instead of a bare 1006
        print(f"[WS] landmarker init failed: {type(e).__name__}: {e}", flush=True)
        await websocket.send_text(json.dumps({"error": f"landmarker init failed: {e}"}))
        await websocket.close()
        return

    await websocket.send_text(json.dumps({
        "status": "READY",
        "word_state": "READY",
        "mode": state.active_mode,
        "frame_index": 0,
        "total_frames": TARGET_SEQ_LEN,
        "valid_frames": 0,
        "buffer_progress": f"0/{TARGET_SEQ_LEN}",
        "prediction": "-",
        "confidence": 0.0,
        "message": "Connected. Ready to start trial."
    }))

    async def run_countdown(this_seq: int):
        try:
            state.word_state = "COUNTDOWN"
            state.last_word_result = None
            state.recorded_features = []
            state.recorded_validity = []
            state.countdown_title = "HAZIRLAŞIN..."
            state.countdown_text = "3"
            state.countdown_val = 3
            print(f"[WORD TRIAL {this_seq}] 3.0s countdown started: HAZIRLAŞIN... (3)", flush=True)
            await websocket.send_text(json.dumps({
                "mode": "word",
                "word_state": "COUNTDOWN",
                "countdown_title": "HAZIRLAŞIN...",
                "countdown_text": "3",
                "countdown_val": 3,
                "frame_index": 0,
                "total_frames": TARGET_SEQ_LEN,
                "valid_frames": 0,
                "buffer_progress": f"0/{TARGET_SEQ_LEN}",
                "buffer_full": False,
                "hand_present": False,
                "prediction": "-",
                "confidence": 0.0,
                "raw_prediction": "-",
                "raw_confidence": 0.0,
                "smoothed_prediction": "-",
                "smoothed_confidence": 0.0,
            }))

            await asyncio.sleep(1.0)
            if state.trial_seq != this_seq or state.active_mode != "word":
                return

            print(f"[WORD TRIAL {this_seq}] 2.0s remaining: (2)", flush=True)
            state.countdown_text = "2"
            state.countdown_val = 2
            await websocket.send_text(json.dumps({
                "mode": "word",
                "word_state": "COUNTDOWN",
                "countdown_title": "HAZIRLAŞIN...",
                "countdown_text": "2",
                "countdown_val": 2,
                "frame_index": 0,
                "total_frames": TARGET_SEQ_LEN,
                "valid_frames": 0,
                "buffer_progress": f"0/{TARGET_SEQ_LEN}",
                "buffer_full": False,
                "hand_present": False,
                "prediction": "-",
                "confidence": 0.0,
                "raw_prediction": "-",
                "raw_confidence": 0.0,
                "smoothed_prediction": "-",
                "smoothed_confidence": 0.0,
            }))

            await asyncio.sleep(1.0)
            if state.trial_seq != this_seq or state.active_mode != "word":
                return

            print(f"[WORD TRIAL {this_seq}] 1.0s remaining: (1)", flush=True)
            state.countdown_text = "1"
            state.countdown_val = 1
            await websocket.send_text(json.dumps({
                "mode": "word",
                "word_state": "COUNTDOWN",
                "countdown_title": "HAZIRLAŞIN...",
                "countdown_text": "1",
                "countdown_val": 1,
                "frame_index": 0,
                "total_frames": TARGET_SEQ_LEN,
                "valid_frames": 0,
                "buffer_progress": f"0/{TARGET_SEQ_LEN}",
                "buffer_full": False,
                "hand_present": False,
                "prediction": "-",
                "confidence": 0.0,
                "raw_prediction": "-",
                "raw_confidence": 0.0,
                "smoothed_prediction": "-",
                "smoothed_confidence": 0.0,
            }))

            await asyncio.sleep(1.0)
            if state.trial_seq != this_seq or state.active_mode != "word":
                return

            # Countdown complete -> begin recording 26 frames
            state.recorded_features = []
            state.recorded_validity = []
            state.countdown_title = "BAŞLA!"
            state.countdown_text = "BAŞLA!"
            state.countdown_val = 0
            state.word_state = "RECORDING"
            print(f"[WORD TRIAL {this_seq}] BAŞLA! Recording started (0/26 frames)", flush=True)
            await websocket.send_text(json.dumps({
                "mode": "word",
                "word_state": "RECORDING",
                "countdown_title": "BAŞLA!",
                "countdown_text": "BAŞLA!",
                "countdown_val": 0,
                "frame_index": 0,
                "total_frames": TARGET_SEQ_LEN,
                "valid_frames": 0,
                "buffer_progress": f"0/{TARGET_SEQ_LEN}",
                "buffer_full": False,
                "hand_present": False,
                "prediction": "-",
                "confidence": 0.0,
                "raw_prediction": "-",
                "raw_confidence": 0.0,
                "smoothed_prediction": "-",
                "smoothed_confidence": 0.0,
            }))
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[WORD TRIAL COUNTDOWN ERROR] {e}", flush=True)

    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "Invalid JSON"}))
                continue

            msg_type = message.get("type")

            # Mode switch and state isolation
            mode = message.get("mode", state.active_mode)
            if mode != state.active_mode:
                state.active_mode = mode
                if state.countdown_task is not None and not state.countdown_task.done():
                    state.countdown_task.cancel()
                state.trial_seq += 1
                state.word_state = "READY"
                state.recorded_features = []
                state.recorded_validity = []
                state.last_word_result = None
                state.countdown_val = None
                state.countdown_text = ""
                state.countdown_title = ""
                state.stabilizer.reset()
                _dbg(f"Mode switched to: {mode} (all states reset)")

            # Control commands
            if msg_type in ("start", "start_trial"):
                if state.active_mode == "word":
                    if state.word_state in ("COUNTDOWN", "RECORDING"):
                        _dbg(f"Ignored start command while in {state.word_state}")
                        continue
                    if state.countdown_task is not None and not state.countdown_task.done():
                        state.countdown_task.cancel()
                    state.trial_seq += 1
                    state.word_state = "COUNTDOWN"
                    state.last_word_result = None
                    state.recorded_features = []
                    state.recorded_validity = []
                    state.countdown_task = asyncio.create_task(run_countdown(state.trial_seq))
                    continue

            elif msg_type in ("skip", "reset"):
                if state.active_mode == "word":
                    if state.countdown_task is not None and not state.countdown_task.done():
                        state.countdown_task.cancel()
                    state.trial_seq += 1
                    state.word_state = "READY"
                    state.recorded_features = []
                    state.recorded_validity = []
                    state.last_word_result = None
                    state.countdown_val = None
                    state.countdown_text = ""
                    state.countdown_title = ""
                    await websocket.send_text(json.dumps({
                        "mode": "word",
                        "word_state": "READY",
                        "status": "reset",
                        "frame_index": 0,
                        "total_frames": TARGET_SEQ_LEN,
                        "valid_frames": 0,
                        "buffer_progress": f"0/{TARGET_SEQ_LEN}",
                        "buffer_full": False,
                        "hand_present": False,
                        "prediction": "-",
                        "confidence": 0.0,
                        "raw_prediction": "-",
                        "raw_confidence": 0.0,
                        "smoothed_prediction": "-",
                        "smoothed_confidence": 0.0,
                    }))
                    continue
                else:
                    state.stabilizer.reset()
                    await websocket.send_text(json.dumps({
                        "mode": "alphabet",
                        "status": "reset",
                        "spelled_word": "",
                        "accepted_letter": "-",
                    }))
                    continue

            elif msg_type == "alphabet_action":
                action = message.get("action")
                if action == "clear":
                    state.stabilizer.clear_word()
                elif action == "backspace":
                    state.stabilizer.backspace()
                await websocket.send_text(json.dumps({
                    "mode": "alphabet",
                    "spelled_word": state.stabilizer.spelled_word,
                    "accepted_letter": state.stabilizer.accepted_letter or "-",
                }))
                continue

            elif msg_type == "get_state":
                if state.active_mode == "word":
                    res = state.last_word_result or {}
                    pred = res.get("prediction", "-")
                    conf = res.get("confidence", 0.0)
                    vf = res.get("valid_frames", 0)
                    cur_frames = len(state.recorded_features)
                    await websocket.send_text(json.dumps({
                        "mode": "word",
                        "word_state": state.word_state,
                        "frame_index": cur_frames,
                        "total_frames": TARGET_SEQ_LEN,
                        "valid_frames": vf,
                        "buffer_progress": f"{cur_frames}/{TARGET_SEQ_LEN}",
                        "prediction": pred,
                        "confidence": conf,
                        "raw_prediction": pred,
                        "raw_confidence": conf,
                    }))
                else:
                    await websocket.send_text(json.dumps({
                        "mode": "alphabet",
                        "spelled_word": state.stabilizer.spelled_word,
                        "accepted_letter": state.stabilizer.accepted_letter or "-",
                    }))
                continue

            elif msg_type == "frame":
                # Check Word Mode states
                if state.active_mode == "word":
                    if state.word_state == "READY":
                        await websocket.send_text(json.dumps({
                            "mode": "word",
                            "word_state": "READY",
                            "status_text": "START TRIAL gözlənilir",
                            "frame_index": 0,
                            "total_frames": TARGET_SEQ_LEN,
                            "valid_frames": 0,
                            "buffer_progress": f"0/{TARGET_SEQ_LEN}",
                            "buffer_full": False,
                            "hand_present": False,
                            "prediction": "-",
                            "confidence": 0.0,
                            "raw_prediction": "-",
                            "raw_confidence": 0.0,
                            "smoothed_prediction": "-",
                            "smoothed_confidence": 0.0,
                        }))
                        continue

                    elif state.word_state == "COUNTDOWN":
                        await websocket.send_text(json.dumps({
                            "mode": "word",
                            "word_state": "COUNTDOWN",
                            "countdown_title": state.countdown_title,
                            "countdown_text": state.countdown_text,
                            "countdown_val": state.countdown_val,
                            "frame_index": 0,
                            "total_frames": TARGET_SEQ_LEN,
                            "valid_frames": 0,
                            "buffer_progress": f"0/{TARGET_SEQ_LEN}",
                            "buffer_full": False,
                            "hand_present": False,
                            "prediction": "-",
                            "confidence": 0.0,
                            "raw_prediction": "-",
                            "raw_confidence": 0.0,
                            "smoothed_prediction": "-",
                            "smoothed_confidence": 0.0,
                        }))
                        continue

                    elif state.word_state == "ANALYZING":
                        await websocket.send_text(json.dumps({
                            "mode": "word",
                            "word_state": "ANALYZING",
                            "frame_index": TARGET_SEQ_LEN,
                            "total_frames": TARGET_SEQ_LEN,
                            "valid_frames": int(sum(state.recorded_validity)) if state.recorded_validity else 0,
                            "buffer_progress": f"{TARGET_SEQ_LEN}/{TARGET_SEQ_LEN}",
                            "buffer_full": True,
                            "hand_present": True,
                            "prediction": "-",
                            "confidence": 0.0,
                            "raw_prediction": "-",
                            "raw_confidence": 0.0,
                            "smoothed_prediction": "-",
                            "smoothed_confidence": 0.0,
                        }))
                        continue

                    elif state.word_state == "RESULT":
                        # In RESULT state, one inference has already completed.
                        # Do NOT run new inference; return stable result.
                        res = state.last_word_result or {}
                        pred = res.get("prediction", "-")
                        conf = res.get("confidence", 0.0)
                        vf = res.get("valid_frames", 0)
                        await websocket.send_text(json.dumps({
                            "mode": "word",
                            "word_state": "RESULT",
                            "frame_index": TARGET_SEQ_LEN,
                            "total_frames": TARGET_SEQ_LEN,
                            "valid_frames": vf,
                            "buffer_progress": f"{TARGET_SEQ_LEN}/{TARGET_SEQ_LEN}",
                            "buffer_full": True,
                            "hand_present": pred != "ƏL AŞKARLANMADI",
                            "prediction": pred,
                            "confidence": conf,
                            "raw_prediction": pred,
                            "raw_confidence": conf,
                            "smoothed_prediction": pred,
                            "smoothed_confidence": conf,
                        }))
                        continue

                # Frame decoding for RECORDING (Word) or ALPHABET mode
                base64_data = message.get("data", "")
                if "," in base64_data:
                    base64_data = base64_data.split(",", 1)[1]
                try:
                    image_bytes = base64.b64decode(base64_data)
                except Exception:
                    await websocket.send_text(json.dumps({"error": "Invalid base64"}))
                    continue

                nparr = np.frombuffer(image_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    await websocket.send_text(json.dumps({"error": "Failed to decode image"}))
                    continue

                state.frame_count += 1
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                current_timestamp_ms = int(asyncio.get_event_loop().time() * 1000)
                timestamp_ms = max(current_timestamp_ms, state.last_timestamp_ms + 1)
                state.last_timestamp_ms = timestamp_ms

                result = state.landmarker.detect_for_video(mp_image, timestamp_ms)
                frame_result = _mp_result_to_frame_result(result)

                # ALPHABET MODE: single-frame inference with temporal stabilization
                if state.active_mode == "alphabet":
                    t0 = time.perf_counter()
                    has_hand = frame_result.num_hands > 0
                    raw_letter = None
                    raw_conf = 0.0
                    is_moving = False

                    if has_hand:
                        hand_lm = frame_result.landmarks[0]

                        # Motion gate: how far did the wrist travel since the
                        # last frame? A hand in transit between poses must not
                        # be allowed to commit a letter.
                        wrist_xy = (float(hand_lm[0][0]), float(hand_lm[0][1]))
                        if state.alpha_prev_wrist is not None:
                            dx = wrist_xy[0] - state.alpha_prev_wrist[0]
                            dy = wrist_xy[1] - state.alpha_prev_wrist[1]
                            is_moving = (dx * dx + dy * dy) ** 0.5 > ALPHA_MOTION_THRESH
                        state.alpha_prev_wrist = wrist_xy

                        hand_label = (
                            frame_result.handedness[0] if frame_result.handedness else "Right"
                        )
                        if ALPHA_HANDEDNESS_IS_MIRRORED:
                            hand_label = "Left" if hand_label == "Right" else "Right"

                        raw_letter, raw_conf = alphabet_classifier.predict_frame(
                            hand_lm, handedness=hand_label, min_confidence=ALPHA_MIN_CONF
                        )
                    else:
                        state.alpha_prev_wrist = None

                    stab_result = state.stabilizer.update(
                        raw_letter=raw_letter,
                        raw_confidence=raw_conf,
                        hand_present=has_hand,
                        is_moving=is_moving,
                    )
                    total_ms = (time.perf_counter() - t0) * 1000.0

                    response = {
                        "mode": "alphabet",
                        "hand_present": has_hand,
                        "raw_prediction": stab_result["raw_letter"],
                        "raw_confidence": round(stab_result["raw_confidence"], 2),
                        "stable_candidate": stab_result["stable_candidate"],
                        "candidate_progress": stab_result["candidate_progress"],
                        "accepted_letter": stab_result["accepted_letter"],
                        "just_accepted": stab_result["just_accepted"],
                        "spelled_word": stab_result["spelled_word"],
                        "alphabet_prediction": stab_result["accepted_letter"],
                        "alphabet_confidence": round(stab_result["raw_confidence"], 2),
                        "smoothed_prediction": stab_result["accepted_letter"],
                        "smoothed_confidence": round(stab_result["raw_confidence"], 2),
                        "latency_ms": round(total_ms, 2),
                    }
                    await websocket.send_text(json.dumps(response))
                    continue

                # WORD MODE: RECORDING STATE (collecting exactly 26 frames)
                if frame_result.num_hands > 0:
                    feat_126 = normalize_frame(frame_result)
                    valid = 1.0
                else:
                    feat_126 = np.zeros(126, dtype=np.float32)
                    valid = 0.0

                state.recorded_features.append(feat_126)
                state.recorded_validity.append(valid)
                rec_len = len(state.recorded_features)

                if rec_len < TARGET_SEQ_LEN:
                    vf_so_far = int(sum(state.recorded_validity))
                    response = {
                        "mode": "word",
                        "word_state": "RECORDING",
                        "frame_index": rec_len,
                        "total_frames": TARGET_SEQ_LEN,
                        "valid_frames": vf_so_far,
                        "buffer_progress": f"{rec_len}/{TARGET_SEQ_LEN}",
                        "buffer_full": False,
                        "hand_present": bool(valid),
                        "prediction": "-",
                        "confidence": 0.0,
                        "raw_prediction": "-",
                        "raw_confidence": 0.0,
                        "smoothed_prediction": "-",
                        "smoothed_confidence": 0.0,
                    }
                    await websocket.send_text(json.dumps(response))
                    continue
                else:
                    # Exactly 26 frames collected! Send ANALYZING first, then run ONE inference
                    trial_feats = np.array(state.recorded_features, dtype=np.float32)
                    trial_val = np.array(state.recorded_validity, dtype=np.float32)
                    vf_count = int(trial_val.sum())

                    state.word_state = "ANALYZING"
                    await websocket.send_text(json.dumps({
                        "mode": "word",
                        "word_state": "ANALYZING",
                        "frame_index": TARGET_SEQ_LEN,
                        "total_frames": TARGET_SEQ_LEN,
                        "valid_frames": vf_count,
                        "buffer_progress": f"{TARGET_SEQ_LEN}/{TARGET_SEQ_LEN}",
                        "buffer_full": True,
                        "hand_present": vf_count >= 5,
                        "prediction": "-",
                        "confidence": 0.0,
                        "raw_prediction": "-",
                        "raw_confidence": 0.0,
                        "smoothed_prediction": "-",
                        "smoothed_confidence": 0.0,
                    }))

                    if vf_count < 5:
                        pred_class = "ƏL AŞKARLANMADI"
                        confidence = 0.0
                        print(f"[WORD TRIAL COMPLETE] Result: ƏL AŞKARLANMADI (valid_frames: {vf_count}/26)", flush=True)
                    else:
                        norm_feats = normalizer(trial_feats, trial_val)
                        inp = torch.from_numpy(norm_feats).unsqueeze(0).to(device)
                        with torch.no_grad():
                            logits = model(inp)
                            probs = torch.softmax(logits, dim=1)[0]
                        top_prob, top_idx = torch.topk(probs, k=1)
                        pred_class = idx_to_class[int(top_idx[0].item())]
                        confidence = float(top_prob[0].item())
                        print(f"[WORD TRIAL COMPLETE] Prediction: {pred_class} ({confidence*100:.2f}%) | valid_frames: {vf_count}/26", flush=True)

                    state.word_state = "RESULT"
                    state.last_word_result = {
                        "prediction": pred_class,
                        "confidence": float(confidence),
                        "valid_frames": vf_count,
                    }
                    response = {
                        "mode": "word",
                        "word_state": "RESULT",
                        "frame_index": TARGET_SEQ_LEN,
                        "total_frames": TARGET_SEQ_LEN,
                        "valid_frames": vf_count,
                        "buffer_progress": f"{TARGET_SEQ_LEN}/{TARGET_SEQ_LEN}",
                        "buffer_full": True,
                        "hand_present": pred_class != "ƏL AŞKARLANMADI",
                        "prediction": pred_class,
                        "confidence": float(confidence),
                        "raw_prediction": pred_class,
                        "raw_confidence": float(confidence),
                        "smoothed_prediction": pred_class,
                        "smoothed_confidence": float(confidence),
                        "segment_event": {
                            "label": pred_class,
                            "confidence": float(confidence),
                        } if pred_class != "ƏL AŞKARLANMADI" else None,
                    }
                    await websocket.send_text(json.dumps(response))
                    continue

            else:
                await websocket.send_text(json.dumps({"error": "Unknown message type"}))

    except WebSocketDisconnect:
        print("Client disconnected", flush=True)
    except Exception as e:
        print(f"WebSocket error: {e}", flush=True)
        try:
            await websocket.send_text(json.dumps({"error": str(e)}))
        except Exception:
            pass
    finally:
        if state.countdown_task is not None and not state.countdown_task.done():
            state.countdown_task.cancel()
