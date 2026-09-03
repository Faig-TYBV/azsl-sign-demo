"""
WebSocket backend for iPhone camera real-time sign language detection.
Reuses existing preprocessing, normalization, model loading, and class mapping.
"""

import asyncio
import json
import base64
import sys
from collections import deque
from pathlib import Path
from typing import Deque, Dict

import cv2
import mediapipe as mp
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Make project-root imports work
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
from src.inference.predict import (
    get_device,
    load_model,
    EXPECTED_CONFIG,
    CheckpointVerificationError,
)

app = FastAPI(title="AzSLD Web Demo Backend")

# Load the model once at startup
print("Loading model...", flush=True)
device = get_device()
checkpoint_path = PROJECT_ROOT / "outputs/checkpoints/gru_temporal_pool_best.pt"
try:
    model, class_to_idx, idx_to_class, _ = load_model(checkpoint_path)
    model.to(device)
    model.eval()
    print(f"Model loaded successfully on {device}", flush=True)
except Exception as e:
    print(f"Failed to load model: {e}", flush=True)
    raise

# Constants for prediction smoothing
WINDOW_SIZE = 5
CONFIDENCE_THRESHOLD = 0.50

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
PRESENCE_ON_FRAMES = 3      # consecutive hand-present frames to enter SIGNING
PRESENCE_OFF_FRAMES = 10    # consecutive end-condition frames to exit SIGNING
MOTION_DELTA_MIN = 0.02     # mean L2 feature delta required to start signing
MOTION_WINDOW = 5           # number of frames to average the motion delta over
MOTION_DELTA_FLOOR = 0.001  # below this motion we treat the hand as still
IDLE_CONFIDENCE_MAX = 0.10  # minimum smoothed confidence to consider a segment valid
STABLE_PRED_FRAMES = 3      # consecutive agreeing smoothed predictions required
MIN_SEGMENT_FRAMES = 26     # minimum frames between consecutive emits
COOLDOWN_FRAMES = 13        # minimum cooldown frames after an emit

# === Hand-presence gate (post-prediction, no retraining required) ============
# The GRU was trained without an explicit "idle/background" class, so softmax
# ALWAYS returns a top-1 even on all-zero feature buffers (confidence ~1/N).
# We compensate by NOT running inference when the buffer has no valid frames,
# and by clamping the displayed fields when the most recent frame is not a
# real hand detection. These thresholds are tuning knobs, not architecture.
SEGMENT_CONFIDENCE_FLOOR = 0.50   # strict: smoothed conf must be >= this to emit
DISPLAY_CONFIDENCE_FLOOR = 0.50   # below this we display "-" instead of the label
MIN_VALID_FRAMES_IN_BUFFER = 1    # skip inference if buffer has 0 valid frames

# Serve the frontend as static files
FRONTEND_DIR = PROJECT_ROOT / "src" / "web_demo" / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


class ConnectionState:
    """State per WebSocket connection."""

    def __init__(self):
        self.feature_buffer = deque(maxlen=TARGET_SEQ_LEN)
        self.validity_buffer = deque(maxlen=TARGET_SEQ_LEN)
        self.pred_history = deque(maxlen=WINDOW_SIZE)
        self.smoothed_prediction = None
        self.smoothed_confidence = 0.0
        self.landmarker = None
        self.last_timestamp_ms = 0
        self.frame_count = 0

        # ---- Temporal segmentation state (additive) ----
        self.seg_state = "IDLE"            # "IDLE" | "SIGNING"
        self.presence_on_count = 0         # consecutive frames with a hand
        self.presence_off_count = 0        # consecutive frames without a hand
        self.motion_window = deque(maxlen=MOTION_WINDOW)
        self.last_segment_label = None     # last label we emitted as a segment
        self.frames_since_last_emit = 10**9
        self.stable_pred_count = 0         # consecutive windows agreeing on smoothed_prediction
        self.prev_smoothed_prediction = None


@app.on_event("startup")
async def startup_event():
    print("WebSocket backend started.", flush=True)


@app.get("/")
async def root():
    index_path = FRONTEND_DIR / "index.html"
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected", flush=True)

    try:
        landmarker = create_hand_landmarker()
    except FileNotFoundError as e:
        await websocket.send_text(json.dumps({"error": str(e)}))
        await websocket.close()
        return

    state = ConnectionState()
    state.landmarker = landmarker

    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "Invalid JSON"}))
                continue

            msg_type = message.get("type")

            if msg_type == "frame":
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
                _dbg("JPEG decoded")

                if frame is None:
                    await websocket.send_text(json.dumps({"error": "Failed to decode image"}))
                    continue

                state.frame_count += 1
                _dbg(f"FRAME RECEIVED: {state.frame_count}")

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                current_timestamp_ms = int(asyncio.get_event_loop().time() * 1000)
                timestamp_ms = max(current_timestamp_ms, state.last_timestamp_ms + 1)
                state.last_timestamp_ms = timestamp_ms

                result = state.landmarker.detect_for_video(mp_image, timestamp_ms)
                _dbg("MediaPipe processed")

                frame_result = _mp_result_to_frame_result(result)
                if frame_result.num_hands > 0:
                    features = normalize_frame(frame_result)
                    valid = True
                else:
                    features = np.zeros(126, dtype=np.float32)
                    valid = False

                state.feature_buffer.append(features)
                state.validity_buffer.append(valid)

                buffer_len = len(state.feature_buffer)
                buffer_full = buffer_len >= TARGET_SEQ_LEN

                # Hand-presence gate (latest frame).
                # If MediaPipe did NOT detect a hand in this frame, do not let
                # the model be the source of truth for the UI — clamp the
                # displayed fields to "-" so the user sees idle state, not a
                # bogus softmax argmax over a zero feature vector.
                current_frame_valid = bool(state.validity_buffer[-1]) if state.validity_buffer else False

                response = {
                    "buffer_progress": f"{buffer_len}/{TARGET_SEQ_LEN}",
                    "buffer_full": buffer_full,
                    "hand_present": current_frame_valid,
                    "raw_prediction": "-",
                    "raw_confidence": 0.0,
                    "smoothed_prediction": state.smoothed_prediction or "-",
                    "smoothed_confidence": float(state.smoothed_confidence),
                }

                if buffer_full:
                    # Skip inference entirely if the buffer has zero valid
                    # hand frames (e.g. user just opened the page, no hand
                    # shown yet, or background-only). Without this gate,
                    # softmax over an all-zero 126-dim buffer would still
                    # pick a top-1 class and the frontend would display a
                    # false-positive prediction.
                    n_valid_in_buffer = int(sum(state.validity_buffer))
                    if n_valid_in_buffer < MIN_VALID_FRAMES_IN_BUFFER:
                        print(f"INFERENCE SKIPPED: zero valid frames in buffer (valid={n_valid_in_buffer}/{buffer_len})", flush=True)
                        # Keep state.smoothed_prediction / smoothed_confidence
                        # as they were (do NOT reset — the user may still want
                        # to see the last word they signed before backing off).
                    else:
                        _dbg("INFERENCE")
                        frame_features = np.stack(list(state.feature_buffer), axis=0)
                        frame_valid = np.array(list(state.validity_buffer), dtype=bool)
                        seq_features, mask = preprocess_sequence(frame_features, frame_valid)

                        input_tensor = torch.from_numpy(seq_features).unsqueeze(0).to(device)
                        mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(device)

                        with torch.no_grad():
                            logits = model(input_tensor)
                            probs = torch.softmax(logits, dim=1)[0]

                        confidence_t, predicted_idx_t = torch.max(probs, dim=0)
                        confidence = float(confidence_t.item())
                        predicted_idx = int(predicted_idx_t.item())
                        pred_class = idx_to_class[predicted_idx]

                        _dbg(f"Prediction: {pred_class} Confidence: {confidence:.2f}")

                        if confidence >= CONFIDENCE_THRESHOLD:
                            state.pred_history.append((predicted_idx, confidence))

                        if state.pred_history:
                            votes = {}
                            for idx, conf in state.pred_history:
                                votes[idx] = votes.get(idx, 0.0) + conf
                            smoothed_idx = max(votes, key=votes.get)
                            smoothed_confidence = votes[smoothed_idx] / len(state.pred_history)
                            state.smoothed_prediction = idx_to_class[smoothed_idx]
                            state.smoothed_confidence = smoothed_confidence
                        else:
                            state.smoothed_prediction = None
                            state.smoothed_confidence = 0.0

                        response.update({
                            "raw_prediction": pred_class,
                            "raw_confidence": float(confidence),
                            "smoothed_prediction": state.smoothed_prediction or "-",
                            "smoothed_confidence": float(state.smoothed_confidence),
                        })

                    # Confidence-floor clamp: if the *displayed* smoothed
                    # confidence is below the floor, show "-" instead. This
                    # prevents low-confidence noise from rendering as a real
                    # prediction in the UI even when the segmenter already
                    # decided not to emit (the segmenter uses a stricter
                    # SEGMENT_CONFIDENCE_FLOOR).
                    if float(response["smoothed_confidence"]) < DISPLAY_CONFIDENCE_FLOOR:
                        response["smoothed_prediction"] = "-"
                        response["smoothed_confidence"] = 0.0

                # If the latest frame had no hand, ALWAYS force idle display,
                # regardless of what the buffer-based inference above said.
                # (Inference is skipped when valid_count==0, but even a few
                # valid frames in the past do not justify showing a label
                # when the *current* moment is "no hand".)
                if not current_frame_valid:
                    response["raw_prediction"] = "-"
                    response["raw_confidence"] = 0.0
                    response["smoothed_prediction"] = "-"
                    response["smoothed_confidence"] = 0.0

                # --- Temporal segmentation (additive layer) -----------------
                # 1) Hand-presence counters
                if state.validity_buffer[-1]:
                    state.presence_on_count += 1
                    state.presence_off_count = 0
                else:
                    state.presence_off_count += 1
                    state.presence_on_count = 0

                # 2) Frame-to-frame L2 motion, maintained in a rolling window
                if len(state.feature_buffer) >= 2:
                    motion_delta = float(np.linalg.norm(
                        state.feature_buffer[-1] - state.feature_buffer[-2]
                    ))
                else:
                    motion_delta = 0.0
                state.motion_window.append(motion_delta)
                motion_mean = float(np.mean(state.motion_window)) if state.motion_window else 0.0

                # 3) IDLE -> SIGNING (only after debounced presence + motion)
                if (
                    state.seg_state == "IDLE"
                    and state.presence_on_count >= PRESENCE_ON_FRAMES
                    and motion_mean >= MOTION_DELTA_MIN
                ):
                    state.seg_state = "SIGNING"
                    state.stable_pred_count = 0
                    state.prev_smoothed_prediction = None
                    print(
                        f"SEGMENT STATE: IDLE -> SIGNING  (presence={state.presence_on_count}, motion={motion_mean:.4f})",
                        flush=True,
                    )

                # 4) SIGNING -> IDLE (debounced end condition)
                if (
                    state.seg_state == "SIGNING"
                    and (
                        state.presence_off_count >= PRESENCE_OFF_FRAMES
                        or motion_mean < MOTION_DELTA_FLOOR
                    )
                ):
                    state.seg_state = "IDLE"
                    state.stable_pred_count = 0
                    state.prev_smoothed_prediction = None
                    print(
                        f"SEGMENT STATE: SIGNING -> IDLE  (presence_off={state.presence_off_count}, motion={motion_mean:.4f})",
                        flush=True,
                    )

                # 5) Track consecutive identical smoothed predictions (only while buffer full)
                if buffer_full and state.smoothed_prediction not in (None, "-"):
                    if state.smoothed_prediction == state.prev_smoothed_prediction:
                        state.stable_pred_count += 1
                    else:
                        state.stable_pred_count = 1
                        state.prev_smoothed_prediction = state.smoothed_prediction
                else:
                    # Without a valid smoothed prediction we don't increment stability.
                    state.prev_smoothed_prediction = state.smoothed_prediction

                # 6) Emit decision
                state.frames_since_last_emit += 1
                segment_event = None
                if (
                    state.seg_state == "IDLE"
                    and state.smoothed_prediction not in (None, "-")
                    and float(state.smoothed_confidence) >= SEGMENT_CONFIDENCE_FLOOR
                    and state.stable_pred_count >= STABLE_PRED_FRAMES
                    and state.frames_since_last_emit >= MIN_SEGMENT_FRAMES
                    and state.frames_since_last_emit >= COOLDOWN_FRAMES
                    and state.smoothed_prediction != state.last_segment_label
                    and current_frame_valid   # do NOT emit on a hand-absent frame
                ):
                    state.last_segment_label = state.smoothed_prediction
                    state.frames_since_last_emit = 0
                    segment_event = {
                        "label": state.smoothed_prediction,
                        "confidence": float(state.smoothed_confidence),
                    }
                    print(
                        f"SEGMENT EMIT: {segment_event['label']} conf={segment_event['confidence']:.2f}",
                        flush=True,
                    )

                # 7) Additive response fields (existing six fields unchanged)
                response["segmentation"] = {
                    "state": state.seg_state,
                    "hand_present": bool(state.validity_buffer[-1]),
                    "motion_mean": motion_mean,
                    "frames_since_last_emit": int(state.frames_since_last_emit),
                    "stable_count": int(state.stable_pred_count),
                    "last_emitted_label": state.last_segment_label,
                }
                if segment_event is not None:
                    response["segment_event"] = segment_event
                # --- end temporal segmentation ------------------------------

                await websocket.send_text(json.dumps(response))
                _dbg("Response sent")

            elif msg_type == "reset":
                state.feature_buffer.clear()
                state.validity_buffer.clear()
                state.pred_history.clear()
                state.smoothed_prediction = None
                state.smoothed_confidence = 0.0
                # Segmentation state reset (additive)
                state.seg_state = "IDLE"
                state.presence_on_count = 0
                state.presence_off_count = 0
                state.motion_window.clear()
                state.stable_pred_count = 0
                state.prev_smoothed_prediction = None
                state.frames_since_last_emit = 10**9
                # last_segment_label is intentionally preserved so the user
                # can still see the most recent emitted word.
                await websocket.send_text(json.dumps({"status": "reset"}))

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
