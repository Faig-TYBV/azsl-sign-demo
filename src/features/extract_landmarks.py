"""
MediaPipe Hand Landmark extraction for AzSLD_Words_200 videos.

Uses the MediaPipe Tasks API (HandLandmarker) with the downloaded
hand_landmarker.task model (models/hand_landmarker.task).

Per frame:
  - Detects up to 2 hands (explicitly reports hand count).
  - Extracts 21 landmarks x (x, y, z) = 63 values per detected hand.
  - Frames with no detection are NOT discarded: a per-frame validity flag
    and zero-filled landmarks are stored instead.

Normalization:
  - Wrist-relative coordinates: each landmark is shifted so the wrist
    (landmark 0) is at the origin -> removes image position sensitivity.
  - Scale normalization: coordinates are divided by the distance between
    wrist and middle-finger MCP (landmark 9) -> removes hand size /
    camera-distance sensitivity.
  - For two-hand videos, each hand is normalized independently; a fixed
    left/right slot ordering keeps feature layout consistent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np

log = logging.getLogger("extract_landmarks")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_LANDMARKS = 21
NUM_COORDS = 3
FEATURES_PER_HAND = NUM_LANDMARKS * NUM_COORDS  # 63
MAX_HANDS = 2

# Preprocessing version — bump when normalization / layout changes.
PREPROCESSING_VERSION = "v1.0-wrist-scale-normalized"

DEFAULT_MODEL_PATH = Path("models/hand_landmarker.task")


@dataclass
class FrameResult:
    """Raw (unnormalized) result for one video frame."""
    num_hands: int                      # 0, 1 or 2
    handedness: list = field(default_factory=list)   # e.g. ["Left", "Right"]
    landmarks: np.ndarray = field(      # shape (MAX_HANDS, 21, 3), zeros if missing
        default_factory=lambda: np.zeros((MAX_HANDS, NUM_LANDMARKS, NUM_COORDS), dtype=np.float32)
    )


@dataclass
class VideoExtractionResult:
    """Full extraction result for one video."""
    video_path: str
    word_label: str
    original_frame_count: int
    original_fps: float
    frames_with_hands: int
    frames_no_hand: int
    max_hands_seen: int                 # 0, 1 or 2 -> one-hand vs two-hand sign info
    raw_frames: list = field(default_factory=list)   # list[FrameResult]
    error: str | None = None


# ---------------------------------------------------------------------------
# Landmarker setup
# ---------------------------------------------------------------------------
def create_hand_landmarker(
    model_path: Path = DEFAULT_MODEL_PATH,
    num_hands: int = MAX_HANDS,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> mp_vision.HandLandmarker:
    """Create a MediaPipe HandLandmarker from the .task model bundle."""
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Hand landmarker model not found at '{model_path}'. "
            "Download it first (see README / run_pilot_extraction.py)."
        )
    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=num_hands,
        min_hand_detection_confidence=min_detection_confidence,
        min_hand_presence_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


# ---------------------------------------------------------------------------
# Per-frame extraction
# ---------------------------------------------------------------------------
def _mp_result_to_frame_result(result) -> FrameResult:
    """Convert a MediaPipe HandLandmarkerResult into a FrameResult."""
    n = len(result.hand_landmarks)
    fr = FrameResult(num_hands=n)

    if n == 0:
        return fr  # all-zero landmarks, validity handled by caller flags

    # Sort hands deterministically by handedness label ("Left" < "Right")
    labels = []
    for i in range(n):
        try:
            labels.append(result.handedness[i][0].category_name)
        except (IndexError, AttributeError):
            labels.append(f"hand{i}")
    order = sorted(range(n), key=lambda i: labels[i])

    for slot, idx in enumerate(order):
        lm = result.hand_landmarks[idx]
        arr = np.array(
            [[p.x, p.y, p.z] for p in lm], dtype=np.float32
        )
        fr.landmarks[slot] = arr
        fr.handedness.append(labels[idx])

    return fr


def extract_video_landmarks(
    video_path: Path | str,
    landmarker: mp_vision.HandLandmarker,
    timestamp_offset_ms: int = 0,
) -> VideoExtractionResult:
    """
    Run hand landmark detection over every frame of a video.

    Frames where no hand is detected are kept with num_hands=0 and
    zero-filled landmarks (never silently discarded).
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return VideoExtractionResult(
            video_path=str(video_path),
            word_label=video_path.parent.name,
            original_frame_count=0,
            original_fps=0.0,
            frames_with_hands=0,
            frames_no_hand=0,
            max_hands_seen=0,
            error="cv2.VideoCapture failed to open",
        )

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    res = VideoExtractionResult(
        video_path=str(video_path),
        word_label=video_path.parent.name,
        original_frame_count=0,
        original_fps=float(fps),
        frames_with_hands=0,
        frames_no_hand=0,
        max_hands_seen=0,
    )

    # MediaPipe requires monotonically increasing timestamps for the lifetime
    # of a landmarker instance; when reusing one landmarker across videos we
    # continue from the caller-provided offset.
    timestamp_ms = timestamp_offset_ms
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            res.original_frame_count += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            # Guarantee strictly increasing timestamps even for odd FPS values
            timestamp_ms += max(1, int(round(1000.0 / fps)))

            fr = _mp_result_to_frame_result(result)
            if fr.num_hands > 0:
                res.frames_with_hands += 1
                res.max_hands_seen = max(res.max_hands_seen, fr.num_hands)
            else:
                res.frames_no_hand += 1
            res.raw_frames.append(fr)

    except Exception as exc:
        res.error = f"{type(exc).__name__}: {exc}"
        log.error("Error extracting %s: %s", video_path.name, res.error)
    finally:
        cap.release()

    return res


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
def normalize_hand(landmarks: np.ndarray) -> np.ndarray:
    """
    Normalize a single hand's landmarks (21, 3):

    1. Wrist-relative translation: subtract landmark 0 (wrist).
       -> invariant to position of the hand in the image.
    2. Scale normalization: divide by ||wrist - middle_finger_MCP||,
       i.e. the Euclidean distance between landmarks 0 and 9.
       -> invariant to hand size / camera distance.

    If scale is ~zero (degenerate detection), returns zeros to avoid NaNs.
    """
    out = landmarks.copy()
    wrist = out[0]
    out -= wrist
    ref = landmarks[9] - landmarks[0]
    scale = float(np.linalg.norm(ref))
    if scale < 1e-6:
        return np.zeros_like(out, dtype=np.float32)
    out /= scale
    return out.astype(np.float32)


def normalize_frame(frame: FrameResult) -> np.ndarray:
    """
    Normalize one frame into a flat feature vector of length
    MAX_HANDS * FEATURES_PER_HAND (126).

    Layout: [left-slot hand (63)] + [right-slot hand (63)].
    Missing hands stay zero-filled.
    """
    feats = np.zeros(MAX_HANDS * FEATURES_PER_HAND, dtype=np.float32)
    for slot in range(min(frame.num_hands, MAX_HANDS)):
        norm = normalize_hand(frame.landmarks[slot])
        feats[slot * FEATURES_PER_HAND:(slot + 1) * FEATURES_PER_HAND] = norm.ravel()
    return feats