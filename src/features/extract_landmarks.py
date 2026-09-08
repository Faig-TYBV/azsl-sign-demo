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

# Absolute, resolved from this file — NOT relative to the working directory.
# The bundle lives at <repo>/src/models/hand_landmarker.task; a CWD-relative
# "models/hand_landmarker.task" only resolved when the process happened to be
# launched from a directory containing one, which broke predict.py, the
# extraction scripts and two tests, and forced backend.py to pass its own
# absolute path as a workaround.
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "hand_landmarker.task"


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


def normalize_frame(frame: FrameResult, version: str = "v1") -> np.ndarray:
    """
    Normalize one frame into a flat feature vector.

    Args:
        frame: FrameResult object containing raw landmark detections.
        version: Preprocessing version ("v1" for 126D, "v2" for Option A 280D).

    Returns:
        126D array (v1) or 280D array (v2).
    """
    feats_v1 = np.zeros(MAX_HANDS * FEATURES_PER_HAND, dtype=np.float32)
    for slot in range(min(frame.num_hands, MAX_HANDS)):
        norm = normalize_hand(frame.landmarks[slot])
        feats_v1[slot * FEATURES_PER_HAND:(slot + 1) * FEATURES_PER_HAND] = norm.ravel()

    if version == "v1":
        return feats_v1
    elif version == "v2":
        landmarks_3d = frame.landmarks[np.newaxis, ...]  # (1, 2, 21, 3)
        validity_mask = np.array([frame.num_hands > 0], dtype=bool)
        raw_wrists = np.array([[[lm[0, 0], lm[0, 1], lm[0, 2]] for lm in frame.landmarks]], dtype=np.float32)
        v2_feats = compute_v2_features(landmarks_3d, validity_mask=validity_mask, raw_wrists_seq=raw_wrists)
        return v2_feats[0]
    else:
        raise ValueError(f"Unknown preprocessing version: {version!r}. Expected 'v1' or 'v2'.")


def compute_v2_features(
    landmarks_seq: np.ndarray,
    validity_mask: np.ndarray | None = None,
    raw_wrists_seq: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute Option A Full Dynamic 280D feature representation for a sequence of frames.

    Features layout (280D total):
      - 126D: Normalized Position Coordinates (v1 pos)
      - 126D: Landmark Velocity / Position Deltas (pos_t - pos_{t-1})
      -   6D: Global Wrist Trajectory / Velocity (wrist_t - wrist_{t-1} per hand)
      -  12D: Fingertip & Palm Orientation Vectors (Index Ray 3D + Palm Normal 3D per hand)
      -  10D: Finger Extension Ratios (Tip-to-wrist / MCP-to-wrist ratio per finger, 5 per hand)

    Args:
        landmarks_seq: Input landmarks sequence. Either shape (T, 126) normalized v1 features,
                       or (T, MAX_HANDS, 21, 3) landmark array.
        validity_mask: Optional boolean or binary float mask of shape (T,) indicating frame validity.
        raw_wrists_seq: Optional array of shape (T, MAX_HANDS, 3) containing raw wrist coordinates.

    Returns:
        Array of shape (T, 280) with float32 dtype and zero NaNs.
    """
    landmarks_arr = np.asarray(landmarks_seq, dtype=np.float32)
    if landmarks_arr.ndim == 2 and landmarks_arr.shape[1] == 126:
        T = landmarks_arr.shape[0]
        pos_feats = landmarks_arr.copy()
        landmarks_3d = pos_feats.reshape(T, MAX_HANDS, NUM_LANDMARKS, NUM_COORDS)
    elif landmarks_arr.ndim == 4 and landmarks_arr.shape[1:] == (MAX_HANDS, NUM_LANDMARKS, NUM_COORDS):
        T = landmarks_arr.shape[0]
        pos_feats = np.zeros((T, MAX_HANDS * FEATURES_PER_HAND), dtype=np.float32)
        for t in range(T):
            for slot in range(MAX_HANDS):
                norm = normalize_hand(landmarks_arr[t, slot])
                pos_feats[t, slot * FEATURES_PER_HAND:(slot + 1) * FEATURES_PER_HAND] = norm.ravel()
        landmarks_3d = pos_feats.reshape(T, MAX_HANDS, NUM_LANDMARKS, NUM_COORDS)
    else:
        raise ValueError(
            f"Expected landmarks_seq shape (T, 126) or (T, {MAX_HANDS}, {NUM_LANDMARKS}, {NUM_COORDS}), "
            f"got {landmarks_arr.shape}"
        )

    # Validity mask (T,)
    if validity_mask is not None:
        valid_frames = np.asarray(validity_mask, dtype=bool)
    else:
        valid_frames = np.linalg.norm(pos_feats, axis=1) > 1e-6

    # 1. Position features (126D)
    pos_feats = np.nan_to_num(pos_feats, nan=0.0)

    # 2. Velocity features (126D)
    vel_feats = np.zeros_like(pos_feats, dtype=np.float32)
    for t in range(1, T):
        if valid_frames[t] and valid_frames[t - 1]:
            vel_feats[t] = pos_feats[t] - pos_feats[t - 1]
    vel_feats = np.nan_to_num(vel_feats, nan=0.0)

    # 3. Global Wrist Velocity (6D)
    wrist_vel_feats = np.zeros((T, MAX_HANDS * NUM_COORDS), dtype=np.float32)
    if raw_wrists_seq is not None:
        wrists_arr = np.asarray(raw_wrists_seq, dtype=np.float32)
        if wrists_arr.shape == (T, MAX_HANDS, NUM_COORDS):
            for t in range(1, T):
                if valid_frames[t] and valid_frames[t - 1]:
                    diff = wrists_arr[t] - wrists_arr[t - 1]
                    wrist_vel_feats[t] = diff.ravel()
    wrist_vel_feats = np.nan_to_num(wrist_vel_feats, nan=0.0)

    # 4. Orientation Vectors (12D: 6D per hand)
    orientation_feats = np.zeros((T, MAX_HANDS * 6), dtype=np.float32)
    for t in range(T):
        if not valid_frames[t]:
            continue
        for slot in range(MAX_HANDS):
            hand_lm = landmarks_3d[t, slot]  # (21, 3)
            if np.linalg.norm(hand_lm) < 1e-6:
                continue

            w = hand_lm[0]      # Wrist
            idx_mcp = hand_lm[5]
            idx_tip = hand_lm[8]
            pinky_mcp = hand_lm[17]

            # Index ray
            ray = idx_tip - idx_mcp
            ray_norm = float(np.linalg.norm(ray))
            ray_unit = (ray / ray_norm) if ray_norm > 1e-6 else np.zeros(3, dtype=np.float32)

            # Palm normal
            u = idx_mcp - w
            v = pinky_mcp - w
            normal = np.cross(u, v)
            norm_val = float(np.linalg.norm(normal))
            normal_unit = (normal / norm_val) if norm_val > 1e-6 else np.zeros(3, dtype=np.float32)

            slot_orient = np.concatenate([ray_unit, normal_unit], axis=0)
            orientation_feats[t, slot * 6:(slot + 1) * 6] = slot_orient
    orientation_feats = np.nan_to_num(orientation_feats, nan=0.0)

    # 5. Finger Extension Ratios (10D: 5D per hand)
    ratio_feats = np.zeros((T, MAX_HANDS * 5), dtype=np.float32)
    tip_indices = [4, 8, 12, 16, 20]
    mcp_indices = [2, 5, 9, 13, 17]

    for t in range(T):
        if not valid_frames[t]:
            continue
        for slot in range(MAX_HANDS):
            hand_lm = landmarks_3d[t, slot]  # (21, 3)
            if np.linalg.norm(hand_lm) < 1e-6:
                continue

            w = hand_lm[0]  # Wrist
            ratios = np.zeros(5, dtype=np.float32)
            for f_idx in range(5):
                tip_dist = float(np.linalg.norm(hand_lm[tip_indices[f_idx]] - w))
                mcp_dist = float(np.linalg.norm(hand_lm[mcp_indices[f_idx]] - w))
                ratios[f_idx] = tip_dist / (mcp_dist + 1e-6)

            ratio_feats[t, slot * 5:(slot + 1) * 5] = ratios
    ratio_feats = np.nan_to_num(ratio_feats, nan=0.0)

    # Concatenate all 5 components: 126 + 126 + 6 + 12 + 10 = 280D
    v2_feats = np.concatenate(
        [pos_feats, vel_feats, wrist_vel_feats, orientation_feats, ratio_feats],
        axis=1,
    )
    return v2_feats.astype(np.float32)


def normalize_sequence(
    raw_frames: list[FrameResult],
    version: str = "v1",
) -> np.ndarray:
    """
    Normalize a list of FrameResult objects into a sequence array.

    Args:
        raw_frames: List of FrameResult objects.
        version: "v1" for (T, 126) or "v2" for (T, 280).

    Returns:
        Array of shape (T, 126) or (T, 280).
    """
    if version == "v1":
        return np.stack([normalize_frame(fr, version="v1") for fr in raw_frames], axis=0)
    elif version == "v2":
        T = len(raw_frames)
        if T == 0:
            return np.zeros((0, 280), dtype=np.float32)

        landmarks_3d = np.zeros((T, MAX_HANDS, NUM_LANDMARKS, NUM_COORDS), dtype=np.float32)
        validity_mask = np.zeros(T, dtype=bool)
        raw_wrists = np.zeros((T, MAX_HANDS, NUM_COORDS), dtype=np.float32)

        for t, fr in enumerate(raw_frames):
            validity_mask[t] = fr.num_hands > 0
            for slot in range(min(fr.num_hands, MAX_HANDS)):
                landmarks_3d[t, slot] = fr.landmarks[slot]
                raw_wrists[t, slot] = fr.landmarks[slot][0]

        return compute_v2_features(landmarks_3d, validity_mask=validity_mask, raw_wrists_seq=raw_wrists)
    else:
        raise ValueError(f"Unknown preprocessing version: {version!r}. Expected 'v1' or 'v2'.")
