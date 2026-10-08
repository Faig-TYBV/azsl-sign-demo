"""
Alphabet (fingerspelling) classifier for AzSL single-frame inference.

Python implementation of the 84-dimensional feature extraction and hierarchical
MLP classification defined in src/web_demo/frontend/js/azsl_alphabet.js and
trained via src/web_demo/frontend/models/azsl_hierarchical_model.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ALPHABET_MODEL_PATH = (
    PROJECT_ROOT / "src" / "web_demo" / "frontend" / "models" / "azsl_hierarchical_model.json"
)

# Canonical 21 landmark indices
LM_WRIST = 0
LM_THUMB_MCP = 2
LM_THUMB_IP = 3
LM_THUMB_TIP = 4
LM_INDEX_MCP = 5
LM_INDEX_PIP = 6
LM_INDEX_TIP = 8
LM_MIDDLE_MCP = 9
LM_MIDDLE_PIP = 10
LM_MIDDLE_TIP = 12
LM_RING_MCP = 13
LM_RING_PIP = 14
LM_RING_TIP = 16
LM_PINKY_MCP = 17
LM_PINKY_PIP = 18
LM_PINKY_TIP = 20

FINGERS = [
    {"base": LM_WRIST, "j1": LM_THUMB_MCP, "j2": LM_THUMB_IP, "tip": LM_THUMB_TIP},
    {"base": LM_WRIST, "j1": LM_INDEX_MCP, "j2": LM_INDEX_PIP, "tip": LM_INDEX_TIP},
    {"base": LM_WRIST, "j1": LM_MIDDLE_MCP, "j2": LM_MIDDLE_PIP, "tip": LM_MIDDLE_TIP},
    {"base": LM_WRIST, "j1": LM_RING_MCP, "j2": LM_RING_PIP, "tip": LM_RING_TIP},
    {"base": LM_WRIST, "j1": LM_PINKY_MCP, "j2": LM_PINKY_PIP, "tip": LM_PINKY_TIP},
]

TIP_PAIRS = [
    (LM_THUMB_TIP, LM_INDEX_TIP),
    (LM_INDEX_TIP, LM_MIDDLE_TIP),
    (LM_MIDDLE_TIP, LM_RING_TIP),
    (LM_RING_TIP, LM_PINKY_TIP),
]

MIN_CONFIDENCE = 0.55
CONTROL_OVERRIDE_CONFIDENCE = 0.75
HEURISTIC_CONF = 0.92
ANGLE_STRAIGHT = 155.0
ANGLE_FIST = 100.0


def _vec_mag(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def _vec_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    mags = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if mags < 1e-9:
        return 0.0
    cos = np.clip(np.dot(v1, v2) / mags, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def _joint_angle_deg(coords: np.ndarray, mcp_idx: int, pip_idx: int, tip_idx: int) -> float:
    to_mcp = coords[mcp_idx] - coords[pip_idx]
    to_tip = coords[tip_idx] - coords[pip_idx]
    return _angle_between(to_mcp, to_tip)


def normalize_landmarks(landmarks: np.ndarray, mirror_x: bool = False) -> np.ndarray:
    """
    Wrist-relative, scale-normalized landmarks.
    landmarks: shape (21, 3)
    """
    in_dtype = landmarks.dtype if hasattr(landmarks, "dtype") and landmarks.dtype in (np.float32, np.float64) else np.float32
    landmarks_calc = np.asarray(landmarks, dtype=np.float64)
    wrist = landmarks_calc[LM_WRIST]
    shifted = landmarks_calc - wrist
    scale = float(np.linalg.norm(shifted[LM_MIDDLE_MCP]))
    if scale < 1e-6:
        scale = 1e-6
    normalized = shifted / scale
    if mirror_x:
        normalized = normalized.copy()
        normalized[:, 0] = -normalized[:, 0]
    return normalized.astype(in_dtype)


def build_feature_vector_84(
    coords: np.ndarray, velocity: Tuple[float, float] = (0.0, 0.0)
) -> np.ndarray:
    """
    Build the exact 84-dimensional feature vector:
      [0..62] 21 normalized landmarks x (x,y,z)
      [63..77] 15 finger joint angles
      [78..81] 4 fingertip-gap distances
      [82..83] 2 wrist velocity components
    """
    out_dtype = coords.dtype if hasattr(coords, "dtype") and coords.dtype in (np.float32, np.float64) else np.float32
    coords_calc = np.asarray(coords, dtype=np.float64)
    features = np.zeros(84, dtype=np.float64)

    # [0..62] 21 landmarks x (x, y, z)
    features[:63] = coords_calc.reshape(-1)

    # [63..77] 15 finger joint angles (5 fingers x 3 angles)
    middle_dir = coords_calc[LM_MIDDLE_PIP] - coords_calc[LM_MIDDLE_MCP]
    for f, F in enumerate(FINGERS):
        base_p = coords_calc[F["base"]]
        j1 = coords_calc[F["j1"]]
        j2 = coords_calc[F["j2"]]
        tip = coords_calc[F["tip"]]

        base_flex = _angle_between(base_p - j1, j2 - j1)
        tip_flex = _angle_between(j1 - j2, tip - j2)

        this_dir = j2 - j1
        spread = _angle_between(this_dir, middle_dir)

        features[63 + f * 3] = base_flex
        features[63 + f * 3 + 1] = tip_flex
        features[63 + f * 3 + 2] = spread

    # [78..81] 4 fingertip distances
    for t, (p1, p2) in enumerate(TIP_PAIRS):
        features[78 + t] = _vec_dist(coords_calc[p1], coords_calc[p2])

    # [82..83] wrist velocity
    features[82] = velocity[0]
    features[83] = velocity[1]

    return features.astype(out_dtype)


# A feature whose training standard deviation is this small was effectively
# constant, so it carries no information — but dividing by it amplifies any
# difference in that feature by a factor of ~1e6. One joint angle in this model
# has std ~4.4e-07, and it turned a sub-microscopic float discrepancy between
# the Python and JavaScript implementations into a ~2 sigma swing in the model
# input, and a 0.23 swing in the resulting confidence.
#
# Guarding it removes a noise amplifier rather than any signal. `== 0` was too
# narrow: the value is near zero, not exactly zero.
SCALER_MIN_STD = 1e-6


def _apply_scaler(input_vec: np.ndarray, scaler: Dict[str, Any]) -> np.ndarray:
    mean = np.array(scaler["mean"], dtype=np.float32)
    std = np.array(scaler["std"], dtype=np.float32)
    std = np.where(std < SCALER_MIN_STD, 1.0, std)
    return (input_vec - mean) / std


def _mlp_forward(
    model_dict: Dict[str, Any], input_vec: np.ndarray
) -> List[Dict[str, Any]]:
    activation = input_vec
    layers = model_dict["layers"]
    num_layers = len(layers)

    for li, layer in enumerate(layers):
        weights = np.array(layer["weights"], dtype=np.float32)
        biases = np.array(layer["biases"], dtype=np.float32)
        out = activation @ weights + biases
        is_output = li == num_layers - 1
        if not is_output:
            out = np.maximum(0.0, out)
        activation = out

    output_activation = model_dict.get("outputActivation", "softmax")
    classes = model_dict["classes"]

    if output_activation == "sigmoid_binary":
        p1 = float(1.0 / (1.0 + np.exp(-activation[0])))
        candidates = [
            {"label": str(classes[0]), "confidence": 1.0 - p1},
            {"label": str(classes[1]), "confidence": p1},
        ]
    else:
        max_logit = np.max(activation)
        exps = np.exp(activation - max_logit)
        sum_exp = float(np.sum(exps)) or 1e-9
        probs = exps / sum_exp
        candidates = [
            {"label": str(classes[i]), "confidence": float(probs[i])}
            for i in range(len(classes))
        ]

    candidates.sort(key=lambda c: c["confidence"], reverse=True)
    return candidates


def _finger_state(
    coords: np.ndarray, mcp_idx: int, pip_idx: int, tip_idx: int
) -> str:
    angle = _joint_angle_deg(coords, mcp_idx, pip_idx, tip_idx)
    if angle >= ANGLE_STRAIGHT:
        return "extended"
    if angle <= ANGLE_FIST:
        return "curled"
    return "mid"


def _thumb_extended(coords: np.ndarray, margin: float = 1.15) -> bool:
    tip_to_pinky = _vec_dist(coords[LM_THUMB_TIP], coords[LM_PINKY_MCP])
    mcp_to_pinky = _vec_dist(coords[LM_THUMB_MCP], coords[LM_PINKY_MCP])
    return tip_to_pinky > mcp_to_pinky * margin


def _thumb_pointing_up(coords: np.ndarray, threshold: float = 0.15) -> bool:
    return bool(
        coords[LM_THUMB_TIP, 1] < coords[LM_THUMB_MCP, 1] - threshold
        and coords[LM_THUMB_TIP, 1] < -threshold
    )


def detect_control_gesture(coords: np.ndarray) -> Optional[Dict[str, Any]]:
    fs_index = _finger_state(coords, LM_INDEX_MCP, LM_INDEX_PIP, LM_INDEX_TIP)
    fs_middle = _finger_state(coords, LM_MIDDLE_MCP, LM_MIDDLE_PIP, LM_MIDDLE_TIP)
    fs_ring = _finger_state(coords, LM_RING_MCP, LM_RING_PIP, LM_RING_TIP)
    fs_pinky = _finger_state(coords, LM_PINKY_MCP, LM_PINKY_PIP, LM_PINKY_TIP)

    thumb_out = _thumb_extended(coords)
    thumb_up = _thumb_pointing_up(coords)

    if (
        fs_index == "curled"
        and fs_middle == "curled"
        and fs_ring == "curled"
        and fs_pinky == "curled"
        and not thumb_out
        and thumb_up
    ):
        return {"label": "SPACE", "confidence": HEURISTIC_CONF}

    if (
        fs_index == "extended"
        and fs_middle == "curled"
        and fs_ring == "curled"
        and fs_pinky == "curled"
        and not thumb_out
        and not thumb_up
    ):
        return {"label": "DEL", "confidence": HEURISTIC_CONF}

    return None


class AlphabetClassifier:
    def __init__(self, model_path: Path = DEFAULT_ALPHABET_MODEL_PATH):
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Alphabet model JSON not found: {self.model_path}")
        with open(self.model_path, "r", encoding="utf-8") as f:
            self.model_data = json.load(f)

        self.level1_model = self.model_data["level1"]["model"]
        self.level1_scaler = self.model_data["level1"]["scaler"]
        self.clusters = self.model_data["clusters"]

    def classify_hierarchical(
        self, coords: np.ndarray, velocity: Tuple[float, float] = (0.0, 0.0)
    ) -> Dict[str, Any]:
        full84 = build_feature_vector_84(coords, velocity)
        scaled_l1 = _apply_scaler(full84, self.level1_scaler)
        cluster_candidates = _mlp_forward(self.level1_model, scaled_l1)

        top_cluster = str(cluster_candidates[0]["label"])
        cluster_entry = self.clusters.get(top_cluster)
        if not cluster_entry:
            return {"label": None, "confidence": 0.0, "candidates": []}

        feature_indices = cluster_entry.get("featureIndices")
        if feature_indices and len(feature_indices) != len(full84):
            sub_input = full84[feature_indices]
        else:
            sub_input = full84

        scaled_sub = _apply_scaler(sub_input, cluster_entry["scaler"])
        letter_candidates = _mlp_forward(cluster_entry["model"], scaled_sub)

        return {
            "label": letter_candidates[0]["label"],
            "confidence": letter_candidates[0]["confidence"],
            "candidates": letter_candidates[:2],
        }

    def predict_frame(
        self,
        landmarks: np.ndarray,
        handedness: str = "Right",
        min_confidence: float = MIN_CONFIDENCE,
    ) -> Tuple[Optional[str], float]:
        if landmarks is None or landmarks.shape != (21, 3):
            return None, 0.0

        mirror_x = handedness == "Left"
        coords = normalize_landmarks(landmarks, mirror_x=mirror_x)

        # Control gestures (SPACE = thumbs-up, DEL = index-point) are checked
        # FIRST. The letter head uses a small MLP with no "not-a-letter" class,
        # so its softmax saturates (>0.75 on almost any hand pose); leaving the
        # old `confidence >= CONTROL_OVERRIDE_CONFIDENCE` short-circuit ahead of
        # this made SPACE/DEL practically unreachable.
        control = detect_control_gesture(coords)
        if control:
            return control["label"], float(control["confidence"])

        letter_result = self.classify_hierarchical(coords, (0.0, 0.0))
        if letter_result["confidence"] >= min_confidence:
            return letter_result["label"], float(letter_result["confidence"])

        return None, float(letter_result["confidence"])


# --- Stabilizer tuning ---------------------------------------------------------
# Everything is measured in milliseconds, not frames. The workspace page sends
# frames only as fast as the backend answers, so the frame rate swings with
# machine load. A frame-count gate ("10 frames") meant 0.7 s on a fast machine
# and 2 s or more on a slow one.
#
# The letter MLP has no "not-a-letter" class and its softmax saturates, so the
# gates below — not the confidence floor alone — are what keep a hand at rest
# or in transit from spelling nonsense.
# The version that recognised letters well (the original in-browser one) accepted
# a held letter at 0.35. 0.82, and later 0.60/0.72, left real hands that score
# 40-70% showing a letter that never committed. The hold time and the motion
# gate, not a high floor, are what keep a hand at rest from spelling nonsense.
STAB_FRAME_CONF = 0.35     # a frame below this is "no evidence": it pauses the hold, doesn't reset it
STAB_COMMIT_CONF = 0.45    # mean confidence over the hold required to commit
STAB_HOLD_MS = 450.0       # how long one letter must be held
STAB_MIN_FRAMES = 3        # floor on agreeing frames, for very low frame rates
STAB_GRACE_MS = 250.0      # interruptions shorter than this are forgiven
STAB_SWITCH_FRAMES = 2     # consecutive frames of a new letter that replace the current one
# Wrist speed, in normalised image units per second, above which the hand is
# "in transit". 0.5/s is roughly the old 0.030-per-frame at 15 fps, but stays
# the same at any frame rate instead of tightening as fps drops.
STAB_MOTION_SPEED = 0.5


class AlphabetStabilizer:
    """
    Temporal stabilization and debounce layer for fingerspelling.

    Commits a letter once it has been held for ``hold_ms`` with a mean
    confidence of at least ``commit_confidence`` while the hand is still.
    Unlike a strict consecutive-frame streak, a short interruption — one
    low-confidence frame, a one-frame flicker to another letter, a frame
    MediaPipe dropped, a wobble of the wrist — only pauses the hold instead of
    restarting it.

    A committed letter is latched: holding it does not type it again. Taking
    the hand out of frame (or committing a different letter) releases the
    latch, so real double letters ("LL") are still possible.

    src/web_demo/frontend/js/azsl_alphabet.js carries a line-for-line port
    (AlphabetStabilizer) for the in-call path; tests/test_alphabet_stabilizer.py
    holds the two to the same output.
    """

    def __init__(
        self,
        min_confidence: float = STAB_FRAME_CONF,
        commit_confidence: float = STAB_COMMIT_CONF,
        hold_ms: float = STAB_HOLD_MS,
        min_frames: int = STAB_MIN_FRAMES,
        grace_ms: float = STAB_GRACE_MS,
        switch_frames: int = STAB_SWITCH_FRAMES,
        motion_speed: float = STAB_MOTION_SPEED,
    ):
        self.min_confidence = min_confidence
        self.commit_confidence = commit_confidence
        self.hold_ms = hold_ms
        self.min_frames = min_frames
        self.grace_ms = grace_ms
        self.switch_frames = switch_frames
        self.motion_speed = motion_speed

        self.accepted_letter: Optional[str] = None
        self.committed_letter: Optional[str] = None
        self.spelled_word: str = ""
        self._reset_streak()
        self._reset_tracking()

    def _reset_streak(self) -> None:
        self.candidate: Optional[str] = None
        self.cand_start = 0.0
        self.cand_last = 0.0
        self.cand_frames = 0
        self.cand_conf_sum = 0.0
        self._reset_challenger()

    def _reset_challenger(self) -> None:
        self.challenger: Optional[str] = None
        self.challenger_start = 0.0
        self.challenger_frames = 0
        self.challenger_conf_sum = 0.0

    def _reset_tracking(self) -> None:
        self.prev_wrist: Optional[Tuple[float, float]] = None
        self.prev_t: Optional[float] = None
        self.absent_since: Optional[float] = None

    def reset(self) -> None:
        self.accepted_letter = None
        self.committed_letter = None
        self.spelled_word = ""
        self._reset_streak()
        self._reset_tracking()

    def clear_word(self) -> None:
        self.spelled_word = ""

    def backspace(self) -> None:
        if self.spelled_word:
            self.spelled_word = self.spelled_word[:-1]

    def _progress(self) -> float:
        if self.candidate is None:
            return 0.0
        if self.candidate == self.committed_letter:
            return 1.0
        by_time = (self.cand_last - self.cand_start) / self.hold_ms if self.hold_ms > 0 else 1.0
        by_frames = self.cand_frames / self.min_frames if self.min_frames > 0 else 1.0
        return max(0.0, min(1.0, by_time, by_frames))

    def _result(
        self, raw_letter: Optional[str], raw_conf: float, moving: bool, just_accepted: bool
    ) -> Dict[str, Any]:
        progress = self._progress()
        cand_conf = self.cand_conf_sum / self.cand_frames if self.cand_frames else 0.0
        return {
            "raw_letter": raw_letter if raw_letter else "-",
            "raw_confidence": float(raw_conf),
            "stable_candidate": self.candidate if self.candidate else "-",
            "candidate_confidence": float(cand_conf),
            # Half-up, matching Math.round in the JS port (round() is half-even).
            "candidate_ratio": int(progress * 1000 + 0.5) / 1000,
            "candidate_progress": f"{int(progress * 100 + 0.5)}%",
            "is_moving": bool(moving),
            "accepted_letter": self.accepted_letter or "-",
            "just_accepted": just_accepted,
            "spelled_word": self.spelled_word,
        }

    def _commit(self, letter: str) -> None:
        self.accepted_letter = letter
        self.committed_letter = letter
        if letter == "SPACE":
            if self.spelled_word and not self.spelled_word.endswith(" "):
                self.spelled_word += " "
        elif letter == "DEL":
            if self.spelled_word:
                self.spelled_word = self.spelled_word[:-1]
        else:
            self.spelled_word += letter

    def update(
        self,
        raw_letter: Optional[str],
        raw_confidence: float,
        hand_present: bool = True,
        is_moving: bool = False,
        wrist: Optional[Tuple[float, float]] = None,
        now_ms: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        raw_letter / raw_confidence: the classifier's top guess for this frame,
            unfiltered (predict_frame with min_confidence=0), so the UI can show
            what the model sees before it is sure.
        wrist: (x, y) of landmark 0 in normalised image coords; enables the
            built-in per-second motion gate.
        is_moving: an external motion verdict, OR-ed with the built-in one.
        now_ms: frame time in milliseconds; defaults to the monotonic clock.
        """
        now = time.monotonic() * 1000.0 if now_ms is None else float(now_ms)

        # Hand gone. A dropout shorter than the grace window is MediaPipe
        # losing the hand for a frame, not the signer lowering it: keep the
        # streak and the latch. A longer absence resets both.
        if not hand_present:
            if self.absent_since is None:
                self.absent_since = now
            if now - self.absent_since >= self.grace_ms:
                self._reset_streak()
                self.committed_letter = None
                self.prev_wrist = None
                self.prev_t = None
            return self._result(None, 0.0, False, False)
        self.absent_since = None

        # Motion gate, per second rather than per frame, so a lower frame rate
        # doesn't make the same slow movement look like a jump.
        moving = bool(is_moving)
        if wrist is not None:
            if self.prev_wrist is not None and self.prev_t is not None:
                dt_s = max(now - self.prev_t, 1000.0 / 60.0) / 1000.0
                dx = wrist[0] - self.prev_wrist[0]
                dy = wrist[1] - self.prev_wrist[1]
                if (dx * dx + dy * dy) ** 0.5 / dt_s > self.motion_speed:
                    moving = True
            self.prev_wrist = (float(wrist[0]), float(wrist[1]))
            self.prev_t = now

        evidence = (
            not moving and raw_letter is not None and raw_confidence >= self.min_confidence
        )

        if evidence and raw_letter == self.candidate:
            self.cand_frames += 1
            self.cand_conf_sum += raw_confidence
            self.cand_last = now
            self._reset_challenger()
        elif evidence:
            # A different letter. Take it at once if the current candidate is
            # stale; otherwise make it hold for a couple of frames, so a
            # single-frame flicker can't wipe out a hold in progress.
            if raw_letter == self.challenger:
                self.challenger_frames += 1
                self.challenger_conf_sum += raw_confidence
            else:
                self.challenger = raw_letter
                self.challenger_start = now
                self.challenger_frames = 1
                self.challenger_conf_sum = raw_confidence
            stale = self.candidate is None or now - self.cand_last > self.grace_ms
            if stale or self.challenger_frames >= self.switch_frames:
                self.candidate = self.challenger
                self.cand_start = self.challenger_start
                self.cand_last = now
                self.cand_frames = self.challenger_frames
                self.cand_conf_sum = self.challenger_conf_sum
                self._reset_challenger()
        elif self.candidate is not None and now - self.cand_last > self.grace_ms:
            # No usable evidence for longer than the grace window: the hand is
            # in transit or the pose isn't a letter. Drop the hold.
            self._reset_streak()

        just_accepted = False
        if (
            evidence
            and self.candidate == raw_letter
            and self.candidate != self.committed_letter
            and self.cand_frames >= self.min_frames
            and self.cand_last - self.cand_start >= self.hold_ms
            and self.cand_conf_sum / self.cand_frames >= self.commit_confidence
        ):
            self._commit(self.candidate)
            just_accepted = True

        return self._result(raw_letter, raw_confidence, moving, just_accepted)
