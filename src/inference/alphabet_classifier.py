"""
Alphabet (fingerspelling) classifier for AzSL single-frame inference.

Python implementation of the 84-dimensional feature extraction and hierarchical
MLP classification defined in src/web_demo/frontend/js/azsl_alphabet.js and
trained via src/web_demo/frontend/models/azsl_hierarchical_model.json.
"""

from __future__ import annotations

import json
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


def _apply_scaler(input_vec: np.ndarray, scaler: Dict[str, Any]) -> np.ndarray:
    mean = np.array(scaler["mean"], dtype=np.float32)
    std = np.array(scaler["std"], dtype=np.float32)
    std = np.where(std == 0, 1.0, std)
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


class AlphabetStabilizer:
    """
    Temporal stabilization and debounce layer for fingerspelling.
    Prevents single-frame flickering, duplicate letter additions while a gesture is held,
    and handles control gestures (SPACE / DEL).
    """

    def __init__(
        self,
        min_confidence: float = MIN_CONFIDENCE,
        stability_frames: int = 3,
    ):
        self.min_confidence = min_confidence
        self.stability_frames = stability_frames

        self.candidate: Optional[str] = None
        self.consecutive_count: int = 0
        self.candidate_conf: float = 0.0
        self.accepted_letter: Optional[str] = None
        self.committed_letter: Optional[str] = None
        self.already_committed: bool = False
        self.spelled_word: str = ""

    def reset(self) -> None:
        self.candidate = None
        self.consecutive_count = 0
        self.candidate_conf = 0.0
        self.accepted_letter = None
        self.committed_letter = None
        self.already_committed = False
        self.spelled_word = ""

    def clear_word(self) -> None:
        self.spelled_word = ""

    def backspace(self) -> None:
        if self.spelled_word:
            self.spelled_word = self.spelled_word[:-1]

    def update(
        self,
        raw_letter: Optional[str],
        raw_confidence: float,
        hand_present: bool = True,
        is_moving: bool = False,
    ) -> Dict[str, Any]:
        just_accepted = False

        # Hand fully out of frame: reset everything AND release the commit latch,
        # so deliberately lowering the hand and re-signing the same letter (a
        # real double letter, e.g. "AA") works.
        if not hand_present:
            self.candidate = None
            self.consecutive_count = 0
            self.candidate_conf = 0.0
            self.committed_letter = None
            self.already_committed = False
            return {
                "raw_letter": "-",
                "raw_confidence": 0.0,
                "stable_candidate": "-",
                "candidate_progress": f"0/{self.stability_frames}",
                "accepted_letter": self.accepted_letter or "-",
                "just_accepted": False,
                "spelled_word": self.spelled_word,
            }

        # Hand present but not spelling right now: it's mid-motion between poses,
        # the frame has no confident letter, or confidence is below threshold.
        # Drop the in-progress streak so nothing commits, but KEEP committed_letter
        # so a single flicker can't double-type the letter just accepted.
        if is_moving or raw_letter is None or raw_confidence < self.min_confidence:
            self.candidate = None
            self.consecutive_count = 0
            self.candidate_conf = 0.0
            return {
                "raw_letter": raw_letter if raw_letter else "-",
                "raw_confidence": float(raw_confidence),
                "stable_candidate": "-",
                "candidate_progress": f"0/{self.stability_frames}",
                "accepted_letter": self.accepted_letter or "-",
                "just_accepted": False,
                "spelled_word": self.spelled_word,
            }

        # Hand is present and confidence >= min_confidence
        if raw_letter == self.candidate:
            self.consecutive_count += 1
            self.candidate_conf = max(self.candidate_conf, raw_confidence)
        else:
            self.candidate = raw_letter
            self.consecutive_count = 1
            self.candidate_conf = raw_confidence

        # Check if candidate is confirmed for stability_frames
        if self.consecutive_count >= self.stability_frames:
            if self.candidate != self.committed_letter:
                self.accepted_letter = self.candidate
                self.committed_letter = self.candidate
                self.already_committed = True
                just_accepted = True

                if self.candidate == "SPACE":
                    if self.spelled_word and not self.spelled_word.endswith(" "):
                        self.spelled_word += " "
                elif self.candidate == "DEL":
                    if self.spelled_word:
                        self.spelled_word = self.spelled_word[:-1]
                else:
                    self.spelled_word += self.candidate

        return {
            "raw_letter": raw_letter,
            "raw_confidence": float(raw_confidence),
            "stable_candidate": self.candidate if self.candidate else "-",
            "candidate_progress": f"{min(self.consecutive_count, self.stability_frames)}/{self.stability_frames}",
            "accepted_letter": self.accepted_letter or "-",
            "just_accepted": just_accepted,
            "spelled_word": self.spelled_word,
        }
