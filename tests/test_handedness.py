"""
The same sign with the left hand must give the same result as with the right.

A left-handed sign is the mirror image of the right-handed one: in the camera
image x -> 1 - x, and MediaPipe's left/right labels swap. These tests build
exactly that mirror image and require the served result to be identical, in
both modes, rather than just checking that a flag is set.
"""

import asyncio
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.extract_landmarks import FrameResult, normalize_frame  # noqa: E402
from src.inference import handedness as H  # noqa: E402

FEATURES_JS = PROJECT_ROOT / "src" / "web_demo" / "frontend" / "js" / "azsl_features.js"
TRIAL_FRAMES = 26


def flip_label(label):
    return {"Left": "Right", "Right": "Left"}[label]


def raw_label_for(hand):
    """The label MediaPipe would report for a real `hand` on a raw frame."""
    return flip_label(hand) if H.MEDIAPIPE_LABELS_ARE_SWAPPED else hand


def hand_landmarks(rng, center):
    lm = rng.uniform(-0.08, 0.08, size=(21, 3)).astype(np.float32)
    lm[:, :2] += np.array(center, dtype=np.float32)
    lm[9] = lm[0] + np.array([0.02, -0.09, 0.0], dtype=np.float32)
    return lm


def frame(hands):
    """hands: list of (raw_label, (21, 3) image-space landmarks)."""
    fr = FrameResult(num_hands=len(hands))
    for slot, (label, lm) in enumerate(sorted(hands, key=lambda h: h[0])):
        fr.landmarks[slot] = lm
        fr.handedness.append(label)
    return fr


def mirror_image(hands):
    """What the camera sees when the signer mirrors the sign."""
    out = []
    for label, lm in hands:
        m = lm.copy()
        m[:, 0] = 1.0 - m[:, 0]
        out.append((flip_label(label), m))
    return out


def one_hand_trial(rng, hand="Right"):
    lm = hand_landmarks(rng, (0.4, 0.5))
    drift = rng.normal(0, 0.004, size=(21, 3)).astype(np.float32)
    return [[(raw_label_for(hand), lm + drift * t)] for t in range(TRIAL_FRAMES)]


def two_hand_trial(rng, dominant="Right"):
    """The dominant hand changes shape; the other stays still."""
    active = hand_landmarks(rng, (0.35, 0.5))
    still = hand_landmarks(rng, (0.65, 0.55))
    drift = rng.normal(0, 0.01, size=(21, 3)).astype(np.float32)
    other = flip_label(dominant)
    return [
        [(raw_label_for(dominant), active + drift * t), (raw_label_for(other), still)]
        for t in range(TRIAL_FRAMES)
    ]


def as_features(trial):
    feats = np.stack([normalize_frame(frame(h)) for h in trial]).astype(np.float32)
    labels = [list(frame(h).handedness) for h in trial]
    return feats, labels


# --- the transform itself ------------------------------------------------------

def test_mirroring_a_frame_matches_mirroring_the_image():
    rng = np.random.default_rng(0)
    for hands in (one_hand_trial(rng)[0], two_hand_trial(rng)[0]):
        direct = normalize_frame(frame(mirror_image(hands)))
        via_features = H.mirror_frame_126(normalize_frame(frame(hands)))
        np.testing.assert_allclose(via_features, direct, atol=1e-6)


def test_mirroring_twice_is_the_identity():
    rng = np.random.default_rng(1)
    for hands in (one_hand_trial(rng)[0], two_hand_trial(rng)[0]):
        f = normalize_frame(frame(hands))
        np.testing.assert_array_equal(H.mirror_frame_126(H.mirror_frame_126(f)), f)


@pytest.mark.parametrize("builder", [one_hand_trial, two_hand_trial])
def test_left_and_right_trials_canonicalize_identically(builder):
    rng = np.random.default_rng(2)
    right = builder(rng, "Right")
    left = [mirror_image(h) for h in right]

    r_feats, r_mirrored = H.canonicalize_trial(*as_features(right))
    l_feats, l_mirrored = H.canonicalize_trial(*as_features(left))
    assert (r_mirrored, l_mirrored) == (False, True)
    np.testing.assert_allclose(l_feats, r_feats, atol=1e-6)


def test_one_mislabelled_frame_does_not_flip_the_trial():
    rng = np.random.default_rng(3)
    trial = one_hand_trial(rng, "Right")
    feats, labels = as_features(trial)
    labels[5] = [flip_label(labels[5][0])]
    assert H.dominant_hand(feats, labels) == "Right"


def test_unknown_labels_leave_the_trial_untouched():
    """The parity test drives inference without labels; that must stay exact."""
    rng = np.random.default_rng(4)
    feats, _ = as_features(one_hand_trial(rng, "Left"))
    out, mirrored = H.canonicalize_trial(feats, [[] for _ in feats])
    assert mirrored is False
    np.testing.assert_array_equal(out, feats)


def test_browser_labels_are_sanitised_in_place():
    assert H.clean_labels(["hand0", "Right", "Left"]) == [None, "Right"]
    assert H.clean_labels("Left") == []
    assert H.clean_labels(None) == []


# --- served results --------------------------------------------------------------

torch = pytest.importorskip("torch", reason="the recognition stack is optional")
pytest.importorskip("mediapipe")


class FakeWS:
    def __init__(self):
        self.msgs = []

    async def send_text(self, text):
        self.msgs.append(json.loads(text))


class FakeState:
    def __init__(self):
        self.recorded_features = []
        self.recorded_validity = []
        self.recorded_labels = []
        self.word_state = "RECORDING"
        self.last_word_result = None


def serve_word(trial):
    from src.web_demo import backend as B

    async def run():
        state, ws = FakeState(), FakeWS()
        for hands in trial:
            fr = frame(hands)
            await B._record_word_frame(
                state, ws, normalize_frame(fr), 1.0, hand_labels=list(fr.handedness)
            )
        return [m for m in ws.msgs if m.get("word_state") == "RESULT"][0]

    return asyncio.run(run())


@pytest.mark.parametrize("seed", [10, 11, 12])
@pytest.mark.parametrize("builder", [one_hand_trial, two_hand_trial])
def test_word_mode_gives_the_same_word_with_either_hand(builder, seed):
    right = builder(np.random.default_rng(seed), "Right")
    left = [mirror_image(h) for h in right]
    r, l = serve_word(right), serve_word(left)
    assert l["prediction"] == r["prediction"]
    assert l["confidence"] == pytest.approx(r["confidence"], abs=1e-5)


@pytest.mark.parametrize("seed", range(8))
def test_alphabet_mode_gives_the_same_letter_with_either_hand(seed):
    from src.inference.alphabet_classifier import AlphabetClassifier

    clf = AlphabetClassifier()
    rng = np.random.default_rng(seed)
    lm = hand_landmarks(rng, (0.45, 0.5))
    mirrored = lm.copy()
    mirrored[:, 0] = 1.0 - mirrored[:, 0]

    # Exactly what backend.py does: real hand from MediaPipe's raw label.
    r = clf.predict_frame(lm, H.actual_hand(raw_label_for("Right")), min_confidence=0.0)
    l = clf.predict_frame(mirrored, H.actual_hand(raw_label_for("Left")), min_confidence=0.0)
    assert l[0] == r[0]
    assert l[1] == pytest.approx(r[1], abs=1e-5)


# --- browser side ----------------------------------------------------------------

def test_js_uses_the_same_handedness_setting():
    js = FEATURES_JS.read_text(encoding="utf-8")
    m = re.search(r"var MEDIAPIPE_LABELS_ARE_SWAPPED = (true|false);", js)
    assert m, "azsl_features.js no longer declares the setting"
    assert (m.group(1) == "true") == H.MEDIAPIPE_LABELS_ARE_SWAPPED


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_js_labels_follow_the_feature_slots():
    harness = f"""
const F = require({json.dumps(str(FEATURES_JS))});
const hands = [{{label: 'Right', landmarks: []}}, {{label: 'Left', landmarks: []}}];
process.stdout.write(JSON.stringify({{
  labels: F.frameLabels(hands),
  one: F.frameLabels([{{label: 'Right', landmarks: []}}]),
  none: F.frameLabels([]),
  actual: [F.actualHand('Left'), F.actualHand('Right'), F.actualHand('hand0')],
}}));
"""
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "h.cjs"
        script.write_text(harness, encoding="utf-8")
        out = subprocess.run(["node", str(script)], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    got = json.loads(out.stdout)
    # Python's sorted() puts "Left" in slot 0, as frameFeatures126 does.
    assert got["labels"] == ["Left", "Right"]
    assert got["one"] == ["Right"]
    assert got["none"] == []
    assert got["actual"] == [H.actual_hand("Left"), H.actual_hand("Right"), None]
