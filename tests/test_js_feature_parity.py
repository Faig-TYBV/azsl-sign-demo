"""
The browser's feature extraction must equal the Python pipeline's, exactly.

Word-mode recognition now happens in two places: MediaPipe runs in the browser
and only the 126-dim vectors travel to the server. Those vectors feed a GRU
trained on the output of ``src/features/extract_landmarks.py``.

A divergence here does not raise anything. It degrades accuracy quietly, and
would be blamed on the model, the camera, or the network long before anyone
suspected the feature code. So the two implementations are run against shared
inputs and required to agree.

Skipped when node is unavailable; the JS cannot be exercised without it.
"""

import json
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

from src.features.extract_landmarks import (  # noqa: E402
    FrameResult,
    normalize_frame,
    normalize_hand,
)

JS_FILE = PROJECT_ROOT / "src" / "web_demo" / "frontend" / "js" / "azsl_features.js"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is required to run the browser code"
)


def run_js(hands):
    """Feed `hands` through azsl_features.js under node, return the 126 floats."""
    harness = f"""
const path = {json.dumps(str(JS_FILE))};
const F = require(path);
// argv[0] is node and argv[1] is this script, so the payload is argv[2].
const hands = JSON.parse(process.argv[2]);
process.stdout.write(JSON.stringify(F.frameFeatures126(hands)));
"""
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "harness.cjs"
        script.write_text(harness, encoding="utf-8")
        out = subprocess.run(
            ["node", str(script), json.dumps(hands)],
            capture_output=True, text=True, timeout=60,
        )
    if out.returncode != 0:
        raise AssertionError(f"node failed: {out.stderr}")
    return np.array(json.loads(out.stdout), dtype=np.float32)


def run_python(hands):
    """The same input through the pipeline the model was trained on."""
    frame = FrameResult(num_hands=len(hands))
    # Mirror _mp_result_to_frame_result: stable sort on the handedness label.
    order = sorted(range(len(hands)), key=lambda i: hands[i]["label"])
    for slot, idx in enumerate(order[:2]):
        frame.landmarks[slot] = np.array(hands[idx]["landmarks"], dtype=np.float32)
        frame.handedness.append(hands[idx]["label"])
    return normalize_frame(frame)


def random_hand(rng, scale=1.0, offset=(0.0, 0.0, 0.0)):
    lm = rng.uniform(-1, 1, size=(21, 3)).astype(np.float32) * scale
    lm += np.array(offset, dtype=np.float32)
    # Keep the wrist->middle-MCP reference well clear of the degenerate case.
    lm[9] = lm[0] + np.array([0.3, 0.4, 0.1], dtype=np.float32) * scale
    return lm.tolist()


def assert_match(hands, label):
    js = run_js(hands)
    py = run_python(hands)
    assert js.shape == py.shape == (126,), f"{label}: shape {js.shape} vs {py.shape}"
    worst = float(np.max(np.abs(js - py)))
    assert worst < 1e-5, (
        f"{label}: JS and Python disagree by {worst:.3e}\n"
        f"  first mismatch at index {int(np.argmax(np.abs(js - py)))}"
    )


def test_single_hand_matches():
    rng = np.random.default_rng(1)
    assert_match([{"label": "Right", "landmarks": random_hand(rng)}], "one hand")


def test_two_hands_match_and_respect_slot_order():
    rng = np.random.default_rng(2)
    left = random_hand(rng, offset=(-0.5, 0, 0))
    right = random_hand(rng, offset=(0.5, 0, 0))

    # Given in either order, "Left" must land in slot 0 both times.
    a = [{"label": "Left", "landmarks": left}, {"label": "Right", "landmarks": right}]
    b = [{"label": "Right", "landmarks": right}, {"label": "Left", "landmarks": left}]
    assert_match(a, "two hands, left first")
    assert_match(b, "two hands, right first")

    assert np.allclose(run_js(a), run_js(b), atol=1e-6), (
        "slot assignment must depend on the handedness label, not arrival order"
    )


def test_no_hands_is_all_zero():
    js = run_js([])
    assert js.shape == (126,)
    assert not js.any()
    assert np.array_equal(js, run_python([]))


def test_degenerate_hand_is_zeroed_not_nan():
    """A collapsed detection must not divide by ~0 and emit NaNs."""
    flat = [[0.5, 0.5, 0.0] for _ in range(21)]  # wrist and middle MCP coincide
    js = run_js([{"label": "Right", "landmarks": flat}])
    py = run_python([{"label": "Right", "landmarks": flat}])
    assert not np.isnan(js).any(), "JS produced NaNs the model never saw in training"
    assert not js[:63].any(), "a degenerate hand must be zero-filled"
    assert np.array_equal(js, py)


def test_scale_invariance_holds_in_both():
    """Doubling the hand's size must not change the features, in either impl."""
    rng = np.random.default_rng(5)
    small = np.array(random_hand(rng, scale=0.5))
    big = (small * 2.0).tolist()

    js_small = run_js([{"label": "Right", "landmarks": small.tolist()}])
    js_big = run_js([{"label": "Right", "landmarks": big}])
    assert np.allclose(js_small, js_big, atol=1e-5), "JS lost scale invariance"
    assert np.allclose(js_small, run_python([{"label": "Right", "landmarks": small.tolist()}]), atol=1e-5)


def test_translation_invariance_holds_in_both():
    """Moving the hand across the frame must not change the features."""
    rng = np.random.default_rng(6)
    base = np.array(random_hand(rng))
    moved = (base + np.array([0.25, -0.15, 0.05], dtype=np.float32)).tolist()

    assert np.allclose(
        run_js([{"label": "Right", "landmarks": base.tolist()}]),
        run_js([{"label": "Right", "landmarks": moved}]),
        atol=1e-5,
    ), "JS lost translation invariance"


@pytest.mark.parametrize("seed", [11, 22, 33, 44, 55])
def test_random_frames_match(seed):
    """Fuzz: the two implementations must agree on arbitrary plausible input."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 3))
    labels = ["Left", "Right"][:n]
    hands = [{"label": labels[i], "landmarks": random_hand(rng)} for i in range(n)]
    assert_match(hands, f"random seed {seed}")


def test_python_normalize_hand_still_uses_the_reference_this_js_assumes():
    """Guard the assumption the JS is built on.

    If the Python ever changes its scale reference away from wrist->landmark 9,
    the JS silently becomes wrong. Catch that here rather than in the field.
    """
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[9] = [2.0, 0.0, 0.0]          # reference length 2
    lm[5] = [4.0, 0.0, 0.0]
    out = normalize_hand(lm)
    assert np.allclose(out[5], [2.0, 0.0, 0.0]), (
        "normalize_hand no longer scales by ||wrist - landmark 9||; "
        "azsl_features.js must be updated to match"
    )


# --------------------------------------------------------------------------- #
# Speech recognition: wrong-language results
# --------------------------------------------------------------------------- #
FRIENDS_HTML = PROJECT_ROOT / "src" / "web_demo" / "frontend" / "friends.html"


def _extract_script_guard(tmp: Path) -> Path:
    """Pull the pure script-detection helpers out of the page for node."""
    import re

    html = FRIENDS_HTML.read_text(encoding="utf-8")
    script = re.findall(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", html, re.S)[0]
    body = script[script.index("const CYRILLIC_RE"): script.index("function startSpeech()")]
    out = tmp / "guard.cjs"
    out.write_text(
        body + "\nmodule.exports={scriptMismatch,stripWrongScript};\n", encoding="utf-8"
    )
    return out


def _run_guard(fn, text, lang):
    """Run one guard function under node.

    The payload goes through argv as ASCII-escaped JSON, and stdout is decoded
    as UTF-8 explicitly. Passing "necəsən" as a raw argument mangles it to
    "necЙ™sЙ™n" on Windows, where argv and the default pipe encoding follow the
    system codepage rather than UTF-8 — a test artefact that looks exactly like
    the encoding bug being tested for.
    """
    with tempfile.TemporaryDirectory() as tmp:
        guard = _extract_script_guard(Path(tmp))
        runner = Path(tmp) / "run.cjs"
        runner.write_text(
            f"const g=require({json.dumps(str(guard))});"
            "const a=JSON.parse(process.argv[2]);"
            f"process.stdout.write(JSON.stringify(g[{json.dumps(fn)}](a.text, a.lang)));",
            encoding="utf-8",
        )
        payload = json.dumps({"text": text, "lang": lang}, ensure_ascii=True)
        out = subprocess.run(
            ["node", str(runner), payload],
            capture_output=True, encoding="utf-8", timeout=60,
        )
    if out.returncode != 0:
        raise AssertionError(f"node failed: {out.stderr}")
    return json.loads(out.stdout)


@pytest.mark.parametrize(
    "text,lang,expected",
    [
        # Real output from a reported session: az-AZ requested, Russian returned.
        ("salam", "az-AZ", False),
        ("necəsən", "az-AZ", False),
        ("nə var nə yox", "az-AZ", False),
        ("удал", "az-AZ", True),
        ("ить", "az-AZ", True),
        ("ть мой", "az-AZ", True),
        ("и", "az-AZ", True),
        # The one a majority rule would wave through: 7 Latin vs 4 Cyrillic.
        ("necəsən удал", "az-AZ", True),
        # Russian is legitimate when Russian is what was asked for.
        ("Привет как дела", "ru-RU", False),
        ("salam", "ru-RU", True),
        ("Nasılsın", "tr-TR", False),
    ],
)
def test_wrong_script_results_are_detected(text, lang, expected):
    """Azerbaijani is written in Latin script; Cyrillic means a wrong language.

    Chrome's speech engine silently falls back when it cannot serve the
    requested language, and in this region it falls back to Russian. The result
    is not a near miss to be corrected later — Russian in the middle of an
    Azerbaijani sentence is worse than nothing.
    """
    assert _run_guard("scriptMismatch", text, lang) is expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("necəsən удал", "necəsən"),
        ("salam ить necəsən", "salam necəsən"),
        ("удал ить", ""),                 # nothing worth keeping
        ("nə var nə yox", "nə var nə yox"),
    ],
)
def test_wrong_script_words_are_stripped_not_the_whole_phrase(text, expected):
    """The engine often gets part of an utterance right; keep that part."""
    assert _run_guard("stripWrongScript", text, "az-AZ") == expected


# --------------------------------------------------------------------------- #
# Alphabet: the browser classifier must agree with the server's
# --------------------------------------------------------------------------- #
ALPHABET_JS = PROJECT_ROOT / "src" / "web_demo" / "frontend" / "js" / "azsl_alphabet.js"
ALPHABET_MODEL = (
    PROJECT_ROOT / "src" / "web_demo" / "frontend" / "models" / "azsl_hierarchical_model.json"
)


def run_js_alphabet(landmarks, handedness):
    """Classify one hand in the browser implementation, under node.

    Deliberately mirrors what friends.html does, including the control-gesture
    ordering, so this fails if the page and the Python ever diverge.
    """
    harness = f"""
global.window = global;
require({json.dumps(str(ALPHABET_JS))});
const A = global.AzslAlphabet;
A.setAzslModel(require({json.dumps(str(ALPHABET_MODEL))}));
const a = JSON.parse(process.argv[2]);
const points = a.landmarks.map(p => ({{ x: p[0], y: p[1], z: p[2] }}));
const mirrorX = a.handedness === 'Left';
const coords = A.normalizeLandmarks(points, mirrorX);
let out;
const control = A.detectControlGesture(coords);
if (control) out = {{ label: control.label, confidence: control.confidence }};
else {{
  const r = A.classifyHierarchical(coords, {{ x: 0, y: 0 }});
  out = {{ label: r.label, confidence: r.confidence || 0 }};
}}
process.stdout.write(JSON.stringify(out));
"""
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "alpha.cjs"
        script.write_text(harness, encoding="utf-8")
        payload = json.dumps(
            {"landmarks": landmarks, "handedness": handedness}, ensure_ascii=True
        )
        out = subprocess.run(
            ["node", str(script), payload], capture_output=True, encoding="utf-8", timeout=60
        )
    if out.returncode != 0:
        raise AssertionError(f"node failed: {out.stderr}")
    return json.loads(out.stdout)


def run_python_alphabet(landmarks, handedness):
    from src.inference.alphabet_classifier import AlphabetClassifier

    clf = AlphabetClassifier()
    label, conf = clf.predict_frame(
        np.array(landmarks, dtype=np.float32), handedness=handedness, min_confidence=0.0
    )
    return {"label": label, "confidence": float(conf)}


def plausible_hand(rng):
    """A hand-shaped set of landmarks, not uniform noise.

    Random points give a degenerate pose both implementations reject the same
    way, which would let a real divergence pass unnoticed.
    """
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[0] = [0.5, 0.8, 0.0]                                   # wrist
    lm[9] = lm[0] + [0.0, -0.18, 0.0]                         # middle MCP
    for finger, base in enumerate([1, 5, 9, 13, 17]):
        spread = (finger - 2) * 0.045
        for j in range(4):
            idx = base + j
            if idx > 20:
                break
            lm[idx] = lm[0] + [spread, -0.05 - 0.045 * j, 0.0]
    lm += rng.normal(0, 0.012, size=(21, 3)).astype(np.float32)
    return lm.tolist()


@pytest.mark.parametrize("handedness", ["Left", "Right"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6])
def test_browser_alphabet_agrees_with_the_server(seed, handedness):
    """The page classifies fingerspelling locally; it must not drift from Python.

    The two run the same algorithm from the same weights, so they should agree
    exactly. They previously did not: the page called predictGesture() with a
    missing velocity argument, which threw on every frame, and passed arrays
    where the module expects {x, y, z} objects.
    """
    rng = np.random.default_rng(seed)
    landmarks = plausible_hand(rng)

    js = run_js_alphabet(landmarks, handedness)
    py = run_python_alphabet(landmarks, handedness)

    assert js["label"] == py["label"], (
        f"browser said {js['label']!r}, server said {py['label']!r} "
        f"for the same hand ({handedness})"
    )
    assert abs(js["confidence"] - py["confidence"]) < 1e-4, (
        f"confidence differs: browser {js['confidence']:.6f} vs "
        f"server {py['confidence']:.6f}"
    )


def test_browser_alphabet_returns_a_real_prediction_not_an_exception():
    """Guards the actual regression: a call that threw on every frame.

    The failure was silent — swallowed into a debug-only log — so alphabet mode
    simply never produced a letter, which reads as a bad model rather than a
    broken call.
    """
    rng = np.random.default_rng(99)
    out = run_js_alphabet(plausible_hand(rng), "Right")
    assert "label" in out and "confidence" in out
    assert isinstance(out["confidence"], (int, float))
    assert out["confidence"] > 0, "a valid hand must yield a non-zero confidence"
