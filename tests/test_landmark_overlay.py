"""
The MediaPipe landmark overlay on the recognition workspace.

Why this is tested rather than eyeballed: the overlay's whole job is to tell
"the model read my hand wrongly" apart from "the model never saw my hand". If
its geometry is off, it draws a skeleton beside the hand — which looks exactly
like a tracking failure, so a broken diagnostic actively misleads instead of
merely not helping.

Two things are checked: the payload the server attaches to each frame reply,
and the coordinate mapping the page uses to place it, which is run under node
against the same cases a real video element would produce.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FRONTEND = PROJECT_ROOT / "src" / "web_demo" / "frontend"
INDEX = FRONTEND / "index.html"


# --------------------------------------------------------------------------- #
# The server-side payload
# --------------------------------------------------------------------------- #
class FakeFrameResult:
    """Stands in for extract_landmarks.FrameResult without importing MediaPipe."""

    def __init__(self, num_hands, landmarks=None):
        self.num_hands = num_hands
        self.handedness = ["Right", "Left"][:num_hands]
        self.landmarks = (
            landmarks if landmarks is not None
            else np.zeros((2, 21, 3), dtype=np.float32)
        )


def _overlay_fn():
    """Import the helper without pulling in torch/MediaPipe at module scope."""
    pytest.importorskip("mediapipe", reason="backend.py imports the ML stack")
    pytest.importorskip("torch")
    from src.web_demo.backend import _landmarks_for_overlay

    return _landmarks_for_overlay


def test_no_hand_sends_an_empty_list_not_a_missing_key():
    """The page distinguishes [] from absent: one clears the overlay, the other
    leaves the last drawing. A no-hand frame must clear it."""
    fn = _overlay_fn()
    assert fn(FakeFrameResult(0)) == []


def test_one_hand_sends_twenty_one_xy_pairs():
    fn = _overlay_fn()
    lm = np.zeros((2, 21, 3), dtype=np.float32)
    lm[0, :, 0] = np.linspace(0.0, 1.0, 21)
    lm[0, :, 1] = 0.5
    lm[0, :, 2] = 9.0          # z is deliberately not sent

    hands = fn(FakeFrameResult(1, lm))

    assert len(hands) == 1
    assert len(hands[0]) == 21
    assert all(len(p) == 2 for p in hands[0]), "z must not be sent - it is not drawn"
    assert hands[0][0] == [0.0, 0.5]
    assert hands[0][-1] == [1.0, 0.5]


def test_two_hands_are_kept_separate():
    """They are drawn in different colours, which is how you notice MediaPipe
    latching onto a face or a background object as a second hand."""
    fn = _overlay_fn()
    lm = np.zeros((2, 21, 3), dtype=np.float32)
    lm[0, :, 0] = 0.25
    lm[1, :, 0] = 0.75

    hands = fn(FakeFrameResult(2, lm))

    assert len(hands) == 2
    assert hands[0][0][0] == 0.25
    assert hands[1][0][0] == 0.75


def test_coordinates_are_rounded_to_keep_the_payload_small():
    """At 15 fps this rides on every frame reply, so the rounding is the
    difference between roughly 500 bytes and 2 KB per frame."""
    fn = _overlay_fn()
    lm = np.zeros((2, 21, 3), dtype=np.float32)
    lm[0, 0] = [0.123456789, 0.987654321, 0.0]

    point = fn(FakeFrameResult(1, lm))[0][0]

    assert point == [round(point[0], 4), round(point[1], 4)]
    assert len(json.dumps(point)) <= 18, point


def test_a_hand_count_larger_than_the_array_does_not_crash():
    """num_hands and the landmark array come from different fields; trusting
    num_hands alone would IndexError on a malformed result."""
    fn = _overlay_fn()
    small = np.zeros((1, 21, 3), dtype=np.float32)
    assert len(fn(FakeFrameResult(2, small))) == 1


def test_frame_replies_carry_the_overlay_key():
    """A reply without it leaves the overlay frozen on the previous frame."""
    backend = (PROJECT_ROOT / "src" / "web_demo" / "backend.py").read_text(encoding="utf-8")

    # Word mode: both the recording and the analysing reply.
    assert backend.count('"landmarks": landmarks or []') == 2
    # Alphabet mode reads them straight off the detection result.
    assert '"landmarks": _landmarks_for_overlay(frame_result)' in backend
    # And the JPEG path passes what it detected into the word recorder.
    assert "landmarks=_landmarks_for_overlay(frame_result)" in backend


def test_the_browser_feature_path_does_not_claim_server_landmarks():
    """friends.html detects locally and draws its own overlay. The server has no
    landmarks on that path, so it must not invent any."""
    backend = (PROJECT_ROOT / "src" / "web_demo" / "backend.py").read_text(encoding="utf-8")
    # The features call site passes no landmarks, so the default None -> [].
    assert "await _record_word_frame(state, websocket, feat_126, valid)\n" in backend
    assert "landmarks: list | None = None" in backend


# --------------------------------------------------------------------------- #
# The page's coordinate mapping, run under node
# --------------------------------------------------------------------------- #
def _extract(name: str) -> str:
    """Pull one function out of index.html by name."""
    html = INDEX.read_text(encoding="utf-8")
    start = html.index(f"function {name}(")
    depth = 0
    for i in range(start, len(html)):
        if html[i] == "{":
            depth += 1
        elif html[i] == "}":
            depth -= 1
            if depth == 0:
                return html[start:i + 1]
    raise AssertionError(f"could not bound function {name}")


def run_rect(client_w, client_h, video_w, video_h, fit):
    """Evaluate videoContentRect() under node for one geometry."""
    node = shutil_which_node()
    src = _extract("videoContentRect")
    harness = f"""
const video = {{
  clientWidth: {client_w}, clientHeight: {client_h},
  videoWidth: {video_w}, videoHeight: {video_h},
}};
const window = {{ getComputedStyle: () => ({{ objectFit: {fit!r} }}) }};
{src}
console.log(JSON.stringify(videoContentRect()));
"""
    tmp = Path(tempfile.gettempdir()) / "azsl_rect_probe.js"
    tmp.write_text(harness, encoding="utf-8")
    out = subprocess.run([node, str(tmp)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def shutil_which_node():
    import shutil

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    return node


def test_contain_letterboxes_and_the_mapping_accounts_for_it():
    """A 4:3 frame in a 16:9 box has bars left and right.

    Mapping normalised coordinates onto the element box instead of the picture
    would push the skeleton sideways by the width of one bar - which reads as a
    tracking error, not a layout bug.
    """
    rect = run_rect(1600, 900, 480, 360, "contain")

    # 4:3 scaled to 900 tall = 1200 wide, centred in 1600 -> 200px bars.
    assert rect["h"] == pytest.approx(900)
    assert rect["w"] == pytest.approx(1200)
    assert rect["x"] == pytest.approx(200)
    assert rect["y"] == pytest.approx(0)


def test_cover_crops_and_is_handled_too():
    """The mobile breakpoint switches to object-fit: cover, so the same maths
    has to work with the picture overflowing instead of inset."""
    rect = run_rect(360, 640, 480, 360, "cover")

    # Filling 640 tall from a 4:3 source needs 853px of width, overflowing 360.
    assert rect["h"] == pytest.approx(640)
    assert rect["w"] == pytest.approx(853.333, abs=0.01)
    assert rect["x"] == pytest.approx((360 - 853.333) / 2, abs=0.01)
    assert rect["x"] < 0, "cover overflows, so the origin is negative"


def test_an_exact_aspect_match_has_no_offset():
    rect = run_rect(480, 360, 480, 360, "contain")
    assert (rect["x"], rect["y"]) == (0, 0)
    assert (rect["w"], rect["h"]) == (480, 360)


def test_a_video_with_no_dimensions_yet_falls_back_to_the_box():
    """videoWidth is 0 until metadata loads; dividing by it would give NaN and
    the canvas would silently draw nothing."""
    rect = run_rect(640, 480, 0, 0, "contain")
    assert rect == {"x": 0, "y": 0, "w": 640, "h": 480}


# --------------------------------------------------------------------------- #
# Page wiring
# --------------------------------------------------------------------------- #
def test_the_overlay_is_actually_drawn_somewhere():
    """The canvas element existed for a long time with nothing drawing to it -
    styled, positioned, and completely dead. That is what this guards."""
    html = INDEX.read_text(encoding="utf-8")

    assert 'id="landmark-canvas"' in html
    assert "getElementById('landmark-canvas')" in html, "the canvas is never claimed"
    assert "function drawLandmarks(" in html
    assert "if (Array.isArray(data.landmarks)) drawLandmarks(data.landmarks);" in html, (
        "nothing calls the drawing function on a frame reply"
    )


def test_the_skeleton_covers_every_finger():
    """21 points with the palm edge is 21 bones; a missing one leaves a finger
    detached, which looks like a detection fault."""
    html = INDEX.read_text(encoding="utf-8")
    block = html[html.index("const HAND_CONNECTIONS"):]
    block = block[:block.index("];")]
    pairs = [
        tuple(int(n) for n in pair.split(","))
        for pair in __import__("re").findall(r"\[(\d+,\s*\d+)\]", block)
    ]
    assert len(pairs) == 21, f"expected 21 bones, found {len(pairs)}"
    # Every landmark except none should appear; all 21 joints must be connected.
    touched = {i for pair in pairs for i in pair}
    assert touched == set(range(21)), f"unconnected landmarks: {set(range(21)) - touched}"


def test_the_overlay_is_cleared_when_there_is_nothing_to_show():
    """Three things invalidate the drawing: no camera, a new trial, and a resize.

    The resize case passes the function by reference to addEventListener and is
    asserted in test_high_dpi_and_resize_are_handled; the other two are direct
    calls. A stale skeleton is worse than none - it claims a hand is being
    tracked when nothing is.
    """
    html = INDEX.read_text(encoding="utf-8")
    assert "function clearLandmarks(" in html
    assert html.count("clearLandmarks();") == 2, "expected stopCamera and resetBuffer"
    # Anchored to their surroundings, so a future edit cannot drop one silently.
    assert "showLoading(false);\n    // No camera, no hand" in html
    assert "framesSent = 0;\n    clearLandmarks();" in html


def test_high_dpi_and_resize_are_handled():
    """A canvas whose backing store never matches its CSS box draws blurry, and
    stretches after a resize - both of which look like bad tracking."""
    html = INDEX.read_text(encoding="utf-8")
    assert "devicePixelRatio" in html
    assert "landmarkCanvas.width = Math.round(cssW * dpr)" in html
    assert "addEventListener('resize', clearLandmarks)" in html


def test_the_countdown_carries_the_overlay_but_idle_does_not():
    """The countdown is when you are framing your hand, so it needs the skeleton.

    READY deliberately does not run detection: a page can sit in that state for
    minutes, and detecting there would keep a 0.1-CPU instance busy for as long
    as the tab is open. The countdown is bounded to ~3 seconds per trial, so the
    cost is paid only while a trial is actually starting.
    """
    backend = (PROJECT_ROOT / "src" / "web_demo" / "backend.py").read_text(encoding="utf-8")

    countdown = backend[backend.index('elif state.word_state == "COUNTDOWN":'):]
    countdown = countdown[:countdown.index('elif state.word_state == "ANALYZING":')]
    assert '"landmarks": cd_landmarks' in countdown
    assert "_detect_only(state," in countdown

    ready = backend[backend.index('if state.word_state == "READY":'):]
    ready = ready[:ready.index('elif state.word_state == "COUNTDOWN":')]
    assert "_detect_only" not in ready, (
        "detecting while idle would peg the instance for as long as the tab is open"
    )


def test_a_failed_detection_does_not_become_an_error_message():
    """_detect_only backs the overlay only. If it throws, the reply it rides on
    is still valid, so a drawing failure must not surface as an error."""
    backend = (PROJECT_ROOT / "src" / "web_demo" / "backend.py").read_text(encoding="utf-8")
    block = backend[backend.index("def _detect_only("):]
    block = block[:block.index("async def _record_word_frame(")]
    assert "except Exception:" in block
    assert "return None" in block
    assert "send_text" not in block, "this helper must never talk to the client"
