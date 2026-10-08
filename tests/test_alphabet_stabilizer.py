"""
AlphabetStabilizer: when a held letter commits, and what does not reset a hold.

The gate used to be "N consecutive frames above a confidence floor", which
made commit latency depend on the frame rate (the page self-paces to the
backend) and let a single dipped frame throw away a hold in progress. These
tests pin the time-based behaviour, and that the JS port in azsl_alphabet.js
(used on the call page) produces the same output frame for frame.
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.alphabet_classifier import AlphabetStabilizer  # noqa: E402

JS_FILE = PROJECT_ROOT / "src" / "web_demo" / "frontend" / "js" / "azsl_alphabet.js"


def feed(stab, frames):
    """frames: (t_ms, letter, conf, hand_present, wrist). Returns all results."""
    out = []
    for t, letter, conf, present, wrist in frames:
        out.append(stab.update(letter, conf, hand_present=present, wrist=wrist, now_ms=t))
    return out


def hold(letter, conf, start, fps, duration_ms, wrist=(0.5, 0.5)):
    step = 1000.0 / fps
    n = int(duration_ms / step) + 1
    return [(start + i * step, letter, conf, True, wrist) for i in range(n)]


def commit_time(results, frames):
    for r, f in zip(results, frames):
        if r["just_accepted"]:
            return f[0]
    return None


@pytest.mark.parametrize("fps", [4, 8, 15, 30])
def test_commit_latency_does_not_depend_on_frame_rate(fps):
    frames = hold("A", 0.9, 0, fps, 1200)
    stab = AlphabetStabilizer()
    t = commit_time(feed(stab, frames), frames)
    assert t is not None
    # Hold is 450 ms; at 4 fps the next frame after that lands at 500 ms,
    # and the 3-frame floor also lands there.
    assert 450 <= t <= 500
    assert stab.spelled_word == "A"


def test_single_low_confidence_frame_does_not_restart_the_hold():
    frames = hold("B", 0.9, 0, 10, 600)
    frames[3] = (frames[3][0], "B", 0.4, True, (0.5, 0.5))
    stab = AlphabetStabilizer()
    t = commit_time(feed(stab, frames), frames)
    assert t is not None and t <= 500


def test_single_frame_flicker_to_another_letter_is_ignored():
    frames = hold("C", 0.9, 0, 10, 600)
    frames[2] = (frames[2][0], "O", 0.95, True, (0.5, 0.5))
    stab = AlphabetStabilizer()
    results = feed(stab, frames)
    assert commit_time(results, frames) is not None
    assert stab.spelled_word == "C"


def test_one_frame_hand_dropout_is_forgiven():
    frames = hold("D", 0.9, 0, 10, 600)
    frames[3] = (frames[3][0], None, 0.0, False, None)
    stab = AlphabetStabilizer()
    assert commit_time(feed(stab, frames), frames) is not None


def test_low_confidence_pose_never_commits():
    frames = hold("E", 0.65, 0, 15, 2000)  # above the frame floor, below commit mean
    stab = AlphabetStabilizer()
    feed(stab, frames)
    assert stab.spelled_word == ""


def test_hand_in_transit_does_not_commit():
    # 0.03 image-units per frame at 30 fps = 0.9/s, well above the gate.
    frames = [(i * 33.3, "F", 0.95, True, (0.2 + 0.03 * i, 0.5)) for i in range(30)]
    stab = AlphabetStabilizer()
    results = feed(stab, frames)
    assert stab.spelled_word == ""
    assert all(r["is_moving"] for r in results[1:])


def test_slow_drift_at_low_fps_is_not_motion():
    # 0.03 per frame at 4 fps = 0.12/s: a hand holding a pose, drifting slightly.
    # The old per-frame gate called this "moving" and blocked the letter forever.
    frames = [(i * 250.0, "G", 0.9, True, (0.5 + 0.03 * i, 0.5)) for i in range(5)]
    stab = AlphabetStabilizer()
    feed(stab, frames)
    assert stab.spelled_word == "G"


def test_held_letter_types_once_and_double_letter_needs_hand_out():
    stab = AlphabetStabilizer()
    feed(stab, hold("L", 0.9, 0, 15, 3000))
    assert stab.spelled_word == "L"
    # Brief dropout: still latched.
    feed(stab, [(3100, None, 0.0, False, None)])
    feed(stab, hold("L", 0.9, 3150, 15, 1000))
    assert stab.spelled_word == "L"
    # Hand properly out of frame, then the same letter again.
    feed(stab, [(4300, None, 0.0, False, None), (4700, None, 0.0, False, None)])
    feed(stab, hold("L", 0.9, 4800, 15, 1000))
    assert stab.spelled_word == "LL"


def test_switching_letters_commits_the_new_one_promptly():
    stab = AlphabetStabilizer()
    a = hold("A", 0.9, 0, 15, 700)
    feed(stab, a)
    b = hold("B", 0.9, 733, 15, 800)
    t = commit_time(feed(stab, b), b)
    assert stab.spelled_word == "AB"
    assert t - 733 <= 500


def test_progress_reported_while_holding():
    stab = AlphabetStabilizer()
    results = feed(stab, hold("M", 0.9, 0, 10, 300))
    ratios = [r["candidate_ratio"] for r in results]
    assert ratios == sorted(ratios)
    assert 0 < ratios[-1] < 1
    assert results[-1]["stable_candidate"] == "M"


# --- JS parity ----------------------------------------------------------------

def run_js(frames):
    harness = f"""
require({json.dumps(str(JS_FILE))});
const A = globalThis.AzslAlphabet;
const frames = JSON.parse(require('fs').readFileSync(process.argv[2], 'utf8'));
const st = new A.AlphabetStabilizer();
const out = frames.map(([t, letter, conf, present, wrist]) => {{
  const r = st.update({{ letter, confidence: conf, handPresent: present, wrist, nowMs: t }});
  return [r.stableCandidate, r.candidateRatio, r.isMoving, r.justAccepted, r.spelledWord];
}});
process.stdout.write(JSON.stringify(out));
"""
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "harness.cjs"
        script.write_text(harness, encoding="utf-8")
        data = Path(tmp) / "frames.json"
        data.write_text(json.dumps(frames), encoding="utf-8")
        out = subprocess.run(
            ["node", str(script), str(data)], capture_output=True, text=True, timeout=60
        )
    if out.returncode != 0:
        raise AssertionError(f"node failed: {out.stderr}")
    return json.loads(out.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_js_port_matches_python():
    import random

    rng = random.Random(7)
    letters = ["A", "B", "C", "SPACE", "DEL", None]
    frames = []
    t = 0.0
    wrist = [0.5, 0.5]
    letter = "A"
    for _ in range(600):
        t += rng.choice([33.3, 66.7, 125.0, 250.0])
        if rng.random() < 0.08:
            letter = rng.choice(letters)
        present = rng.random() > 0.05
        if rng.random() < 0.1:
            wrist[0] += rng.uniform(-0.08, 0.08)
        wrist[0] += rng.uniform(-0.003, 0.003)
        flick = rng.choice(letters) if rng.random() < 0.1 else letter
        conf = rng.uniform(0.4, 1.0)
        frames.append([t, flick if present else None, conf if present else 0.0,
                       present, list(wrist) if present else None])

    py = AlphabetStabilizer()
    expected = []
    for f_t, f_letter, f_conf, f_present, f_wrist in frames:
        r = py.update(f_letter, f_conf, hand_present=f_present,
                      wrist=tuple(f_wrist) if f_wrist else None, now_ms=f_t)
        expected.append([r["stable_candidate"], r["candidate_ratio"], r["is_moving"],
                         r["just_accepted"], r["spelled_word"]])

    got = run_js(frames)
    assert got == expected
    assert any(row[3] for row in expected), "scenario should commit at least one letter"
