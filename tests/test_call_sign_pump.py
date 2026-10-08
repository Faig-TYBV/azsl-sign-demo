"""
In-call sign recognition must actually reach the classifier.

Two bugs kept fingerspelling in calls (the default sign mode) from ever
producing a letter, and neither was visible as an error:

  * setConvMode() deliberately skips the recognition socket for on-device
    fingerspelling, but the frame pump bailed out on every tick when there was
    no open socket. Not one frame was classified.
  * setConvMode() also reset the stabilizer to a plain object, so the code that
    creates the real AlphabetStabilizer on demand skipped it and then called
    .update() on an object with no such method.

These run the page's own functions (pulled out of friends.html) under node
against the real azsl_alphabet.js / azsl_features.js, with only the DOM, the
camera and MediaPipe stubbed.
"""

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND = PROJECT_ROOT / "src" / "web_demo" / "frontend"
FRIENDS = FRONTEND / "friends.html"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is required")


def extract(html: str, name: str) -> str:
    """One top-level function's source, by name."""
    m = re.search(rf"^(async )?function {name}\(", html, flags=re.M)
    assert m, f"{name} not found in friends.html"
    end = html.index("\n}\n", m.start())
    return html[m.start():end + 3]


def run(scenario: str) -> dict:
    html = FRIENDS.read_text(encoding="utf-8")
    funcs = "\n".join(
        extract(html, n)
        for n in ("localAlphabetActive", "startFramePump", "stopFramePump",
                  "classifyAlphabetLocally")
    )
    model = (FRONTEND / "models" / "azsl_hierarchical_model.json").as_posix()
    script = f"""
require({json.dumps(str(FRONTEND / "js" / "azsl_alphabet.js"))});
require({json.dumps(str(FRONTEND / "js" / "azsl_features.js"))});
const fs = require('fs');
const window = globalThis;
window.AzslAlphabet.setAzslModel(JSON.parse(fs.readFileSync({json.dumps(model)}, 'utf8')));

const WebSocket = {{ OPEN: 1 }};
const CONV_CAPTURE_W = 320, CONV_CAPTURE_H = 240, CONV_FRAME_MS = 100;
const sent = [], classified = [], errors = [];
let tick = null;
global.setInterval = (fn) => {{ tick = fn; return 1; }};
global.clearInterval = () => {{ tick = null; }};
const document = {{ createElement: () => ({{ getContext: () => ({{ drawImage() {{}} }}),
                                            toDataURL: () => 'data:,x' }}) }};
const video = {{ videoWidth: 640 }};
const $ = (id) => (id === 'local-video' ? video : null);

// One right hand, roughly open.
const hand = [];
for (let i = 0; i < 21; i++) hand.push([0.5 + 0.01 * (i % 5), 0.6 - 0.015 * i, 0]);
const local = {{ ready: true, alphabetModel: {{}}, stab: null }};
const conv = {{ signMode: 'alphabet', wordState: 'READY', recogWs: null, frameInFlight: false,
               frameTimer: null }};
function detectLocal() {{ return [{{ label: 'Left', landmarks: hand }}]; }}
function drawHandOverlay() {{}}
function recogSend(p) {{ sent.push(p.mode || p.type); }}
function callLog() {{}}
function toast() {{}}
function applyLocalAlphabet(out) {{ classified.push(out.raw); }}

{funcs}

{scenario}

function frames(n) {{
  for (let i = 0; i < n; i++) {{
    try {{ tick && tick(); }} catch (e) {{ errors.push(e.message); }}
  }}
}}
main();
process.stdout.write(JSON.stringify({{ sent, classified: classified.length, errors }}));
"""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "pump.cjs"
        path.write_text(script, encoding="utf-8")
        out = subprocess.run(["node", str(path)], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def test_on_device_fingerspelling_runs_without_a_socket():
    result = run("""
function main() {
  local.stab = null;          // what setConvMode now does
  startFramePump();
  frames(5);
}""")
    assert result["errors"] == []
    assert result["classified"] == 5, "frames never reached the alphabet classifier"
    assert result["sent"] == [], "on-device fingerspelling must not use the server"


def test_word_mode_still_needs_the_socket():
    result = run("""
function main() {
  conv.signMode = 'word';
  conv.wordState = 'RECORDING';
  startFramePump();
  frames(3);                  // no socket: nothing can be sent
  conv.recogWs = { readyState: 1 };
  frames(1);
}""")
    assert result["errors"] == []
    assert result["sent"] == ["word"]


def test_set_conv_mode_does_not_plant_a_stabilizer_without_update():
    html = FRIENDS.read_text(encoding="utf-8")
    block = extract(html, "setConvMode")
    assert "local.stab = {" not in block, (
        "a plain object here makes classifyAlphabetLocally skip creating the "
        "real AlphabetStabilizer and then call .update() on it"
    )
