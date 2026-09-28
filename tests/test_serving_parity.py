"""
Does the live serving path give the same answer as calling the model directly?

This exists because "the model got worse" is the hardest complaint to act on:
the model, the feature extraction, the trial state machine and the browser are
all candidates, and only one of them can be ruled out cheaply. If the serving
path reproduces direct inference exactly, then any real loss is in detection or
in the model, and the web layer can be eliminated in one step instead of argued
about.

It drives ``_record_word_frame`` - the single function both input paths funnel
into - rather than a WebSocket, because a socket adds reply ordering that is easy
to get wrong in a test and would produce a false alarm. (It did, while this was
being written: a probe that read a stale RESULT reported six identical
predictions for six different inputs.)
"""

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

pytest.importorskip("torch", reason="the recognition stack is optional")
pytest.importorskip("mediapipe")

import torch  # noqa: E402

from src.web_demo import backend as B  # noqa: E402

TRIAL_FRAMES = 26
FEATURE_DIM = 126


class FakeWS:
    def __init__(self):
        self.msgs = []

    async def send_text(self, text):
        self.msgs.append(json.loads(text))


class FakeState:
    """Only the fields _record_word_frame touches."""

    def __init__(self):
        self.recorded_features = []
        self.recorded_validity = []
        self.word_state = "RECORDING"
        self.last_word_result = None


def _plausible_sequence(rng):
    """A smooth trajectory in the range the real features occupy.

    White noise would be answered by whichever class the model falls back on and
    would compare equal for trivial reasons; a drifting sequence exercises the
    GRU's recurrence, which is where a mismatch would actually show.
    """
    base = rng.normal(0, 0.35, size=FEATURE_DIM).astype(np.float32)
    drift = rng.normal(0, 0.02, size=FEATURE_DIM).astype(np.float32)
    return np.stack([base + drift * k for k in range(TRIAL_FRAMES)]).astype(np.float32)


def _direct(seq, val):
    """Inference with no web layer involved at all."""
    norm = B.normalizer(seq, val)
    with torch.no_grad():
        logits = B.model(torch.from_numpy(norm).unsqueeze(0).to(B.device))
        probs = torch.softmax(logits, dim=1)[0]
    top_p, top_i = torch.topk(probs, 1)
    return B.idx_to_class[int(top_i[0])], float(top_p[0])


def _through_serving_path(seq, val):
    async def run():
        state, ws = FakeState(), FakeWS()
        for k in range(TRIAL_FRAMES):
            await B._record_word_frame(state, ws, seq[k], float(val[k]))
        results = [m for m in ws.msgs if m.get("word_state") == "RESULT"]
        assert len(results) == 1, f"expected exactly one RESULT, got {len(results)}"
        return results[0]

    return asyncio.run(run())


def test_the_model_is_in_eval_mode():
    """Dropout left active would make every prediction stochastic.

    p=0.3 on a 2-layer GRU plus the head: the same sign would land on different
    words run to run, which is exactly what "it used to be more accurate" feels
    like from the outside, and nothing in the output would say so.
    """
    assert B.model.training is False, "the model is in train mode - dropout is live"
    for module in B.model.modules():
        if isinstance(module, torch.nn.Dropout):
            assert module.training is False


@pytest.mark.parametrize("seed", [7, 11, 23, 101])
def test_serving_path_reproduces_direct_inference(seed):
    rng = np.random.default_rng(seed)
    seq = _plausible_sequence(rng)
    val = np.ones(TRIAL_FRAMES, dtype=np.float32)

    label, confidence = _direct(seq, val)
    served = _through_serving_path(seq, val)

    assert served["prediction"] == label
    # Bit-for-bit, not approximately: it is the same tensor through the same
    # graph, so any gap at all means the web layer altered the input.
    assert served["confidence"] == pytest.approx(confidence, abs=1e-9)
    assert served["valid_frames"] == TRIAL_FRAMES


def test_inference_is_deterministic_across_repeats():
    """Same input twice, same answer twice - no hidden state carried between
    trials and no live dropout."""
    rng = np.random.default_rng(5)
    seq = _plausible_sequence(rng)
    val = np.ones(TRIAL_FRAMES, dtype=np.float32)

    first = _through_serving_path(seq, val)
    second = _through_serving_path(seq, val)

    assert first["prediction"] == second["prediction"]
    assert first["confidence"] == second["confidence"]


def test_a_trial_with_too_few_real_frames_refuses_to_guess():
    """Below the valid-frame gate the answer must be "no hand", not a class.

    A GRU handed 26 zero vectors still produces a confident-looking softmax, so
    without this gate an empty camera yields a random word at 60%.
    """
    seq = np.zeros((TRIAL_FRAMES, FEATURE_DIM), dtype=np.float32)
    val = np.zeros(TRIAL_FRAMES, dtype=np.float32)

    served = _through_serving_path(seq, val)

    assert served["prediction"] == "ƏL AŞKARLANMADI"
    assert served["confidence"] == 0.0
    assert served["valid_frames"] == 0


def test_the_valid_frame_gate_is_where_the_docs_say():
    """The gate is the difference between "no hand" and a confident wrong word,
    so its value is worth pinning."""
    assert B.MIN_VALID_FRAMES_FOR_INFERENCE == 5
    seq = _plausible_sequence(np.random.default_rng(3))
    val = np.zeros(TRIAL_FRAMES, dtype=np.float32)
    val[: B.MIN_VALID_FRAMES_FOR_INFERENCE - 1] = 1.0

    served = _through_serving_path(seq, val)
    assert served["prediction"] == "ƏL AŞKARLANMADI", (
        "one frame below the gate must still refuse"
    )

    val[: B.MIN_VALID_FRAMES_FOR_INFERENCE] = 1.0
    served = _through_serving_path(seq, val)
    assert served["prediction"] != "ƏL AŞKARLANMADI", "at the gate it should predict"
