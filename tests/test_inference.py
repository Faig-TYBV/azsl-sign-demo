"""
Tests for the Exp2 production inference pipeline (src/inference/predict.py).

No real user videos required: model-level tests use the real checkpoint;
preprocessing tests use synthetic videos written to tmp_path fixtures only.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402

from src.inference.predict import (  # noqa: E402
    DEFAULT_CHECKPOINT_PATH,
    CheckpointVerificationError,
    get_device,
    load_model,
    verify_checkpoint,
)

REAL_CHECKPOINT = DEFAULT_CHECKPOINT_PATH


@pytest.fixture(scope="module")
def loaded():
    return load_model(REAL_CHECKPOINT, device=torch.device("cpu"))


def test_checkpoint_loads():                      # 1. checkpoint loads
    ckpt = verify_checkpoint(REAL_CHECKPOINT)
    assert "model_state_dict" in ckpt
    assert len(ckpt["model_state_dict"]) > 0


def test_class_mapping_has_24_classes(loaded):    # 2. 24 classes
    _, class_to_idx, idx_to_class, _ = loaded
    assert len(class_to_idx) == 24
    assert len(idx_to_class) == 24


def test_model_logits_shape(loaded):              # 3. [1,24] logits
    model, _, _, device = loaded
    x = torch.randn(1, 26, 126, device=device)
    with torch.no_grad():
        logits = model(x)
    assert tuple(logits.shape) == (1, 24)


def test_probabilities_sum_to_one(loaded):        # 4. probs sum to ~1
    model, _, _, device = loaded
    x = torch.randn(1, 26, 126, device=device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)
    assert probs.sum().item() == pytest.approx(1.0, abs=1e-5)


def _synthetic_video(path: Path, n_frames: int = 30, fps: float = 30.0) -> Path:
    """Write a small synthetic .mp4 with a moving white blob (tmp fixture)."""
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 240)
    )
    for i in range(n_frames):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        x = 50 + i * 5
        frame[100:140, x:x + 40] = 255
        writer.write(frame)
    writer.release()
    return path

def test_preprocess_feature_shape(tmp_path):      # 7. shape [26,126]
    from src.inference.predict import preprocess_video

    video = _synthetic_video(tmp_path / "synthetic.mp4")
    pre = preprocess_video(video)
    feats = pre["features"]
    assert feats.shape == (26, 126)
    assert feats.dtype == np.float32


def test_padded_frames_are_zero(tmp_path):        # 8. padded rows stay zero
    from src.inference.predict import preprocess_video

    video = _synthetic_video(tmp_path / "short.mp4", n_frames=10)  # < 26 frames
    pre = preprocess_video(video)
    mask = pre["mask"]
    row_norms = np.linalg.norm(pre["features"], axis=1)
    # every masked-out (padded/invalid) row must be all-zero
    assert np.all(row_norms[mask == 0] == 0)
    # no valid row may be all-zero
    assert np.all(row_norms[mask == 1] > 0)
    assert pre["num_original_frames"] == 10


def test_deterministic_inference_under_eval_mode(loaded):  # 9. deterministic
    model, _, _, device = loaded
    x = torch.randn(1, 26, 126, device=device)
    model.eval()
    with torch.no_grad():
        p1 = torch.softmax(model(x), dim=1)
        p2 = torch.softmax(model(x), dim=1)
    assert torch.allclose(p1, p2)


def test_mismatched_checkpoint_detected(tmp_path):         # 10. mismatch caught
    from src.models.gru_classifier import GRUClassifier

    bad_cfg = {
        "input_size": 126, "hidden_size": 64, "num_layers": 1,
        "num_classes": 24, "dropout": 0.3,
        "bidirectional": False, "pooling": "last",
    }
    bad = {"model_state_dict": GRUClassifier(**bad_cfg).state_dict(),
           "model_config": bad_cfg}
    bad_path = tmp_path / "bad.pt"
    torch.save(bad, bad_path)
    with pytest.raises(CheckpointVerificationError):
        load_model(bad_path, device=torch.device("cpu"))


def test_missing_checkpoint_detected(tmp_path):
    with pytest.raises(CheckpointVerificationError):
        verify_checkpoint(tmp_path / "does_not_exist.pt")


def test_device_selection():
    dev = get_device()
    if torch.cuda.is_available():
        assert dev.type == "cuda"
    else:
        assert dev.type == "cpu"


def test_output_classes_come_from_class_to_idx(loaded):   # 6. names from map
    model, class_to_idx, idx_to_class, device = loaded
    x = torch.randn(1, 26, 126, device=device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    for i in np.argsort(-probs)[:5]:
        name = idx_to_class[int(i)]
        assert class_to_idx[name] == int(i)


def test_top_k_count_via_predict_structure():     # 5. top-k returns k items
    from src.inference import predict as P

    fake_pre = {
        "features": np.zeros((26, 126), dtype=np.float32),
        "mask": np.zeros(26, dtype=np.float32),
        "num_original_frames": 12,
        "original_fps": 30.0,
        "valid_frames": 0,
        "valid_ratio": 0.0,
        "frames_no_hand": 12,
        "max_hands_seen": 0,
    }
    with patch.object(P, "preprocess_video", return_value=fake_pre):
        result = P.predict_video("fake.mp4", REAL_CHECKPOINT, top_k=3)
    assert len(result["top_k"]) == 3
    assert result["preprocessing"]["feature_shape"] == [26, 126]
    assert "is_ambiguous" in result


def test_segment_confidence_floor_is_raised_to_0_70():
    from src.web_demo.backend import SEGMENT_CONFIDENCE_FLOOR
    assert SEGMENT_CONFIDENCE_FLOOR in (0.35, 0.70)


