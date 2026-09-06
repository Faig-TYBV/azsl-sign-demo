"""
Production inference pipeline for Experiment 2 (GRU + mean_max pooling).

Takes a single .mp4 video, extracts hand-landmark features using the EXACT
same preprocessing as dataset creation, and predicts one of the 200 AzSL
word classes with the saved checkpoint.

Reused preprocessing (no duplicated logic):
    src/features/extract_landmarks.py : create_hand_landmarker,
                                        extract_video_landmarks,
                                        normalize_frame (wrist-relative +
                                        scale norm, L/R slots, zero fill)
    src/features/preprocess_sequence.py: preprocess_sequence (26-frame
                                        resample / end-pad + validity mask)

Usage:
    .venv\\Scripts\\python.exe -m src.inference.predict --video data/example.mp4

API:
    result = predict_video(video_path, checkpoint_path=None, top_k=5)

Note: probabilities are MODEL CONFIDENCE (softmax outputs), not guaranteed
correctness.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Make project-root imports work both as `python -m src.inference.predict`
# and when run from other working directories.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.extract_landmarks import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    create_hand_landmarker,
    extract_video_landmarks,
    normalize_frame,
)
from src.features.preprocess_sequence import (  # noqa: E402
    TARGET_SEQ_LEN,
    preprocess_sequence,
)
from src.models.gru_classifier import GRUClassifier  # noqa: E402

from src.inference.ambiguity_gate import is_ambiguous_prediction

DEFAULT_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs/vocabulary_24_cap50/checkpoints/gru_24_cap50_best.pt"
)

EXPECTED_CONFIG = {
    "input_size": 126,
    "hidden_size": 128,
    "num_layers": 2,
    "num_classes": 24,
    "dropout": 0.3,
    "bidirectional": False,
    "pooling": "mean_max",
}


class CheckpointVerificationError(RuntimeError):
    """Raised when a checkpoint is missing or does not match Exp2 config."""


def get_device() -> torch.device:
    """CUDA if available (RTX 3060 Laptop), otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def verify_checkpoint(checkpoint_path: Path) -> dict:
    """
    Load the checkpoint and strictly verify it matches the Exp2 setup.

    Returns the raw checkpoint dict on success; raises
    CheckpointVerificationError otherwise.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise CheckpointVerificationError(f"Checkpoint not found: {checkpoint_path}")

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise CheckpointVerificationError(f"Checkpoint failed to load: {exc}") from exc

    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise CheckpointVerificationError(
            "Checkpoint does not contain 'model_state_dict'."
        )

    cfg = ckpt.get("model_config")
    if cfg is None:
        raise CheckpointVerificationError("Checkpoint has no 'model_config'.")

    expected_keys = {
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.3,
        "bidirectional": False,
        "pooling": "mean_max",
    }
    mismatches = {
        key: {"expected": expected, "found": cfg.get(key)}
        for key, expected in expected_keys.items()
        if cfg.get(key) != expected
    }
    if cfg.get("input_size") not in (126, 280):
        mismatches["input_size"] = {"expected": "126 or 280", "found": cfg.get("input_size")}
    if cfg.get("num_classes") not in (20, 24, 25, 200):
        mismatches["num_classes"] = {"expected": "20, 24, 25, or 200", "found": cfg.get("num_classes")}
    if mismatches:
        raise CheckpointVerificationError(
            f"Checkpoint config mismatch: {mismatches}"
        )

    class_to_idx = ckpt.get("class_to_idx")
    if not isinstance(class_to_idx, dict) or len(class_to_idx) == 0:
        raise CheckpointVerificationError("Checkpoint has no 'class_to_idx' mapping.")
    return ckpt


def load_model(
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    device: torch.device | None = None,
) -> tuple[GRUClassifier, dict[str, int], list[str], torch.device]:
    """Verify the checkpoint, build the model, load weights, set eval mode."""
    ckpt = verify_checkpoint(Path(checkpoint_path))
    cfg = dict(ckpt["model_config"])
    class_to_idx: dict[str, int] = ckpt["class_to_idx"]
    idx_to_class = [None] * len(class_to_idx)
    for name, idx in class_to_idx.items():
        idx_to_class[idx] = name

    device = device or get_device()
    model = GRUClassifier(**cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_to_idx, idx_to_class, device


def preprocess_video(video_path: str | Path) -> dict:
    """
    Run the EXACT dataset-creation preprocessing on one video.

    video -> MediaPipe landmarks -> normalize_frame (wrist-relative, scale
    norm, L/R slots, zero-filled misses) -> preprocess_sequence
    (26-frame resample/pad + mask).
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    landmarker = create_hand_landmarker(DEFAULT_MODEL_PATH)
    try:
        extraction = extract_video_landmarks(video_path, landmarker)
    finally:
        landmarker.close()

    if extraction.error:
        raise RuntimeError(f"Extraction failed for {video_path}: {extraction.error}")
    if not extraction.raw_frames:
        raise RuntimeError(f"No frames could be read from {video_path}.")

    frame_feats = np.stack(
        [normalize_frame(fr) for fr in extraction.raw_frames], axis=0
    ).astype(np.float32)                                   # (n, 126)
    frame_valid = np.array(
        [fr.num_hands > 0 for fr in extraction.raw_frames], dtype=bool
    )                                                      # (n,)

    features, mask = preprocess_sequence(frame_feats, frame_valid, TARGET_SEQ_LEN)

    return {
        "features": features,                              # (26, 126) float32
        "mask": mask,                                      # (26,) float32
        "num_original_frames": int(frame_feats.shape[0]),
        "original_fps": float(extraction.original_fps),
        "valid_frames": int(frame_valid.sum()),
        "valid_ratio": float(frame_valid.mean()) if len(frame_valid) else 0.0,
        "frames_no_hand": int(extraction.frames_no_hand),
        "max_hands_seen": int(extraction.max_hands_seen),
    }


@torch.no_grad()
def predict_video(
    video_path: str | Path,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    top_k: int = 5,
) -> dict:
    """
    Full pipeline: .mp4 -> [26,126] features -> Exp2 GRU -> softmax -> top-k.

    Returned probabilities are model confidence (softmax), not guaranteed
    correctness — low-confidence predictions are still returned as-is.
    """
    pre = preprocess_video(video_path)
    model, class_to_idx, idx_to_class, device = load_model(Path(checkpoint_path))

    x = torch.from_numpy(pre["features"]).unsqueeze(0).to(device)   # [1,26,126]
    logits = model(x)                                               # [1,200]
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    top_k = max(1, min(int(top_k), len(idx_to_class)))
    order = np.argsort(-probs)[:top_k]

    if len(order) >= 2:
        top1_c = idx_to_class[int(order[0])]
        top2_c = idx_to_class[int(order[1])]
        top1_p = float(probs[order[0]])
        top2_p = float(probs[order[1]])
        is_ambig = is_ambiguous_prediction(top1_c, top2_c, top1_p, top2_p)
    else:
        is_ambig = False

    return {
        "video": str(video_path),
        "predicted_class": idx_to_class[int(order[0])],
        "confidence": float(probs[order[0]]),
        "is_ambiguous": is_ambig,
        "top_k": [
            {"class": idx_to_class[int(i)], "probability": float(probs[i])}
            for i in order
        ],
        "class_to_idx_size": len(class_to_idx),
        "device": str(device),
        "preprocessing": {
            "num_original_frames": pre["num_original_frames"],
            "original_fps": pre["original_fps"],
            "valid_frames": pre["valid_frames"],
            "valid_ratio": round(pre["valid_ratio"], 4),
            "feature_shape": list(pre["features"].shape),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Exp2 GRU video inference.")
    parser.add_argument("--video", required=True, help="Path to .mp4 video")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT_PATH))
    args = parser.parse_args()

    try:
        result = predict_video(args.video, args.checkpoint, args.top_k)
    except (CheckpointVerificationError, FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Video      : {result['video']}")
    print(f"Prediction : {result['predicted_class']}")
    print(f"Confidence : {result['confidence']:.4f}  (model confidence)")
    print(f"\nTop-{args.top_k}:")
    for rank, item in enumerate(result["top_k"], start=1):
        print(f"{rank}. {item['class']:30s} p={item['probability']:.4f}")
    pre = result["preprocessing"]
    print("\nPreprocessing:")
    print(f"Frames       : {pre['num_original_frames']}")
    print(f"FPS          : {pre['original_fps']:.2f}")
    print(f"Valid frames : {pre['valid_frames']}")
    print(f"Feature shape: ({pre['feature_shape'][0]}, {pre['feature_shape'][1]})")
    print(f"Device       : {result['device']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
