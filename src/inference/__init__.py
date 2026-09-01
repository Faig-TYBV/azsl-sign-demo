"""Production inference pipeline for the Exp2 GRU model.

Public API lives in src.inference.predict:
    predict_video, preprocess_video, load_model,
    verify_checkpoint, get_device, CheckpointVerificationError

Imported lazily via __getattr__ to avoid the runpy double-import
warning when running `python -m src.inference.predict`.
"""

from typing import Any


def __getattr__(name: str) -> Any:
    if name in {
        "CheckpointVerificationError",
        "get_device",
        "load_model",
        "predict_video",
        "preprocess_video",
        "verify_checkpoint",
    }:
        from src.inference import predict

        return getattr(predict, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
