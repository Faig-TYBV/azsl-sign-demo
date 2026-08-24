"""
Train-only feature normalization for AzSLD sequences.

Fits per-dimension mean/std from VALID training frames only
(mask == 1). Padded / no-detection frames are zero-filled by the
preprocessing pipeline and must remain exactly zero after
normalization — they are excluded from statistics and left untouched
during transformation.

Validation/test data are transformed with the SAME training-derived
statistics; their values never influence the statistics.
"""

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np


def fit_normalization(
    samples,  # Sequence[FeatureSample]
    epsilon: float = 1e-6,
) -> Dict[str, list]:
    """Fit per-dim mean/std over valid training frames only.

    Returns dict with 'mean', 'std' (lists of 126 floats) and metadata.
    Original .npz files are only read, never written.
    """
    sums = None
    sq_sums = None
    n_valid = 0
    for s in samples:
        with np.load(s.path) as z:
            feats = z["features"]
            mask = z["mask"].astype(bool)
        valid = feats[mask]  # (n_valid_frames, 126)
        if sums is None:
            sums = valid.sum(axis=0)
            sq_sums = (valid.astype(np.float64) ** 2).sum(axis=0)
        else:
            sums += valid.sum(axis=0)
            sq_sums += (valid.astype(np.float64) ** 2).sum(axis=0)
        n_valid += valid.shape[0]

    if n_valid == 0:
        raise ValueError("No valid frames found in training samples")

    mean = (sums / n_valid).astype(np.float32)
    var = np.maximum(sq_sums / n_valid - mean.astype(np.float64) ** 2, 0.0)
    std = np.sqrt(var).astype(np.float32)
    std_safe = np.maximum(std, epsilon)

    return {
        "mean": mean.tolist(),
        "std": std_safe.tolist(),
        "raw_std": std.tolist(),
        "epsilon": epsilon,
        "n_valid_frames": int(n_valid),
        "n_samples": len(samples),
    }


class FeatureNormalizer:
    """Applies (X - mean) / std_safe to valid frames; invalid stay zero.

    Deterministic given fitted statistics; inputs are never mutated.
    """

    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    @classmethod
    def from_stats(cls, stats: Dict) -> "FeatureNormalizer":
        return cls(stats["mean"], stats["std"])

    def __call__(self, features: np.ndarray, mask: np.ndarray) -> np.ndarray:
        out = np.array(features, dtype=np.float32, copy=True)
        valid = mask.astype(bool)
        out[valid] = (out[valid] - self.mean) / self.std
        # Invalid frames remain exactly as stored (zero-filled); enforce zero.
        out[~valid] = 0.0
        return out

    def save(self, path: Path, extra: Optional[dict] = None) -> None:
        payload = {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            **(extra or {}),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> "FeatureNormalizer":
        with open(path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        return cls(stats["mean"], stats["std"])
