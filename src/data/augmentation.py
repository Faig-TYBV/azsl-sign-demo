"""
Training-time temporal augmentation for AzSLD feature sequences.

Operates dynamically on in-memory tensors — stored .npz files are never
modified. Applied ONLY to the training split by the caller.

Representation semantics (from preprocess_sequence.py):
  * features (26, 126) float32, wrist-scale normalized landmarks
  * invalid / padded / no-detection frames are ZERO-FILLED (mask == 0.0)
=> the neutral representation is zeros, and augmentation must not corrupt
   padding: all operations act only on frames where mask == 1.
"""

import numpy as np

from typing import Optional, Tuple


class TemporalAugmentation:
    """Mild temporal augmentation pipeline for (26, 126) sequences.

    Operations (applied in order, each independently randomized):
      1. Gaussian landmark noise on VALID frames only (sigma = noise_std)
      2. Temporal frame masking: zero out randomly chosen valid frames
         (expected fraction = temporal_mask_prob)
      3. Temporal dropout: additionally zero out random valid frames
         (probability temporal_dropout_prob per frame)
    """

    def __init__(
        self,
        noise_std: float = 0.01,
        temporal_mask_prob: float = 0.10,
        temporal_dropout_prob: float = 0.05,
        seed: Optional[int] = None,
    ):
        if min(noise_std, temporal_mask_prob, temporal_dropout_prob) < 0:
            raise ValueError("Augmentation parameters must be non-negative")
        self.noise_std = float(noise_std)
        self.temporal_mask_prob = float(temporal_mask_prob)
        self.temporal_dropout_prob = float(temporal_dropout_prob)
        self.rng = np.random.default_rng(seed)

    def __call__(self, features: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Return an augmented COPY of ``features``; inputs are untouched."""
        feats = np.array(features, dtype=np.float32, copy=True)  # (26,126)
        T = feats.shape[0]
        valid = mask.astype(bool) if mask is not None else np.ones(T, bool)

        # 1. Gaussian landmark noise (valid frames only)
        if self.noise_std > 0:
            noise = self.rng.normal(0.0, self.noise_std, feats.shape).astype(np.float32)
            feats[valid] += noise[valid]

        # 2. Temporal frame masking: choose ~prob*T valid timesteps to zero.
        # Never mask ALL valid frames — always leave at least one intact.
        if self.temporal_mask_prob > 0:
            candidates = np.flatnonzero(valid)
            n_mask = min(int(round(self.temporal_mask_prob * T)),
                         max(0, len(candidates) - 1))
            if n_mask > 0:
                chosen = self.rng.choice(candidates, size=n_mask, replace=False)
                feats[chosen] = 0.0
                valid = valid.copy()
                valid[chosen] = False

        # 3. Temporal dropout: per-valid-frame Bernoulli zeroing.
        # Again, guarantee at least one surviving valid frame.
        if self.temporal_dropout_prob > 0:
            survivors = np.flatnonzero(valid)
            drop = (self.rng.random(T) < self.temporal_dropout_prob) & valid
            if len(survivors) >= 1:
                keep = self.rng.choice(survivors)
                drop[keep] = False
            feats[drop] = 0.0

        return feats
