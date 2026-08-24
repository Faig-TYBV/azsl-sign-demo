"""
Temporal delta (velocity) features for AzSLD sequences.

Computes frame-to-frame deltas D[t] = X[t] - X[t-1] with strict
mask/padding semantics:

  * first frame:                D[0] = 0
  * invalid/padded frame:       D[t] = 0
  * invalid -> valid boundary:  D[first valid] = 0
  * valid -> invalid boundary:  D at the invalid frame = 0

Deltas are computed strictly WITHIN one sequence — no information ever
crosses video boundaries. The original features are never modified;
the combined input is concat([X, D], axis=-1) -> [26, 252].

No normalization/scaling is applied to deltas (controlled experiment).
"""

import numpy as np


def compute_temporal_deltas(features: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Compute masked temporal deltas for one sequence.

    Parameters
    ----------
    features : (T, F) float32
    mask     : (T,) float32/bool — 1.0 valid frame, 0.0 padded/no-detection

    Returns
    -------
    (T, F) float32 deltas; inputs are not mutated.
    """
    feats = np.asarray(features)
    T = feats.shape[0]
    deltas = np.zeros_like(feats, dtype=np.float32)

    if T == 0:
        return deltas

    valid = mask.astype(bool)
    # Indices t where both t and t-1 are valid -> normal delta.
    both_valid = valid[1:] & valid[:-1]
    idx = np.flatnonzero(both_valid) + 1
    deltas[idx] = feats[idx] - feats[idx - 1]
    return deltas.astype(np.float32)


def combine_with_deltas(features: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return concat([X, D], axis=-1): shape (T, 2F). Inputs untouched."""
    d = compute_temporal_deltas(features, mask)
    return np.concatenate([features, d], axis=-1).astype(np.float32)
