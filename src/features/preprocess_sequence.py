"""
Temporal preprocessing of landmark sequences for AzSLD_Words_200.

Target sequence length: 26 frames (matches dataset P50/P75 frame counts).

Strategy:
  - Longer sequences are temporally resampled (uniformly spaced indices over
    the whole gesture) so the full gesture is represented — never just the
    first 26 frames.
  - Shorter sequences are zero-padded at the END, consistently.
  - A per-frame validity mask is produced alongside the features so the
    model / loss can ignore padded frames and no-detection frames.
"""

from __future__ import annotations

import numpy as np

TARGET_SEQ_LEN = 26


def temporal_resample_indices(n_frames: int, target_len: int = TARGET_SEQ_LEN) -> np.ndarray:
    """
    Deterministically map `n_frames` frames onto `target_len` slots.

    - n_frames == 0            -> all indices 0 (caller pads anyway)
    - n_frames < target_len    -> identity indices (pad later)
    - n_frames >= target_len   -> uniformly spaced indices covering the
                                  full gesture: round(i * (n-1)/(T-1))
    """
    if n_frames <= 0:
        return np.zeros(target_len, dtype=int)
    if n_frames < target_len:
        return np.arange(n_frames)
    return np.round(np.linspace(0, n_frames - 1, target_len)).astype(int)


def preprocess_sequence(
    frame_features: np.ndarray,
    frame_valid: np.ndarray,
    target_len: int = TARGET_SEQ_LEN,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resample / pad a sequence of per-frame feature vectors.

    Parameters
    ----------
    frame_features : (n_frames, feature_dim) float32
    frame_valid    : (n_frames,) bool — True if at least one hand detected

    Returns
    -------
    features : (target_len, feature_dim) float32
        Resampled (or end-padded) features.
    mask     : (target_len,) float32
        1.0 for real frames with a hand detection, 0.0 for padded frames
        or frames with no detection.
    """
    n = frame_features.shape[0]
    feature_dim = frame_features.shape[1] if n > 0 else 0

    idx = temporal_resample_indices(n, target_len)

    out = np.zeros((target_len, feature_dim), dtype=np.float32)
    mask = np.zeros(target_len, dtype=np.float32)

    for t, i in enumerate(idx):
        if i < n:
            out[t] = frame_features[i]
            if frame_valid[i]:
                mask[t] = 1.0

    return out, mask