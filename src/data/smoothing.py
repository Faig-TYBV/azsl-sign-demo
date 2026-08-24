"""
Lightweight temporal smoothing of landmark feature sequences.

Reduces frame-level MediaPipe jitter while preserving genuine hand
motion. A conservative CENTERED 3-FRAME MOVING AVERAGE is applied per
feature dimension, strictly WITHIN contiguous valid regions:

    out[t] = mean(X[t-1..t+1] restricted to valid neighbors)

Run-edge slots use their available valid neighbors only (2-frame or
identity), so smoothing never crosses an invalid/padded boundary,
never touches invalid frames (which remain exactly zero), never leaks
between sequences, and is fully deterministic. Inputs are not mutated;
stored .npz files are untouched.
"""

import numpy as np


class TemporalSmoother:
    """Centered 3-frame moving average over contiguous valid runs."""

    def __init__(self, window: int = 3):
        if window != 3:
            raise ValueError("Only the conservative window=3 moving "
                             "average is supported")
        self.window = window

    def __call__(self, features: np.ndarray, mask: np.ndarray) -> np.ndarray:
        feats = np.asarray(features)
        T = feats.shape[0]
        out = np.array(feats, dtype=np.float32, copy=True)
        if T == 0:
            return out

        valid = mask.astype(bool)
        for t in range(T):
            if not valid[t]:
                continue  # invalid frames stay exactly as stored (zero)
            lo = t - 1 if t - 1 >= 0 and valid[t - 1] else t
            hi = t + 1 if t + 1 < T and valid[t + 1] else t
            out[t] = feats[lo:hi + 1].mean(axis=0)

        # Invalid frames explicitly forced to zero.
        out[~valid] = 0.0
        return out.astype(np.float32)
