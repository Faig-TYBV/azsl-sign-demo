"""
Time-aware velocity features for AzSLD sequences.

Raw adjacent-resampled-frame deltas (Exp4) conflate feature change with
the unknown temporal spacing of the resampling grid. Here we divide by
the ACTUAL time interval between the original frames that each
resampled slot represents:

    V[t] = (X[t] - X[t-1]) / dt[t],   dt[t] = (idx[t] - idx[t-1]) / fps

where idx[] are the deterministic resample indices produced by
preprocess_sequence.temporal_resample_indices(), and n_frames / fps come
from data/features/full/extraction_metadata.jsonl.

Mask semantics (identical to delta_features):
  * first slot:            V[0] = 0
  * invalid/padded slot:   V = 0
  * invalid->valid bound.: V = 0 at the first valid slot
  * valid->invalid bound.: V = 0 at the invalid slot

Velocities are computed strictly WITHIN one sequence — no leakage across
videos or splits. Original features are never mutated.
"""

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np


def load_timing_map(metadata_jsonl: Path) -> Dict[str, dict]:
    """Build {npz_relative_posix_path: {"n": frames, "fps": fps}} from the
    append-only extraction metadata. Read-only."""
    timing: Dict[str, dict] = {}
    with open(metadata_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            npz_path = Path(rec["npz_path"]).as_posix()
            timing[npz_path] = {
                "n": int(rec["original_frame_count"]),
                "fps": float(rec["original_fps"]) or 30.0,
            }
    return timing


def resampled_timestamps(n_frames: int, target_len: int = 26) -> np.ndarray:
    """Original frame index selected for each resampled slot (deterministic,
    mirrors preprocess_sequence.temporal_resample_indices)."""
    if n_frames <= 0:
        return np.zeros(target_len, dtype=int)
    if n_frames < target_len:
        idx = np.arange(n_frames)
        # Slots beyond n are padding; map them to the last real index so
        # their dt is defined but they are masked out anyway.
        full = np.full(target_len, n_frames - 1, dtype=int)
        full[: len(idx)] = idx
        return full
    return np.round(np.linspace(0, n_frames - 1, target_len)).astype(int)


def compute_dt(n_frames: int, fps: float, target_len: int = 26) -> np.ndarray:
    """dt[t] in SECONDS between the original frames of slot t and t-1."""
    ts = resampled_timestamps(n_frames, target_len) / float(fps)
    dt = np.empty(len(ts), dtype=np.float32)
    dt[0] = 0.0
    dt[1:] = np.diff(ts)
    # Guard against degenerate zero intervals (duplicate rounded indices).
    dt[1:] = np.maximum(dt[1:], 1e-3)
    return dt


def compute_time_aware_velocity(
    features: np.ndarray,
    mask: np.ndarray,
    dt: np.ndarray,
) -> np.ndarray:
    """V[t] = (X[t]-X[t-1]) / dt[t] with strict mask/boundary semantics.

    Inputs are never mutated. Returns float32 (T, F).
    """
    feats = np.asarray(features)
    T = feats.shape[0]
    vel = np.zeros_like(feats, dtype=np.float32)
    if T == 0:
        return vel

    valid = mask.astype(bool)
    both_valid = valid[1:] & valid[:-1]
    idx = np.flatnonzero(both_valid) + 1
    safe_dt = np.maximum(np.asarray(dt, dtype=np.float32), 1e-6)
    vel[idx] = (feats[idx] - feats[idx - 1]) / safe_dt[idx, None]
    return vel.astype(np.float32)


def combine_with_velocity(features: np.ndarray, mask: np.ndarray,
                          dt: np.ndarray) -> np.ndarray:
    """Return concat([X, V], axis=-1): shape (T, 2F). Inputs untouched."""
    v = compute_time_aware_velocity(features, mask, dt)
    return np.concatenate([features, v], axis=-1).astype(np.float32)
