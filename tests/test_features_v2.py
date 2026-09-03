"""
Unit tests for Option A (Full Dynamic 280D) feature extraction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.extract_landmarks import (
    FrameResult,
    compute_v2_features,
    normalize_frame,
    normalize_sequence,
)


def _make_dummy_landmarks(T: int = 10, valid_from: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Helper to generate dummy 3D landmarks sequence (T, 2, 21, 3)."""
    np.random.seed(42)
    landmarks = np.zeros((T, 2, 21, 3), dtype=np.float32)
    mask = np.zeros(T, dtype=bool)
    raw_wrists = np.zeros((T, 2, 3), dtype=np.float32)

    for t in range(valid_from, T):
        mask[t] = True
        for slot in range(2):
            # Non-zero landmarks
            lm = np.random.randn(21, 3).astype(np.float32) * 0.1
            lm += np.array([slot * 0.5, 0.2, 0.1], dtype=np.float32)
            landmarks[t, slot] = lm
            raw_wrists[t, slot] = lm[0]

    return landmarks, mask, raw_wrists


def test_v2_output_shape_strictly_280d():
    """Verify output shape is strictly (T, 280) for a valid sequence."""
    T = 15
    landmarks, mask, raw_wrists = _make_dummy_landmarks(T, valid_from=0)
    feats = compute_v2_features(landmarks, validity_mask=mask, raw_wrists_seq=raw_wrists)

    assert feats.shape == (T, 280)
    assert feats.dtype == np.float32
    assert not np.isnan(feats).any()
    assert not np.isinf(feats).any()


def test_first_frame_velocity_and_zero_padding_no_nans():
    """Verify first frame velocity handles zero-padding correctly without NaNs."""
    T = 10
    # First 2 frames invalid (zero-padded)
    landmarks, mask, raw_wrists = _make_dummy_landmarks(T, valid_from=2)
    feats = compute_v2_features(landmarks, validity_mask=mask, raw_wrists_seq=raw_wrists)

    assert feats.shape == (T, 280)
    assert not np.isnan(feats).any()

    # Frame 0 and Frame 1 velocity features (indices 126..251) must be strictly 0
    vel_slice = feats[:, 126:252]
    assert np.all(vel_slice[0] == 0.0)
    assert np.all(vel_slice[1] == 0.0)
    # Frame 2 is first valid frame, so velocity from frame 1 (invalid) must also be 0
    assert np.all(vel_slice[2] == 0.0)


def test_orientation_vectors_are_unit_vectors():
    """Verify orientation vectors are normalized unit vectors (norm approx 1.0) for valid hands."""
    T = 5
    landmarks, mask, raw_wrists = _make_dummy_landmarks(T, valid_from=0)
    feats = compute_v2_features(landmarks, validity_mask=mask, raw_wrists_seq=raw_wrists)

    # Orientation features slice: indices 258..269 (12D: 6D per hand)
    orient_slice = feats[:, 258:270]  # shape (T, 12)

    for t in range(T):
        # Hand 0 index ray (indices 0..2) & palm normal (indices 3..5)
        h0_ray = orient_slice[t, 0:3]
        h0_normal = orient_slice[t, 3:6]
        h1_ray = orient_slice[t, 6:9]
        h1_normal = orient_slice[t, 9:12]

        assert np.linalg.norm(h0_ray) == pytest.approx(1.0, abs=1e-4)
        assert np.linalg.norm(h0_normal) == pytest.approx(1.0, abs=1e-4)
        assert np.linalg.norm(h1_ray) == pytest.approx(1.0, abs=1e-4)
        assert np.linalg.norm(h1_normal) == pytest.approx(1.0, abs=1e-4)


def test_backward_compatibility_v1_vs_v2():
    """Verify backward compatibility of normalize_frame and normalize_sequence."""
    fr = FrameResult(num_hands=1)
    fr.landmarks[0] = np.random.randn(21, 3).astype(np.float32)

    v1_feat = normalize_frame(fr, version="v1")
    assert v1_feat.shape == (126,)

    v2_feat = normalize_frame(fr, version="v2")
    assert v2_feat.shape == (280,)

    seq = [fr, fr]
    v1_seq = normalize_sequence(seq, version="v1")
    assert v1_seq.shape == (2, 126)

    v2_seq = normalize_sequence(seq, version="v2")
    assert v2_seq.shape == (2, 280)
