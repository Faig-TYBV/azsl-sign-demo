"""
Tests for AlphabetClassifier single-frame fingerspelling inference.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest
from src.inference.alphabet_classifier import (
    AlphabetClassifier,
    normalize_landmarks,
    build_feature_vector_84,
    detect_control_gesture,
)


@pytest.fixture(scope="module")
def classifier():
    return AlphabetClassifier()


def test_alphabet_classifier_loads(classifier):
    assert classifier.level1_model is not None
    assert len(classifier.clusters) == 6


def test_normalize_landmarks_shape():
    raw = np.random.rand(21, 3).astype(np.float32)
    norm = normalize_landmarks(raw, mirror_x=False)
    assert norm.shape == (21, 3)
    # Wrist should be at origin (0, 0, 0)
    assert np.allclose(norm[0], [0.0, 0.0, 0.0], atol=1e-5)


def test_build_feature_vector_84():
    raw = np.random.rand(21, 3).astype(np.float32)
    norm = normalize_landmarks(raw)
    feat = build_feature_vector_84(norm, (0.0, 0.0))
    assert feat.shape == (84,)
    assert feat.dtype == np.float32


def test_predict_frame_returns_valid_label_and_confidence(classifier):
    raw = np.random.rand(21, 3).astype(np.float32)
    letter, conf = classifier.predict_frame(raw, "Right", min_confidence=0.0)
    assert letter is not None
    assert isinstance(letter, str)
    assert 0.0 <= conf <= 1.0


def test_predict_frame_none_for_invalid_landmarks(classifier):
    letter, conf = classifier.predict_frame(None)
    assert letter is None
    assert conf == 0.0
