"""
Unit tests for ambiguity gate logic (src/inference/ambiguity_gate.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from src.inference.ambiguity_gate import (
    PRONOUN_CLUSTER,
    check_ambiguity_gate,
    is_ambiguous_prediction,
)


def test_pronoun_pair_with_small_margin_is_ambiguous():
    """Pronoun pair ('MƏN', 'MƏNƏ') with margin < 0.15 is flagged as ambiguous."""
    assert is_ambiguous_prediction("MƏN", "MƏNƏ", 0.45, 0.35, margin_threshold=0.15) is True
    assert is_ambiguous_prediction("MƏNƏ", "MƏNİM", 0.40, 0.38, margin_threshold=0.15) is True
    assert is_ambiguous_prediction("MƏNİM", "MƏN", 0.50, 0.40, margin_threshold=0.15) is True


def test_pronoun_pair_with_large_margin_is_not_ambiguous():
    """Pronoun pair ('MƏN', 'MƏNƏ') with margin >= 0.15 is NOT ambiguous."""
    assert is_ambiguous_prediction("MƏN", "MƏNƏ", 0.60, 0.40, margin_threshold=0.15) is False
    assert is_ambiguous_prediction("MƏN", "MƏNİM", 0.75, 0.20, margin_threshold=0.15) is False


def test_non_pronoun_pair_is_not_ambiguous():
    """If top-1 or top-2 is outside the pronoun cluster, margin check is ignored."""
    assert is_ambiguous_prediction("MƏN", "BAXMAQ", 0.45, 0.44, margin_threshold=0.15) is False
    assert is_ambiguous_prediction("ALMAQ", "MƏNƏ", 0.40, 0.39, margin_threshold=0.15) is False
    assert is_ambiguous_prediction("ALMAQ", "VERMƏK", 0.50, 0.49, margin_threshold=0.15) is False


def test_check_ambiguity_gate_wrapper():
    """Test check_ambiguity_gate convenience wrapper."""
    top_k_ambig = [("MƏN", 0.42), ("MƏNƏ", 0.35), ("ALMAQ", 0.10)]
    top_k_clear = [("MƏN", 0.65), ("MƏNƏ", 0.20), ("ALMAQ", 0.05)]
    top_k_other = [("MƏN", 0.42), ("BAXMAQ", 0.35), ("MƏNƏ", 0.10)]

    assert check_ambiguity_gate(top_k_ambig) is True
    assert check_ambiguity_gate(top_k_clear) is False
    assert check_ambiguity_gate(top_k_other) is False
    assert check_ambiguity_gate([("MƏN", 0.90)]) is False  # len < 2


def test_custom_threshold_and_cluster():
    """Test custom margin threshold and confusable cluster."""
    custom_cluster = frozenset({"A", "B"})
    assert (
        is_ambiguous_prediction(
            "A", "B", 0.50, 0.45, margin_threshold=0.10, confusable_cluster=custom_cluster
        )
        is True
    )
    assert (
        is_ambiguous_prediction(
            "A", "B", 0.50, 0.35, margin_threshold=0.10, confusable_cluster=custom_cluster
        )
        is False
    )
