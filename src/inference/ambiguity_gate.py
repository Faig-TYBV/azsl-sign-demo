"""
Ambiguity gate and postprocessing mitigations for AzSL word recognition.

Suppresses or defers emission when top-1 and top-2 predictions belong to
confusable morphological pairs (specifically pronoun cluster: MƏN, MƏNƏ, MƏNİM)
and their confidence delta (margin) is less than margin_threshold (default 0.15).
"""

from __future__ import annotations

from typing import Sequence

# Morphological pronoun cluster identified in error analysis
PRONOUN_CLUSTER: frozenset[str] = frozenset({"MƏN", "MƏNƏ", "MƏNİM"})
DEFAULT_MARGIN_THRESHOLD: float = 0.15


def is_ambiguous_prediction(
    top1_class: str,
    top2_class: str,
    top1_prob: float,
    top2_prob: float,
    margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
    confusable_cluster: frozenset[str] | set[str] = PRONOUN_CLUSTER,
) -> bool:
    """
    Check if top-1 and top-2 predictions belong to a highly confusable
    morphological pair (e.g. pronoun cluster) and their confidence margin is
    less than margin_threshold.

    Args:
        top1_class: Label of top-1 predicted class.
        top2_class: Label of top-2 predicted class.
        top1_prob: Confidence/probability of top-1 prediction.
        top2_prob: Confidence/probability of top-2 prediction.
        margin_threshold: Minimum required probability margin (top1 - top2).
        confusable_cluster: Set of mutually confusable class labels.

    Returns:
        True if prediction is ambiguous (both classes in cluster AND margin < threshold).
        False otherwise.
    """
    if top1_class in confusable_cluster and top2_class in confusable_cluster:
        margin = top1_prob - top2_prob
        if margin < margin_threshold:
            return True
    return False


def check_ambiguity_gate(
    top_k_predictions: Sequence[tuple[str, float]],
    margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
    confusable_cluster: frozenset[str] | set[str] = PRONOUN_CLUSTER,
) -> bool:
    """
    Convenience wrapper taking a sequence of (class_name, probability) tuples.

    Returns:
        True if the top-1 and top-2 predictions are ambiguous.
    """
    if len(top_k_predictions) < 2:
        return False
    top1_class, top1_prob = top_k_predictions[0]
    top2_class, top2_prob = top_k_predictions[1]
    return is_ambiguous_prediction(
        top1_class,
        top2_class,
        top1_prob,
        top2_prob,
        margin_threshold=margin_threshold,
        confusable_cluster=confusable_cluster,
    )
