"""
Left- and right-handed signing must give the same result.

A sign made with the left hand is the mirror image of the same sign made with
the right. Neither model knows that on its own:

  * The word GRU sees wrist-relative coordinates with no mirroring at all, and a
    lone hand always lands in slot 0 whichever hand it is. A left-handed sign
    therefore reaches it as the x-flipped version of what it learned.
  * The alphabet MLP was trained on hands "mirrored to Right" (its model JSON
    says so), so it needs to know which hand it is looking at.

This module turns a frame or a whole trial into the reference hand's view.

Only one thing here depends on how MediaPipe labels hands:
MEDIAPIPE_LABELS_ARE_SWAPPED. MediaPipe documents its handedness as assuming
a mirrored (selfie) image; this app feeds raw, un-mirrored camera frames, so a
real right hand is reported as "Left". If that ever stops being true, the
overlay on the workspace page shows it directly: raise your right hand, and it
should read "Sağ".

The mirroring itself is symmetric by construction: a left-handed trial is
flipped onto the right-handed one, so the two reach the model identically
whatever that flag says. The flag only decides which of the two orientations
they both end up in.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

# Keep in sync with MEDIAPIPE_LABELS_ARE_SWAPPED in
# src/web_demo/frontend/js/azsl_features.js (tests/test_handedness.py checks).
MEDIAPIPE_LABELS_ARE_SWAPPED = True

# The hand the word model is treated as having been trained on. Trials signed
# with the other hand are mirrored onto it.
WORD_REFERENCE_HAND = "Right"

FEATURES_PER_HAND = 63
FEATURE_DIM = 126

# Two-hand-only trials have no per-frame vote, so the dominant hand is the one
# whose shape changes more. Within this ratio the hands are treated as equally
# active (symmetric signs), and the trial is left as is.
TWO_HAND_ACTIVITY_MARGIN = 1.25


def actual_hand(raw_label: Optional[str]) -> Optional[str]:
    """MediaPipe's label -> the hand it really is ("Left"/"Right"), or None."""
    if raw_label not in ("Left", "Right"):
        return None
    if MEDIAPIPE_LABELS_ARE_SWAPPED:
        return "Left" if raw_label == "Right" else "Right"
    return raw_label


def _slot_present(feat: np.ndarray, slot: int) -> bool:
    block = feat[slot * FEATURES_PER_HAND:(slot + 1) * FEATURES_PER_HAND]
    return bool(np.any(block != 0))


def mirror_frame_126(feat: np.ndarray) -> np.ndarray:
    """The 126-dim frame as it would be had the camera image been mirrored.

    Mirroring the image maps x -> 1 - x, which in wrist-relative coordinates is
    x -> -x. It also swaps MediaPipe's labels, and slots are ordered by label,
    so with two hands the slots swap too. A lone hand stays in slot 0.
    """
    out = np.asarray(feat, dtype=np.float32).copy()
    out[0::3] = -out[0::3]
    out[out == 0] = 0.0  # no -0.0: keeps empty slots bit-identical to zeros
    if _slot_present(out, 1):
        out = np.concatenate([out[FEATURES_PER_HAND:], out[:FEATURES_PER_HAND]])
    return out


def dominant_hand(
    feats: np.ndarray, frame_labels: Sequence[Optional[Sequence[str]]]
) -> Optional[str]:
    """Which hand is signing in this trial: "Left", "Right", or None (unknown).

    feats: (T, 126). frame_labels: per frame, MediaPipe's raw labels in slot
    order (or None/[] when unknown).

    Frames with a single hand vote for it. That is decided over the whole trial,
    not per frame, so one mislabelled frame can't flip half the sequence. With
    two hands in every frame, the dominant hand is the one doing more: the
    other is usually a still base.
    """
    votes = {"Left": 0, "Right": 0}
    activity = {"Left": 0.0, "Right": 0.0}
    prev = None
    for t, feat in enumerate(feats):
        raw = list(frame_labels[t] or []) if t < len(frame_labels) else []
        hands = [actual_hand(r) for r in raw]
        n_present = int(_slot_present(feat, 0)) + int(_slot_present(feat, 1))

        if n_present == 1 and len(hands) >= 1 and hands[0] is not None:
            votes[hands[0]] += 1
            prev = None
        elif n_present == 2 and len(hands) >= 2 and None not in hands[:2] \
                and hands[0] != hands[1]:
            if prev is not None:
                for slot in (0, 1):
                    block = slice(slot * FEATURES_PER_HAND, (slot + 1) * FEATURES_PER_HAND)
                    activity[hands[slot]] += float(np.linalg.norm(feat[block] - prev[block]))
            prev = feat
        else:
            prev = None

    if votes["Left"] or votes["Right"]:
        if votes["Left"] == votes["Right"]:
            return None
        return "Left" if votes["Left"] > votes["Right"] else "Right"
    if activity["Left"] > TWO_HAND_ACTIVITY_MARGIN * activity["Right"]:
        return "Left"
    if activity["Right"] > TWO_HAND_ACTIVITY_MARGIN * activity["Left"]:
        return "Right"
    return None


def canonicalize_trial(
    feats: np.ndarray,
    frame_labels: Sequence[Optional[Sequence[str]]],
    reference: str = WORD_REFERENCE_HAND,
) -> Tuple[np.ndarray, bool]:
    """Mirror a trial signed with the non-reference hand onto the reference one.

    Returns (features, mirrored). Unknown dominance leaves the trial untouched,
    which is exactly today's behaviour.
    """
    feats = np.asarray(feats, dtype=np.float32)
    dominant = dominant_hand(feats, frame_labels)
    if dominant is None or dominant == reference:
        return feats, False
    return np.stack([mirror_frame_126(f) for f in feats]).astype(np.float32), True


def clean_labels(labels) -> List[Optional[str]]:
    """Labels from the browser are untrusted input. Keep at most two, in slot
    order; anything unrecognised becomes None rather than being dropped, so a
    bad first label can't shift the second into slot 0."""
    if not isinstance(labels, (list, tuple)):
        return []
    return [l if l in ("Left", "Right") else None for l in labels[:2]]
