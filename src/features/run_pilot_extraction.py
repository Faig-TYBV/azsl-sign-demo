"""
Pilot landmark extraction for AzSLD_Words_200.

Selects a small number of videos (default 16) spread across different word
classes, runs the MediaPipe hand landmark pipeline, saves per-video .npz
feature files + JSON metadata under data/features/pilot/, and prints a
summary report.

Usage:
    python src/features/run_pilot_extraction.py [--num-videos 16] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np

# Allow running as a script from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.extract_landmarks import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    FEATURES_PER_HAND,
    MAX_HANDS,
    PREPROCESSING_VERSION,
    create_hand_landmarker,
    extract_video_landmarks,
    normalize_frame,
)
from src.features.preprocess_sequence import (  # noqa: E402
    TARGET_SEQ_LEN,
    preprocess_sequence,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("run_pilot_extraction")

DEFAULT_DATA_ROOT = Path("data/raw/AzSLD_Words_200")
DEFAULT_OUTPUT_DIR = Path("data/features/pilot")
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}


# ---------------------------------------------------------------------------
# Pilot video selection
# ---------------------------------------------------------------------------
def select_pilot_videos(data_root: Path, num_videos: int, seed: int) -> list[Path]:
    """
    Deterministically pick videos spread across different word classes.

    Strategy: shuffle class folders with a fixed seed, then take one random
    video from each class until num_videos is reached.
    """
    class_dirs = sorted(d for d in data_root.iterdir() if d.is_dir())
    if not class_dirs:
        log.error("No class folders found in %s", data_root)
        sys.exit(1)

    rng = random.Random(seed)
    classes = class_dirs[:]
    rng.shuffle(classes)

    selected: list[Path] = []
    for class_dir in classes:
        if len(selected) >= num_videos:
            break
        videos = [
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
        ]
        if not videos:
            continue
        selected.append(rng.choice(videos))

    log.info("Selected %d pilot videos from %d classes (seed=%d)",
             len(selected), len({v.parent.name for v in selected}), seed)
    return selected


# ---------------------------------------------------------------------------
# Per-video processing
# ---------------------------------------------------------------------------
def process_video(video_path: Path, landmarker, output_dir: Path, ts_offset: int = 0) -> dict:
    """Extract, normalize, temporally preprocess and save one video."""
    t0 = time.time()
    result = extract_video_landmarks(video_path, landmarker, timestamp_offset_ms=ts_offset)

    if result.error is not None and result.original_frame_count == 0:
        return {
            "status": "failed",
            "video_path": str(video_path),
            "word_label": video_path.parent.name,
            "error": result.error,
        }

    # Normalize each frame into flat feature vectors
    frame_feats = np.stack(
        [normalize_frame(fr) for fr in result.raw_frames], axis=0
    ) if result.raw_frames else np.zeros((0, MAX_HANDS * FEATURES_PER_HAND), dtype=np.float32)
    frame_valid = np.array(
        [fr.num_hands > 0 for fr in result.raw_frames], dtype=bool
    ) if result.raw_frames else np.zeros(0, dtype=bool)

    # Temporal resampling / padding to TARGET_SEQ_LEN
    features, mask = preprocess_sequence(frame_feats, frame_valid, TARGET_SEQ_LEN)

    # Save .npz
    out_name = f"{video_path.parent.name}__{video_path.stem}.npz"
    out_path = output_dir / out_name
    np.savez_compressed(
        out_path,
        features=features,          # (26, 126) float32
        mask=mask,                  # (26,) float32
        frame_features=frame_feats, # (n_frames, 126) raw normalized
        frame_valid=frame_valid,    # (n_frames,) bool
    )

    detection_rate = (
        result.frames_with_hands / result.original_frame_count
        if result.original_frame_count > 0 else 0.0
    )

    metadata = {
        "status": "success" if result.error is None else "partial_error",
        "word_label": result.word_label,
        "video_path": str(video_path),
        "npz_path": str(out_path),
        "original_frame_count": result.original_frame_count,
        "original_fps": result.original_fps,
        "num_detected_hands_max": result.max_hands_seen,
        "detection_rate": round(detection_rate, 4),
        "frames_with_hands": result.frames_with_hands,
        "frames_no_hand": result.frames_no_hand,
        "sequence_length": int(features.shape[0]),
        "feature_dimension": int(features.shape[1]),
        "preprocessing_version": PREPROCESSING_VERSION,
        "processing_time_sec": round(time.time() - t0, 3),
        "error": result.error,
    }
    return metadata


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Pilot landmark extraction")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--num-videos", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    videos = select_pilot_videos(args.data_root, args.num_videos, args.seed)
    if not videos:
        log.error("No videos selected — aborting.")
        sys.exit(1)

    log.info("Loading HandLandmarker model from %s", args.model_path)
    landmarker = create_hand_landmarker(args.model_path)

    all_metadata: list[dict] = []
    ts_offset = 0  # cumulative timestamp so one landmarker stays valid across videos
    try:
        for i, vp in enumerate(videos, 1):
            log.info("[%d/%d] Processing %s / %s",
                     i, len(videos), vp.parent.name, vp.name)
            meta = process_video(vp, landmarker, args.output_dir, ts_offset)
            ts_offset += (meta.get("original_frame_count", 0) + 1) * 34  # > 1 frame gap
            all_metadata.append(meta)
            log.info(
                "  -> %s | hands=%d | det_rate=%.2f | frames=%d | %.2fs",
                meta["status"], meta.get("num_detected_hands_max", 0),
                meta.get("detection_rate", 0.0),
                meta.get("original_frame_count", 0),
                meta.get("processing_time_sec", 0.0),
            )
    finally:
        landmarker.close()

    # Save run metadata
    run_meta_path = args.output_dir / "pilot_metadata.json"
    with open(run_meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run": {
                    "num_videos_requested": args.num_videos,
                    "seed": args.seed,
                    "preprocessing_version": PREPROCESSING_VERSION,
                    "target_sequence_length": TARGET_SEQ_LEN,
                    "features_per_hand": FEATURES_PER_HAND,
                    "max_hands": MAX_HANDS,
                },
                "samples": all_metadata,
            },
            f, indent=2, ensure_ascii=False,
        )
    log.info("Metadata saved -> %s", run_meta_path)

    # ---------------- Summary report ----------------
    successes = [m for m in all_metadata if m["status"] == "success"]
    partials = [m for m in all_metadata if m["status"] == "partial_error"]
    failures = [m for m in all_metadata if m["status"] == "failed"]
    rates = [m["detection_rate"] for m in successes + partials]
    one_hand = sum(1 for m in successes + partials if m["num_detected_hands_max"] == 1)
    two_hand = sum(1 for m in successes + partials if m["num_detected_hands_max"] == 2)
    no_hand_videos = sum(1 for m in successes + partials if m["detection_rate"] == 0.0)
    missing_det = sum(1 for m in successes + partials if m["frames_no_hand"] > 0)

    print("\n" + "=" * 60)
    print("  PILOT EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"  Videos processed      : {len(all_metadata)}")
    print(f"  Successful            : {len(successes)}")
    print(f"  Partial errors        : {len(partials)}")
    print(f"  Failed                : {len(failures)}")
    if rates:
        print(f"  Avg detection rate    : {np.mean(rates):.3f}")
        print(f"  Min detection rate    : {np.min(rates):.3f}")
        print(f"  Max detection rate    : {np.max(rates):.3f}")
    print(f"  One-hand videos       : {one_hand}")
    print(f"  Two-hand videos       : {two_hand}")
    print(f"  Videos w/ no hands    : {no_hand_videos}")
    print(f"  Videos w/ missing det : {missing_det}")
    if successes + partials:
        m0 = successes[0] if successes else partials[0]
        print(f"  Feature shape         : ({m0['sequence_length']}, {m0['feature_dimension']})")
    print("=" * 60)
    print("  Example output files:")
    for m in (successes + partials)[:5]:
        print(f"    {m['npz_path']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()