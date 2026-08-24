"""
FULL dataset landmark extraction for AzSLD_Words_200.

Reuses the exact pilot pipeline (extract_landmarks + preprocess_sequence).
Processes all videos incrementally — one video at a time, never loading the
whole dataset into RAM.

Storage layout (scalable, per-video files):
    data/features/full/<word_label>/<video_stem>.npz
    data/features/full/extraction_metadata.jsonl   (one JSON line per video)

Resumable: if a video's .npz already exists and loads correctly, it is
skipped. Individual failures are logged and do not stop the run.

Usage:
    python src/features/run_full_extraction.py [--data-root ...] [--output-dir ...]
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np

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
log = logging.getLogger("run_full_extraction")

DEFAULT_DATA_ROOT = Path("data/raw/AzSLD_Words_200")
DEFAULT_OUTPUT_DIR = Path("data/features/full")
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
PROGRESS_EVERY = 100


def npz_is_valid(path: Path) -> bool:
    """A feature file is valid if it exists and loads with expected keys."""
    if not path.is_file():
        return False
    try:
        with np.load(path) as d:
            return all(k in d for k in ("features", "mask", "frame_features", "frame_valid"))
    except Exception:
        return False


def collect_videos(data_root: Path) -> list[Path]:
    videos = []
    for class_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        videos.extend(
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
        )
    return videos


def process_video(video_path: Path, landmarker, output_dir: Path, ts_offset: int) -> dict:
    """Extract + save one video. Returns metadata dict (status included)."""
    t0 = time.time()
    result = extract_video_landmarks(video_path, landmarker, timestamp_offset_ms=ts_offset)

    if result.error is not None and result.original_frame_count == 0:
        return {
            "status": "failed",
            "word_label": video_path.parent.name,
            "video_path": str(video_path),
            "error": result.error,
        }

    frame_feats = np.stack(
        [normalize_frame(fr) for fr in result.raw_frames], axis=0
    ) if result.raw_frames else np.zeros((0, MAX_HANDS * FEATURES_PER_HAND), dtype=np.float32)
    frame_valid = np.array(
        [fr.num_hands > 0 for fr in result.raw_frames], dtype=bool
    ) if result.raw_frames else np.zeros(0, dtype=bool)

    features, mask = preprocess_sequence(frame_feats, frame_valid, TARGET_SEQ_LEN)

    out_dir = output_dir / video_path.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_path.stem}.npz"
    np.savez_compressed(
        out_path,
        features=features,
        mask=mask,
        frame_features=frame_feats,
        frame_valid=frame_valid,
    )

    detection_rate = (
        result.frames_with_hands / result.original_frame_count
        if result.original_frame_count > 0 else 0.0
    )
    return {
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Full dataset landmark extraction")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = args.output_dir / "extraction_metadata.jsonl"

    videos = collect_videos(args.data_root)
    total = len(videos)
    log.info("Total videos found: %d", total)
    total = len(videos)
    log.info("Total videos found: %d", total)

    # --- Pre-flight resumability audit ---
    existing_valid = sum(
        1 for vp in videos
        if npz_is_valid(args.output_dir / vp.parent.name / f"{vp.stem}.npz")
    )
    invalid_or_missing = total - existing_valid
    print("=" * 60)
    print("FULL EXTRACTION PRE-FLIGHT SUMMARY")
    print(f"  Total videos:                  {total}")
    print(f"  Existing VALID feature files:  {existing_valid}")
    print(f"  Invalid / missing features:    {invalid_or_missing}")
    print(f"  Videos to process:             {invalid_or_missing}")
    print("=" * 60)


    # --- Disk space estimate from pilot ---
    pilot_dir = Path("data/features/pilot")
    pilot_npz = list(pilot_dir.glob("*.npz"))
    if pilot_npz:
        avg_bytes = np.mean([p.stat().st_size for p in pilot_npz])
        est_gb = avg_bytes * total / 1e9
        free_gb = shutil.disk_usage(args.output_dir).free / 1e9
        log.info("Estimated feature storage: %.2f GB (avg %.1f KB/video); free disk: %.1f GB",
                 est_gb, avg_bytes / 1024, free_gb)
        if est_gb > free_gb * 0.9:
            log.error("Insufficient disk space — aborting.")
            sys.exit(1)

    landmarker = create_hand_landmarker(args.model_path)
    processed = skipped = failed = 0
    ts_offset = 0
    t_start = time.time()

    try:
        with open(meta_path, "a", encoding="utf-8") as meta_file:
            for i, vp in enumerate(videos, 1):
                out_path = args.output_dir / vp.parent.name / f"{vp.stem}.npz"
                if npz_is_valid(out_path):
                    skipped += 1
                else:
                    meta = process_video(vp, landmarker, args.output_dir, ts_offset)
                    ts_offset += (meta.get("original_frame_count", 0) + 1) * 34
                    meta_file.write(json.dumps(meta, ensure_ascii=False) + "\n")
                    meta_file.flush()
                    if meta["status"] == "failed":
                        failed += 1
                        log.error("FAILED [%d/%d] %s: %s", i, total, vp, meta.get("error"))
                    else:
                        processed += 1

                if i % PROGRESS_EVERY == 0 or i == total:
                    elapsed = time.time() - t_start
                    eta = elapsed / i * (total - i)
                    log.info(
                        "Progress %d/%d | processed=%d skipped=%d failed=%d | "
                        "elapsed=%.0fs eta=%.0fs",
                        i, total, processed, skipped, failed, elapsed, eta,
                    )
    except KeyboardInterrupt:
        log.warning("Interrupted — progress is saved; rerun to resume.")
    finally:
        landmarker.close()

    elapsed = time.time() - t_start
    log.info("Done in %.0fs | processed=%d skipped=%d failed=%d", elapsed, processed, skipped, failed)


if __name__ == "__main__":
    main()