"""
Dataset regeneration script for Option A (Full Dynamic 280D) features.

Reads existing 126D landmark .npz files in data/features/full/, computes Option A
280D features (pos, velocity, wrist velocity, orientation rays, extension ratios),
resamples to 26 frames, and saves to data/features/v2_full/.

Usage:
    python src/data/build_v2_features.py [--src-root data/features/full]
                                         [--dst-root data/features/v2_full]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.extract_landmarks import compute_v2_features
from src.features.preprocess_sequence import TARGET_SEQ_LEN, preprocess_sequence

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("build_v2_features")


def build_v2_dataset(
    src_root: Path = Path("data/features/full"),
    dst_root: Path = Path("data/features/v2_full"),
) -> int:
    """Read all .npz files from src_root, transform to 280D features, write to dst_root."""
    src_root = Path(src_root)
    dst_root = Path(dst_root)

    if not src_root.is_dir():
        log.error("Source features directory '%s' does not exist.", src_root)
        return 1

    class_dirs = sorted([d for d in src_root.iterdir() if d.is_dir()])
    log.info("Found %d class directories in %s", len(class_dirs), src_root)

    total_files = 0
    total_written = 0
    start_time = time.time()

    for class_dir in class_dirs:
        dst_class_dir = dst_root / class_dir.name
        dst_class_dir.mkdir(parents=True, exist_ok=True)

        npz_paths = sorted(class_dir.glob("*.npz"))
        total_files += len(npz_paths)

        for npz_path in npz_paths:
            dst_path = dst_class_dir / npz_path.name
            try:
                with np.load(npz_path) as data:
                    frame_feats = data["frame_features"]  # (N, 126)
                    frame_valid = data["frame_valid"]    # (N,)

                # 1. Compute 280D per-frame features
                v2_frame_feats = compute_v2_features(frame_feats, frame_valid)  # (N, 280)

                # 2. Resample/pad sequence to target length 26
                v2_seq_feats, mask = preprocess_sequence(
                    v2_frame_feats, frame_valid, target_len=TARGET_SEQ_LEN
                )  # (26, 280), (26,)

                # 3. Save compressed .npz archive
                np.savez_compressed(
                    dst_path,
                    features=v2_seq_feats.astype(np.float32),
                    mask=mask.astype(np.float32),
                    frame_features=v2_frame_feats.astype(np.float32),
                    frame_valid=frame_valid.astype(bool),
                )
                total_written += 1

            except Exception as exc:
                log.error("Error processing %s: %s", npz_path, exc)

    elapsed = time.time() - start_time
    fps = total_written / max(elapsed, 1e-6)

    print("=" * 60)
    print("  OPTION A (280D) DATASET REGENERATION SUMMARY")
    print("=" * 60)
    print(f"Source Root    : {src_root}")
    print(f"Destination    : {dst_root}")
    print(f"Total Classes  : {len(class_dirs)}")
    print(f"Total Processed: {total_written}/{total_files} files")
    print(f"Time Elapsed   : {elapsed:.2f}s ({fps:.1f} files/sec)")
    print(f"Feature Shape  : ({TARGET_SEQ_LEN}, 280)")
    print("=" * 60)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-root", type=Path, default=Path("data/features/full"))
    parser.add_argument("--dst-root", type=Path, default=Path("data/features/v2_full"))
    args = parser.parse_args()

    return build_v2_dataset(args.src_root, args.dst_root)


if __name__ == "__main__":
    sys.exit(main())
