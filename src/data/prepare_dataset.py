"""
Dataset preparation / validation command for the extracted features.

Validates all .npz files, builds the deterministic class mapping,
creates the stratified 70/15/15 split (seed 42), computes balanced
class weights from the TRAIN split only, prints a summary report and
saves split metadata to outputs/dataset_split.json.

Usage:
    python src/data/prepare_dataset.py [--features-root PATH] [--metadata PATH]
                                       [--batch-size N] [--num-workers N]
"""

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from azsl_dataset import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_FEATURES_ROOT,
    DEFAULT_METADATA_PATH,
    DEFAULT_NUM_WORKERS,
    EXPECTED_SHAPE,
    build_class_mapping,
    compute_class_weights,
    make_dataloaders,
    save_split_metadata,
    scan_feature_files,
    stratified_split,
    validate_features,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("prepare_dataset")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-root", type=Path, default=DEFAULT_FEATURES_ROOT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    args = parser.parse_args()

    # 1. Class mapping ------------------------------------------------------
    class_to_idx, idx_to_class = build_class_mapping(args.features_root)
    log.info("Discovered %d classes", len(idx_to_class))
    if len(idx_to_class) != 200:
        log.error("EXPECTED 200 classes but found %d — aborting.", len(idx_to_class))

    # 2. File discovery + validation ----------------------------------------
    samples = scan_feature_files(args.features_root, class_to_idx)
    log.info("Scanning %d .npz files (read-only)...", len(samples))
    result = validate_features(samples, idx_to_class)

    print("=" * 60)
    print("  DATASET VALIDATION")
    print("=" * 60)
    print(f"Total files          : {result.total_files} (expected 8557)")
    print(f"Total classes        : {result.total_classes} (expected 200)")
    print(f"Feature shape        : {list(EXPECTED_SHAPE)}")
    print(f"Malformed .npz       : {len(result.malformed_files)}")
    print(f"Shape errors         : {len(result.shape_errors)}")
    print(f"NaN/Inf files        : {len(result.non_finite_files)}")
    print(f"Classes w/o samples  : {len(result.missing_labels)}")
    for entry in (
        result.malformed_files[:10]
        + result.shape_errors[:10]
        + result.non_finite_files[:10]
        + result.missing_labels[:10]
    ):
        print(f"  !! {entry}")
    if not result.ok:
        print("VALIDATION FAILED — fix reported issues before training.")
        return 1
    print("Validation           : OK")

    # 3. Stratified split -----------------------------------------------------
    split = stratified_split(samples, seed=42)
    train_labels = [samples[i].label for i in split.train]

    print("-" * 60)
    print(f"Train samples        : {len(split.train)}")
    print(f"Val samples          : {len(split.val)}")
    print(f"Test samples         : {len(split.test)}")

    counts = Counter(train_labels)
    present = [counts[c] for c in range(len(idx_to_class)) if c in counts]
    absent = [idx_to_class[c] for c in range(len(idx_to_class)) if c not in counts]
    print(f"Train min/class      : {min(present) if present else 'n/a'}")
    print(f"Train max/class      : {max(present) if present else 'n/a'}")
    if absent or split.incomplete_coverage:
        print(
            "Split coverage gaps  : "
            f"{len(set(absent) | set(split.incomplete_coverage))} class(es)"
        )
        for name in sorted(set(absent) | set(split.incomplete_coverage)):
            print(f"  !! {name}: cannot be represented in all three splits")

    # 4. Class weights (TRAIN only) -------------------------------------------
    weights = compute_class_weights(train_labels, num_classes=len(idx_to_class))
    nonzero = weights[weights > 0]
    print("-" * 60)
    print(f"Class weight range   : [{nonzero.min():.4f}, {nonzero.max():.4f}]")
    print(f"CUDA available       : {__import__('torch').cuda.is_available()}")

    # 5. One DataLoader batch ---------------------------------------------------
    train_loader, _, _ = make_dataloaders(
        split, samples, batch_size=args.batch_size, num_workers=args.num_workers
    )
    x_batch, y_batch = next(iter(train_loader))
    print(f"Batch shape (train)  : X={tuple(x_batch.shape)}, y={tuple(y_batch.shape)}")
    print(f"Batch dtype          : X={x_batch.dtype}, y={y_batch.dtype}")

    # 6. Persist metadata ---------------------------------------------------------
    save_split_metadata(args.metadata, samples, idx_to_class, split, weights)
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
