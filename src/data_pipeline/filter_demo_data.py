"""
Filter a 25-word balanced subset dataset from an existing feature dataset directory.

Supports undersampling (e.g., max 100 samples per class) to prevent class imbalance
from dominant classes like 'MƏN'.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

# Exact 25 target classes requested for balanced live demo
TARGET_WORDS = [
    "MƏN",
    "SƏN",
    "BİZ",
    "SİZ",
    "O",
    "BU",
    "SALAM",
    "SAĞLAM",
    "VAR",
    "YOX",
    "İSTƏMƏK",
    "GETMƏK",
    "GƏLMƏK",
    "ALMAQ",
    "OLMAQ",
    "YEMƏK",
    "EV",
    "İŞ",
    "BAKI",
    "AZƏRBAYCAN",
    "TELEFON",
    "BU_GÜN",
    "SABAH",
    "BURDA",
    "HARDA",
]

# Alias mapping for user transliterations and spelling variations
ALIAS_MAP = {
    "BU_GÜN": "BU GÜN",
    "BU_GUN": "BU GÜN",
    "BUGUN": "BU GÜN",
    "BUGÜN": "BU GÜN",
    "MEN": "MƏN",
    "SEN": "SƏN",
    "BIZ": "BİZ",
    "SIZ": "SİZ",
    "SAGLAM": "SAĞLAM",
    "ISTEMEK": "İSTƏMƏK",
    "GETMEK": "GETMƏK",
    "GELMEK": "GƏLMƏK",
    "YEMEK": "YEMƏK",
    "IS": "İŞ",
    "AZERBAYCAN": "AZƏRBAYCAN",
}

RANDOM_SEED = 42


def resolve_folder_name(name: str, existing_names: set[str]) -> str | None:
    """Resolve requested word/alias to exact folder name on disk."""
    if name in existing_names:
        return name
    upper_name = name.upper()
    if upper_name in ALIAS_MAP and ALIAS_MAP[upper_name] in existing_names:
        return ALIAS_MAP[upper_name]
    if upper_name in existing_names:
        return upper_name
    return None


def filter_dataset(
    source_dir: Path,
    dest_dir: Path,
    target_words: list[str],
    max_samples_per_class: int = 100,
) -> dict:
    """
    Iterate through source_dir and copy matching target word files to dest_dir,
    undersampling classes with more than max_samples_per_class.
    """
    source_dir = Path(source_dir).resolve()
    dest_dir = Path(dest_dir).resolve()

    if not source_dir.exists():
        print(f"Error: Source directory does not exist: {source_dir}")
        return {"copied": 0, "missing": target_words}

    dest_dir.mkdir(parents=True, exist_ok=True)
    existing_names = set(d.name for d in source_dir.iterdir() if d.is_dir())

    copied_count = 0
    copied_classes = []
    missing_classes = []
    total_files_copied = 0

    rng = random.Random(RANDOM_SEED)

    print("=" * 65)
    print(f" FILTERING & BALANCING DATASET (Max {max_samples_per_class} samples/class)")
    print("=" * 65)

    for word in target_words:
        resolved = resolve_folder_name(word, existing_names)
        if resolved is not None:
            source_item = source_dir / resolved
            target_dest = dest_dir / resolved

            if target_dest.exists():
                shutil.rmtree(target_dest)
            target_dest.mkdir(parents=True, exist_ok=True)

            all_files = sorted(list(source_item.glob("*.npz")))
            initial_count = len(all_files)

            if max_samples_per_class is not None and initial_count > max_samples_per_class:
                selected_files = rng.sample(all_files, max_samples_per_class)
                undersampled_note = f"(undersampled {initial_count} -> {max_samples_per_class})"
            else:
                selected_files = all_files
                undersampled_note = f"(all {initial_count} kept)"

            for f in selected_files:
                shutil.copy2(f, target_dest / f.name)

            n_copied = len(selected_files)
            total_files_copied += n_copied
            copied_count += 1
            copied_classes.append(resolved)

            print(f"  [+] Copied class '{resolved:<15}' : {n_copied:>3} files {undersampled_note}")
        else:
            missing_classes.append(word)
            print(f"  [-] Target word not found in source: '{word}'")

    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"Source Directory     : {source_dir}")
    print(f"Destination Directory: {dest_dir}")
    print(f"Target Words Count   : {len(target_words)}")
    print(f"Classes Copied       : {copied_count}/{len(target_words)}")
    print(f"Total Feature Files  : {total_files_copied}")
    print(f"Classes Missing      : {len(missing_classes)}")
    if missing_classes:
        print(f"Missing List         : {missing_classes}")
    print("=" * 65)

    return {
        "copied": copied_count,
        "copied_classes": copied_classes,
        "missing": missing_classes,
        "total_files": total_files_copied,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Filter and balance 25-word subset features from full dataset"
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/features/full"),
        help="Path to source extracted features directory",
    )
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=Path("dataset/demo_25_words_balanced"),
        help="Path to destination balanced features directory",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Maximum samples to keep per class (undersampling ceiling)",
    )

    args = parser.parse_args()
    filter_dataset(args.source_dir, args.dest_dir, TARGET_WORDS, args.max_samples)
    return 0


if __name__ == "__main__":
    sys.exit(main())
