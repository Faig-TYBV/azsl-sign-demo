"""
Filter a 20-word subset dataset from an existing feature dataset directory.

Copies feature files/directories for the 20 target words into a destination
directory for lightweight demo model training.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

TARGET_WORDS = [
    "Men",
    "Sen",
    "Biz",
    "O",
    "Salam",
    "Necesen",
    "Yaxsi",
    "Pis",
    "Tesekkur",
    "Sag_ol",
    "Istemek",
    "Getmek",
    "Oyrenmek",
    "Bilmek",
    "Komek_etmek",
    "Bu_gun",
    "Sabah",
    "Universitet",
    "Ad",
    "Vaxt",
]


def filter_dataset(
    source_dir: Path, dest_dir: Path, target_words: list[str]
) -> dict:
    """
    Iterate through source_dir and copy matching target word directories/files to dest_dir.
    """
    source_dir = Path(source_dir).resolve()
    dest_dir = Path(dest_dir).resolve()

    if not source_dir.exists():
        print(f"Error: Source directory does not exist: {source_dir}")
        return {"copied": 0, "missing": target_words}

    dest_dir.mkdir(parents=True, exist_ok=True)

    # Get available items in source_dir
    existing_items = {item.name: item for item in source_dir.iterdir()}
    # Case-insensitive map for convenient matching if folder names vary
    lower_map = {name.lower(): item for name, item in existing_items.items()}

    copied_count = 0
    copied_classes = []
    missing_classes = []

    for word in target_words:
        matched_item = None
        if word in existing_items:
            matched_item = existing_items[word]
        elif word.lower() in lower_map:
            matched_item = lower_map[word.lower()]

        if matched_item is not None:
            target_dest = dest_dir / matched_item.name
            if matched_item.is_dir():
                if target_dest.exists():
                    shutil.rmtree(target_dest)
                shutil.copytree(matched_item, target_dest)
                n_files = len(list(target_dest.glob("*.npz")))
                print(f"  [+] Copied class folder '{matched_item.name}' ({n_files} files)")
            else:
                shutil.copy2(matched_item, target_dest)
                print(f"  [+] Copied file '{matched_item.name}'")

            copied_count += 1
            copied_classes.append(matched_item.name)
        else:
            missing_classes.append(word)
            print(f"  [-] Target word not found in source: '{word}'")

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"Source Directory     : {source_dir}")
    print(f"Destination Directory: {dest_dir}")
    print(f"Target Words Count   : {len(target_words)}")
    print(f"Classes Copied       : {copied_count}")
    print(f"Classes Missing      : {len(missing_classes)}")
    if missing_classes:
        print(f"Missing List         : {missing_classes}")

    return {
        "copied": copied_count,
        "copied_classes": copied_classes,
        "missing": missing_classes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Filter 20-word subset features from full dataset"
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
        default=Path("dataset/demo_20_words_features"),
        help="Path to destination 20-word features directory",
    )

    args = parser.parse_args()
    filter_dataset(args.source_dir, args.dest_dir, TARGET_WORDS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
