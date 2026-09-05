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

# Exact 20 target words matching folder names on disk in the dataset
TARGET_WORDS = [
    "MƏN",
    "SƏN",
    "BİZ",
    "O",
    "SALAM",
    "NECƏ",
    "YAXŞI",
    "DEYİL",
    "XAHİŞ",
    "HALALDIR",
    "İSTƏMƏK",
    "GETMƏK",
    "ÖYRƏNMƏK",
    "BİLMƏK",
    "KÖMƏK",
    "BU GÜN",
    "SABAH",
    "UNİVERSİTET",
    "AD",
    "VAXT",
]

# Alias mapping for user transliterations
ALIAS_MAP = {
    "MEN": "MƏN",
    "SEN": "SƏN",
    "BIZ": "BİZ",
    "O": "O",
    "SALAM": "SALAM",
    "NECESEN": "NECƏ",
    "YAXSI": "YAXŞI",
    "PIS": "DEYİL",
    "TESEKKUR": "XAHİŞ",
    "SAG_OL": "HALALDIR",
    "ISTEMEK": "İSTƏMƏK",
    "GETMEK": "GETMƏK",
    "OYRENMEK": "ÖYRƏNMƏK",
    "BILMEK": "BİLMƏK",
    "KOMEK_ETMEK": "KÖMƏK",
    "KOMEK": "KÖMƏK",
    "BU_GUN": "BU GÜN",
    "SABAH": "SABAH",
    "UNIVERSITET": "UNİVERSİTET",
    "AD": "AD",
    "VAXT": "VAXT",
}


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

    existing_names = set(d.name for d in source_dir.iterdir() if d.is_dir())

    copied_count = 0
    copied_classes = []
    missing_classes = []

    for word in target_words:
        resolved = resolve_folder_name(word, existing_names)
        if resolved is not None:
            source_item = source_dir / resolved
            target_dest = dest_dir / resolved

            if target_dest.exists():
                shutil.rmtree(target_dest)
            shutil.copytree(source_item, target_dest)
            n_files = len(list(target_dest.glob("*.npz")))
            print(f"  [+] Copied class folder '{resolved}' ({n_files} files)")

            copied_count += 1
            copied_classes.append(resolved)
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
