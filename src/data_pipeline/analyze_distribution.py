"""
Exploratory Data Analysis (EDA) on AzSL 200-word dataset feature files.
Scans data/features/full, counts samples per class, and outputs the top classes.
"""

import argparse
from pathlib import Path


def analyze_distribution(dataset_dir: Path, top_n: int = 50):
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        print(f"Error: Dataset directory {dataset_dir} does not exist.")
        return

    class_counts = []
    total_samples = 0

    for class_folder in dataset_dir.iterdir():
        if class_folder.is_dir():
            # Count .npz and .csv files
            files = [
                f for f in class_folder.iterdir()
                if f.is_file() and f.suffix in (".npz", ".csv", ".npy")
            ]
            count = len(files)
            class_counts.append((class_folder.name, count))
            total_samples += count

    # Sort descending by sample count, then alphabetically
    class_counts.sort(key=lambda x: (-x[1], x[0]))

    print("=" * 70)
    print(f" AZSL DATASET DISTRIBUTION ANALYSIS: {dataset_dir}")
    print("=" * 70)
    print(f"Total Classes Found : {len(class_counts)}")
    print(f"Total Sample Files  : {total_samples}")
    if class_counts:
        counts_only = [c[1] for c in class_counts]
        print(f"Min Samples / Class : {min(counts_only)} ('{class_counts[-1][0]}')")
        print(f"Max Samples / Class : {max(counts_only)} ('{class_counts[0][0]}')")
        print(f"Mean Samples / Class: {total_samples / len(class_counts):.1f}")
        print(f"Median Samples      : {sorted(counts_only)[len(counts_only)//2]}")
    print("=" * 70)
    print(f"{'Rank':<5} | {'Class Name':<28} | {'Samples':<8} | {'% of Total':<10}")
    print("-" * 70)

    for rank, (name, count) in enumerate(class_counts[:top_n], start=1):
        pct = (count / total_samples * 100) if total_samples > 0 else 0.0
        print(f"{rank:<5} | {name:<28} | {count:<8} | {pct:>6.2f}%")

    print("-" * 70)
    print(f"Showing Top {min(top_n, len(class_counts))} of {len(class_counts)} classes.")
    print("=" * 70)

    return class_counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze sample distribution in feature dataset.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/features/full"),
        help="Path to dataset directory (default: data/features/full)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=50,
        help="Number of top classes to display (default: 50)",
    )
    args = parser.parse_args()
    analyze_distribution(args.dataset_dir, args.top)
