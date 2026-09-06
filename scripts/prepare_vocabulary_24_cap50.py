import json
import random
from pathlib import Path
from collections import Counter
import numpy as np
import torch

FINAL_VOCABULARY = [
    "MƏN", "SƏN", "BİZ", "SİZ", "SALAM", "SAĞLAM", "İSTƏMƏK", "GETMƏK", "GƏLMƏK",
    "YEMƏK", "ALMAQ", "OLMAQ", "EV", "BAKI", "AZƏRBAYCAN", "TELEFON", "HARDA", "NECƏ",
    "BURDA", "BU GÜN", "SABAH", "VAR", "YOX", "BU"
]

CAP = 50
SEED_BASE = 42

def prepare_subset():
    print("=== PREPARING ISOLATED 24-CLASS DATASET SUBSET (CAP 50) ===")

    # 1. Load original ground-truth split
    orig_split_path = Path("outputs/dataset_split.json")
    with open(orig_split_path, "r", encoding="utf-8") as f:
        orig_data = json.load(f)

    orig_train = orig_data["splits"]["train"]
    orig_val = orig_data["splits"]["val"]
    orig_test = orig_data["splits"]["test"]

    # Group original paths by class
    def group_by_class(paths):
        grouped = {c: [] for c in FINAL_VOCABULARY}
        for p in paths:
            c = Path(p).parent.name
            if c in grouped:
                grouped[c].append(p)
        return grouped

    train_by_class = group_by_class(orig_train)
    val_by_class = group_by_class(orig_val)
    test_by_class = group_by_class(orig_test)

    # Print class-by-class table before selection
    print(f"{'Class':<15} | {'Orig Train':<11} | {'Selected Train':<14} | {'Orig Val':<9} | {'Val':<5} | {'Orig Test':<10} | {'Test':<5}")
    print("-" * 80)

    selected_train = []
    selected_val = []
    selected_test = []

    class_to_idx = {name: i for i, name in enumerate(FINAL_VOCABULARY)}
    idx_to_class = list(FINAL_VOCABULARY)

    per_class_summary = {}

    for idx, c in enumerate(FINAL_VOCABULARY):
        orig_tr_paths = train_by_class[c]
        orig_vl_paths = val_by_class[c]
        orig_ts_paths = test_by_class[c]

        # Deterministic selection for train
        sorted_tr = sorted(orig_tr_paths)
        if len(sorted_tr) > CAP:
            rng = random.Random(SEED_BASE + idx)
            chosen_tr = sorted(rng.sample(sorted_tr, CAP))
        else:
            chosen_tr = sorted_tr

        chosen_vl = sorted(orig_vl_paths)
        chosen_ts = sorted(orig_ts_paths)

        selected_train.extend(chosen_tr)
        selected_val.extend(chosen_vl)
        selected_test.extend(chosen_ts)

        per_class_summary[c] = {
            "class_index": idx,
            "original_train": len(orig_tr_paths),
            "selected_train": len(chosen_tr),
            "original_val": len(orig_vl_paths),
            "selected_val": len(chosen_vl),
            "original_test": len(orig_ts_paths),
            "selected_test": len(chosen_ts),
            "was_capped": len(orig_tr_paths) > CAP
        }

        print(f"{c:<15} | {len(orig_tr_paths):<11d} | {len(chosen_tr):<14d} | {len(orig_vl_paths):<9d} | {len(chosen_vl):<5d} | {len(orig_ts_paths):<10d} | {len(chosen_ts):<5d}")

    print("-" * 80)
    print(f"{'TOTAL':<15} | {sum(len(train_by_class[c]) for c in FINAL_VOCABULARY):<11d} | {len(selected_train):<14d} | {sum(len(val_by_class[c]) for c in FINAL_VOCABULARY):<9d} | {len(selected_val):<5d} | {sum(len(test_by_class[c]) for c in FINAL_VOCABULARY):<10d} | {len(selected_test):<5d}")

    # Assert exact totals
    assert len(selected_train) == 866, f"Expected 866 train samples, got {len(selected_train)}"
    assert len(selected_val) == 676, f"Expected 676 val samples, got {len(selected_val)}"
    assert len(selected_test) == 676, f"Expected 676 test samples, got {len(selected_test)}"
    assert len(FINAL_VOCABULARY) == 24, f"Expected 24 classes, got {len(FINAL_VOCABULARY)}"
    assert all(d["selected_train"] <= 50 for d in per_class_summary.values()), "Cap violation detected!"

    # Compute class weights from the selected training split only
    # weight_i = N / (C * count_i)
    train_labels = [class_to_idx[Path(p).parent.name] for p in selected_train]
    label_counts = Counter(train_labels)
    n_total = len(train_labels)
    n_classes = len(FINAL_VOCABULARY)
    class_weights = []
    for i in range(n_classes):
        cnt = label_counts[i]
        w = n_total / (n_classes * cnt)
        class_weights.append(round(float(w), 6))

    # Output directories
    out_root = Path("outputs/vocabulary_24_cap50")
    train_dir = out_root / "train"
    val_dir = out_root / "val"
    test_dir = out_root / "test"
    meta_dir = out_root / "metadata"

    for d in [train_dir, val_dir, test_dir, meta_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Write split lists
    with open(train_dir / "train_paths.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(selected_train))
    with open(train_dir / "train_paths.json", "w", encoding="utf-8") as f:
        json.dump(selected_train, f, indent=2)

    with open(val_dir / "val_paths.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(selected_val))
    with open(val_dir / "val_paths.json", "w", encoding="utf-8") as f:
        json.dump(selected_val, f, indent=2)

    with open(test_dir / "test_paths.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(selected_test))
    with open(test_dir / "test_paths.json", "w", encoding="utf-8") as f:
        json.dump(selected_test, f, indent=2)

    # Standard repository-compatible split metadata file
    dataset_split_payload = {
        "seed": SEED_BASE,
        "expected_shape": [26, 126],
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "class_weights": class_weights,
        "splits": {
            "train": selected_train,
            "val": selected_val,
            "test": selected_test
        },
        "incomplete_split_coverage_classes": [],
        "counts": {
            "total": len(selected_train) + len(selected_val) + len(selected_test),
            "train": len(selected_train),
            "val": len(selected_val),
            "test": len(selected_test)
        }
    }

    split_meta_path = out_root / "dataset_split_24_cap50.json"
    with open(split_meta_path, "w", encoding="utf-8") as f:
        json.dump(dataset_split_payload, f, indent=2, ensure_ascii=False)

    # Detailed subset metadata
    subset_metadata = {
        "vocabulary_size": len(FINAL_VOCABULARY),
        "vocabulary": FINAL_VOCABULARY,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "training_cap": CAP,
        "train_samples": len(selected_train),
        "validation_samples": len(selected_val),
        "test_samples": len(selected_test),
        "total_samples": len(selected_train) + len(selected_val) + len(selected_test),
        "source_split": "outputs/dataset_split.json",
        "random_seed": SEED_BASE,
        "seed_strategy": "deterministic per-class: random.Random(42 + class_index) over sorted path list",
        "validation_modified": False,
        "test_modified": False,
        "augmentation_applied": False,
        "oversampling_applied": False,
        "synthetic_data": False,
        "per_class_summary": per_class_summary,
        "class_weights": class_weights
    }

    meta_json_path = meta_dir / "subset_metadata.json"
    with open(meta_json_path, "w", encoding="utf-8") as f:
        json.dump(subset_metadata, f, indent=2, ensure_ascii=False)

    print(f"\nSuccessfully generated subset files:")
    print(f"  - {split_meta_path}")
    print(f"  - {meta_json_path}")
    print(f"  - {train_dir / 'train_paths.txt'} ({len(selected_train)} paths)")
    print(f"  - {val_dir / 'val_paths.txt'} ({len(selected_val)} paths)")
    print(f"  - {test_dir / 'test_paths.txt'} ({len(selected_test)} paths)")

if __name__ == "__main__":
    prepare_subset()
