"""
PyTorch dataset pipeline for the extracted AzSLD_Words_200 features.

Loads per-video .npz feature files laid out as:

    data/features/full/<CLASS_NAME>/<video_stem>.npz

where each .npz contains:
    features      : float32 (26, 126)   <- model input (only array loaded here)
    mask          : float32 (26,)
    frame_features: float32 (42, 126)   <- NOT loaded by this module
    frame_valid   : bool    (42,)

Provides:
    - build_class_mapping()      : deterministic class <-> index mapping
    - scan_feature_files()       : discover .npz files + labels
    - validate_features()        : integrity checks (count / shape / finite)
    - stratified_split()         : deterministic 70/15/15 per-class split
    - compute_class_weights()    : balanced weights from TRAIN split only
    - AzslFeatureDataset         : torch.utils.data.Dataset
    - make_dataloaders()         : train/val/test DataLoaders

Usage (validation entry point):
    python src/data/prepare_dataset.py
"""

import json
import logging
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
DEFAULT_FEATURES_ROOT = Path("data/features/full")
DEFAULT_METADATA_PATH = Path("outputs/dataset_split.json")

EXPECTED_SHAPE: Tuple[int, int] = (26, 126)

SPLIT_SEED: int = 42
VAL_FRACTION: float = 0.15
TEST_FRACTION: float = 0.15

DEFAULT_BATCH_SIZE: int = 32
# Windows: default to 0 workers (no multiprocessing) to avoid
# spawn/pickling issues; keep configurable at the call site.
DEFAULT_NUM_WORKERS: int = 0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("azsl_dataset")


# ---------------------------------------------------------------------------
# Class mapping
# ---------------------------------------------------------------------------
def build_class_mapping(features_root: Path) -> Tuple[Dict[str, int], List[str]]:
    """Build a deterministic class <-> integer index mapping.

    Classes are discovered from the immediate sub-directories of
    ``features_root`` and ordered by Python's default string sort
    (Unicode code-point order), which is stable across runs/platforms
    for the same set of names.

    Returns:
        (class_to_idx, idx_to_class)
    """
    features_root = Path(features_root)
    class_names = sorted(d.name for d in features_root.iterdir() if d.is_dir())
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    idx_to_class = class_names
    return class_to_idx, idx_to_class


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FeatureSample:
    """One .npz feature file plus its derived class label/index."""
    path: Path
    class_name: str
    label: int


def scan_feature_files(
    features_root: Path,
    class_to_idx: Dict[str, int],
) -> List[FeatureSample]:
    """Discover all .npz files under ``features_root``.

    The class name is derived from the immediate parent directory of each
    file. Directories not present in ``class_to_idx`` are reported and skipped.
    """
    features_root = Path(features_root)
    samples: List[FeatureSample] = []

    for class_dir in sorted(p for p in features_root.iterdir() if p.is_dir()):
        label = class_to_idx.get(class_dir.name)
        if label is None:
            log.warning("Unknown class directory (not in mapping): %s", class_dir.name)
            continue
        for npz_path in sorted(class_dir.glob("*.npz")):
            samples.append(FeatureSample(npz_path, class_dir.name, label))

    return samples


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
@dataclass
class ValidationResult:
    """Outcome of the dataset integrity checks."""
    total_files: int = 0
    total_classes: int = 0
    shape_errors: List[str] = field(default_factory=list)
    malformed_files: List[str] = field(default_factory=list)
    non_finite_files: List[str] = field(default_factory=list)
    missing_labels: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (
            self.shape_errors
            or self.malformed_files
            or self.non_finite_files
            or self.missing_labels
        )


def validate_features(
    samples: Sequence[FeatureSample],
    idx_to_class: Sequence[str],
) -> ValidationResult:
    """Validate every .npz file (read-only). Failures are REPORTED, never fixed.

    Checks:
      * file loads as a valid .npz containing a 'features' array
      * 'features' has shape (26, 126)
      * no NaN / Inf values
      * every class index is represented by at least one sample
    """
    result = ValidationResult()
    result.total_files = len(samples)
    result.total_classes = len(idx_to_class)

    seen_labels = set()
    for sample in samples:
        try:
            with np.load(sample.path) as data:
                if "features" not in data:
                    result.malformed_files.append(str(sample.path))
                    continue
                feats = data["features"]
        except Exception as exc:  # corrupted / unreadable archive
            result.malformed_files.append(f"{sample.path} ({exc})")
            continue

        if feats.shape != EXPECTED_SHAPE:
            result.shape_errors.append(
                f"{sample.path}: shape {feats.shape}, expected {EXPECTED_SHAPE}"
            )
        if not np.isfinite(feats).all():
            result.non_finite_files.append(str(sample.path))

        seen_labels.add(sample.label)

    result.missing_labels = [
        f"{i}:{idx_to_class[i]}"
        for i in range(len(idx_to_class))
        if i not in seen_labels
    ]
    return result


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------
@dataclass
class SplitResult:
    """Deterministic stratified 70/15/15 split (by feature FILE)."""
    train: List[int] = field(default_factory=list)
    val: List[int] = field(default_factory=list)
    test: List[int] = field(default_factory=list)
    # Classes that could not have samples in all three splits.
    incomplete_coverage: List[str] = field(default_factory=list)

    def get(self, name: str) -> List[int]:
        return getattr(self, name)


def _split_counts(n: int) -> Tuple[int, int, int]:
    """Allocate (train, val, test) counts for a class with ``n`` samples.

    Guarantees >= 1 sample in every split whenever n >= 3.
    For n < 3, assigns greedily: 1 -> train only; 2 -> train + test.
    """
    if n < 3:
        if n == 1:
            return 1, 0, 0
        return 1, 0, 1
    n_val = max(1, round(n * VAL_FRACTION))
    n_test = max(1, round(n * TEST_FRACTION))
    n_train = n - n_val - n_test
    while n_train < 1:  # shrink val/test until train has >= 1
        if n_val > n_test and n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            break
    return n_train, n_val, n_test


def stratified_split(
    samples: Sequence[FeatureSample],
    seed: int = SPLIT_SEED,
) -> SplitResult:
    """Split sample indices into train/val/test, stratified per class.

    Splitting is done per FEATURE FILE (video), never per frame. Within each
    class, files are shuffled with a seeded ``random.Random``, making the
    split fully reproducible. Each file appears in exactly one split.
    """
    rng = random.Random(seed)
    result = SplitResult()

    by_class: Dict[str, List[int]] = {}
    for idx, sample in enumerate(samples):
        by_class.setdefault(sample.class_name, []).append(idx)

    for class_name in sorted(by_class):
        indices = sorted(by_class[class_name])
        rng.shuffle(indices)  # deterministic given seed + insertion order
        n_train, n_val, n_test = _split_counts(len(indices))
        result.train.extend(indices[:n_train])
        result.val.extend(indices[n_train:n_train + n_val])
        result.test.extend(indices[n_train + n_val:])
        if n_val == 0 or n_test == 0:
            result.incomplete_coverage.append(class_name)

    return result


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------
def compute_class_weights(
    train_labels: Sequence[int],
    num_classes: int,
) -> torch.Tensor:
    """Balanced class weights from the TRAINING split only.

    weight_i = N / (C * count_i)

    Returns a float32 tensor of shape (num_classes,). Classes absent from
    the training split get weight 0.0 (reported upstream instead of fixed).
    """
    counts = Counter(train_labels)
    n_total = len(train_labels)
    weights = torch.zeros(num_classes, dtype=torch.float32)
    for cls_idx in range(num_classes):
        count = counts.get(cls_idx, 0)
        if count > 0:
            weights[cls_idx] = n_total / (num_classes * count)
    return weights


# ---------------------------------------------------------------------------
# Dataset / DataLoaders
# ---------------------------------------------------------------------------
class AzslFeatureDataset(Dataset):
    """PyTorch Dataset over per-video .npz feature files.

    Each __getitem__ loads ONLY the 'features' array (26, 126) from disk,
    verifies its shape, and returns ``(float32_tensor, int_label)``.
    The .npz files are opened read-only and never modified.
    """

    def __init__(self, samples: Sequence[FeatureSample], augment=None,
                 with_delta: bool = False):
        """``augment``: optional callable (features, mask) -> augmented
        features. Pass a TemporalAugmentation for the TRAINING split only;
        leave None (default) for validation/test so they stay untouched.
        ``with_delta``: concatenate masked temporal deltas -> [26, 252].
        """
        self.samples: List[FeatureSample] = list(samples)
        self.augment = augment
        self.with_delta = with_delta

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        with np.load(sample.path) as data:
            features = data["features"]
            if self.augment is not None:
                features = self.augment(features, data["mask"])
            if self.with_delta:
                from delta_features import combine_with_deltas
                features = combine_with_deltas(features, data["mask"])
        expected = EXPECTED_SHAPE if not self.with_delta else (
            EXPECTED_SHAPE[0], EXPECTED_SHAPE[1] * 2)
        if features.shape != expected:
            raise ValueError(
                f"{sample.path}: shape {features.shape}, expected {expected}"
            )
        tensor = torch.from_numpy(np.ascontiguousarray(features)).float()
        return tensor, int(sample.label)


def make_dataloaders(
    split: SplitResult,
    samples: Sequence[FeatureSample],
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    pin_memory: Optional[bool] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders.

    pin_memory defaults to CUDA availability when left as None.
    """
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    train_ds = AzslFeatureDataset([samples[i] for i in split.train])
    val_ds = AzslFeatureDataset([samples[i] for i in split.val])
    test_ds = AzslFeatureDataset([samples[i] for i in split.test])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Metadata export
# ---------------------------------------------------------------------------
def save_split_metadata(
    metadata_path: Path,
    samples: Sequence[FeatureSample],
    idx_to_class: Sequence[str],
    split: SplitResult,
    class_weights: torch.Tensor,
) -> None:
    """Persist class mapping, split assignment, and weights as JSON.

    Only file paths / indices / small numbers are written — never feature
    arrays. Paths are stored POSIX-relative to the project root.
    """
    payload = {
        "seed": SPLIT_SEED,
        "expected_shape": list(EXPECTED_SHAPE),
        "class_to_idx": {name: i for i, name in enumerate(idx_to_class)},
        "idx_to_class": list(idx_to_class),
        "class_weights": [round(float(w), 6) for w in class_weights.tolist()],
        "splits": {
            "train": [samples[i].path.as_posix() for i in split.train],
            "val": [samples[i].path.as_posix() for i in split.val],
            "test": [samples[i].path.as_posix() for i in split.test],
        },
        "incomplete_split_coverage_classes": split.incomplete_coverage,
        "counts": {
            "total": len(samples),
            "train": len(split.train),
            "val": len(split.val),
            "test": len(split.test),
        },
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log.info("Saved split metadata to %s", metadata_path)


def load_split_metadata(
    metadata_path: Path,
    features_root: Path = DEFAULT_FEATURES_ROOT,
) -> dict:
    """Reconstruct the saved split WITHOUT re-randomizing.

    Reads outputs/dataset_split.json and rebuilds per-split FeatureSample
    lists by matching the stored relative paths against files on disk.
    Returns a dict with keys: idx_to_class, class_to_idx, class_weights,
    splits (name -> list[FeatureSample]).
    """
    metadata_path = Path(metadata_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    idx_to_class: List[str] = payload["idx_to_class"]
    class_to_idx: Dict[str, int] = payload["class_to_idx"]
    weights = torch.tensor(payload["class_weights"], dtype=torch.float32)

    features_root = Path(features_root)
    # Paths in the metadata are relative to the PROJECT ROOT / CWD
    # (e.g. data/features/full/<class>/<file>.npz).
    splits: Dict[str, List[FeatureSample]] = {}
    for name in ("train", "val", "test"):
        items: List[FeatureSample] = []
        for rel_path in payload["splits"][name]:
            abs_path = Path(rel_path)
            cls_name = abs_path.parent.name
            items.append(FeatureSample(abs_path, cls_name, class_to_idx[cls_name]))
        splits[name] = items

    return {
        "idx_to_class": idx_to_class,
        "class_to_idx": class_to_idx,
        "class_weights": weights,
        "splits": splits,
        "incomplete_split_coverage_classes": payload.get(
            "incomplete_split_coverage_classes", []
        ),
        "seed": payload.get("seed"),
    }


