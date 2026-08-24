"""
Unit tests for the PyTorch dataset preparation layer.

Uses small synthetic .npz fixtures in a temporary directory so the real
dataset is never written to. Run with:

    python -m unittest discover -s tests -v
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "data"))

from azsl_dataset import (  # noqa: E402
    EXPECTED_SHAPE,
    FeatureSample,
    build_class_mapping,
    compute_class_weights,
    make_dataloaders,
    scan_feature_files,
    stratified_split,
    validate_features,
    AzslFeatureDataset,
)


def write_npz(path: Path, shape=EXPECTED_SHAPE, extra=True):
    """Create a synthetic feature .npz mirroring the real structure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"features": np.random.rand(*shape).astype(np.float32)}
    if extra:
        payload["mask"] = np.ones(shape[0], dtype=np.float32)
        payload["frame_features"] = np.random.rand(42, 126).astype(np.float32)
        payload["frame_valid"] = np.ones(42, dtype=bool)
    np.savez(path, **payload)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

        # 3 classes (incl. Unicode name); counts chosen to exercise splits.
        layout = {"SALAM": 5, "ALMAQ": 12, "ZƏHMƏT OLMASA": 2}
        self.paths = []
        for cls, n in layout.items():
            for i in range(n):
                p = self.root / cls / f"{cls}_{i:02d}.npz"
                write_npz(p)
                self.paths.append(p)

        self.class_to_idx, self.idx_to_class = build_class_mapping(self.root)
        self.samples = scan_feature_files(self.root, self.class_to_idx)
        self.split = stratified_split(self.samples, seed=42)

    def tearDown(self):
        self.tmp.cleanup()

    # --- dataset length / sample shape / label range -----------------------
    def test_dataset_length(self):
        ds = AzslFeatureDataset(self.samples)
        self.assertEqual(len(ds), 19)
        self.assertEqual(len(self.samples), 5 + 12 + 2)

    def test_sample_shape_and_dtype(self):
        ds = AzslFeatureDataset(self.samples)
        x, y = ds[0]
        self.assertEqual(tuple(x.shape), EXPECTED_SHAPE)
        self.assertEqual(x.dtype, torch.float32)
        self.assertIsInstance(y, int)

    def test_label_range(self):
        labels = [s.label for s in self.samples]
        self.assertTrue(all(0 <= l < len(self.idx_to_class) for l in labels))
        self.assertEqual(set(labels), set(range(len(self.idx_to_class))))

    # --- split integrity ----------------------------------------------------
    def test_no_overlap_between_splits(self):
        t, v, s = set(self.split.train), set(self.split.val), set(self.split.test)
        self.assertFalse(t & v)
        self.assertFalse(t & s)
        self.assertFalse(v & s)
        self.assertEqual(len(t | v | s), len(self.samples))

    def test_split_is_deterministic(self):
        again = stratified_split(self.samples, seed=42)
        self.assertEqual(self.split.train, again.train)
        self.assertEqual(self.split.val, again.val)
        self.assertEqual(self.split.test, again.test)

    def test_small_class_handled_safely(self):
        # 'ZƏHMƏT OLMASA' has 2 samples -> cannot cover all three splits.
        self.assertIn("ZƏHMƏT OLMASA", self.split.incomplete_coverage)

    # --- class mapping consistency -----------------------------------------
    def test_class_mapping_consistency(self):
        self.assertEqual(len(self.idx_to_class), len(set(self.idx_to_class)))
        for name, idx in self.class_to_idx.items():
            self.assertEqual(self.idx_to_class[idx], name)
        # Unicode names preserved exactly
        self.assertIn("ZƏHMƏT OLMASA", self.class_to_idx)
        self.assertEqual(sorted(self.idx_to_class), self.idx_to_class)

    # --- validation catches malformed data ----------------------------------
    def test_validation_detects_bad_shape_and_nan(self):
        bad_shape = self.root / "SALAM" / "bad.npz"
        np.savez(bad_shape, features=np.random.rand(10, 126).astype(np.float32))
        nan_file = self.root / "SALAM" / "nan.npz"
        feats = np.random.rand(26, 126).astype(np.float32)
        feats[0, 0] = np.nan
        np.savez(nan_file, features=feats)

        samples = scan_feature_files(self.root, self.class_to_idx)
        result = validate_features(samples, self.idx_to_class)
        self.assertFalse(result.ok)
        self.assertEqual(len(result.shape_errors), 1)
        self.assertEqual(len(result.non_finite_files), 1)

    def test_weights_from_train_only(self):
        train_labels = [self.samples[i].label for i in self.split.train]
        w = compute_class_weights(train_labels, num_classes=len(self.idx_to_class))
        self.assertEqual(w.dtype, torch.float32)
        c = len(self.idx_to_class)
        n = len(train_labels)
        expected = n / (c * train_labels.count(0))  # weight of first class
        self.assertAlmostEqual(w[0].item(), expected, places=4)

    # --- DataLoader -----------------------------------------------------------
    def test_dataloader_batch_shapes(self):
        loaders = make_dataloaders(self.split, self.samples, batch_size=4, pin_memory=False)
        names = ["train", "val", "test"]
        sizes = [len(self.split.train), len(self.split.val), len(self.split.test)]
        for loader, size in zip(loaders, sizes):
            x, y = next(iter(loader))
            self.assertEqual(tuple(x.shape)[1:], EXPECTED_SHAPE)
            self.assertLessEqual(x.shape[0], 4)
            self.assertEqual(y.shape[0], x.shape[0])
        del loaders, names


if __name__ == "__main__":
    unittest.main()
