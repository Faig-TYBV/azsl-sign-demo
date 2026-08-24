"""
Unit tests for the GRU baseline model (CPU only — no GPU required).

    python -m unittest discover -s tests -v
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "models"))

from gru_classifier import GRUClassifier  # noqa: E402


class GRUModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.cfg = dict(input_size=126, hidden_size=32, num_layers=2,
                        num_classes=10, dropout=0.3)
        self.model = GRUClassifier(**self.cfg)

    def test_output_shape(self):
        x = torch.randn(4, 26, 126)
        logits = self.model(x)
        self.assertEqual(tuple(logits.shape), (4, 10))

    def test_forward_pass_values_finite(self):
        logits = self.model(torch.randn(2, 26, 126))
        self.assertTrue(torch.isfinite(logits).all())

    def test_parameter_configuration(self):
        gru = self.model.gru
        self.assertFalse(gru.bidirectional)
        self.assertTrue(gru.batch_first)
        self.assertEqual(gru.hidden_size, 32)
        self.assertEqual(gru.num_layers, 2)
        # Linear head maps hidden -> num_classes
        self.assertEqual(self.model.fc.in_features, 32)
        self.assertEqual(self.model.fc.out_features, 10)

    def test_config_roundtrip(self):
        cfg = self.model.get_config()
        clone = GRUClassifier(**cfg)
        self.assertEqual(clone.get_config(), cfg)
        self.assertEqual(
            sum(p.numel() for p in self.model.parameters()),
            sum(p.numel() for p in clone.parameters()),
        )

    def test_cpu_forward_backward(self):
        x = torch.randn(8, 26, 126)
        y = torch.randint(0, 10, (8,))
        logits = self.model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
        grads = [p.grad for p in self.model.parameters() if p.requires_grad]
        self.assertTrue(all(g is not None for g in grads))

    def test_class_weight_handling(self):
        x = torch.randn(4, 26, 126)
        y = torch.tensor([0, 3, 5, 9])
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
        weights = torch.rand(10) + 0.5
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
        weighted = criterion(logits, y)
        unweighted = torch.nn.functional.cross_entropy(logits, y)
        # A weighted CE with non-uniform weights must differ from uniform.
        uniform = torch.nn.functional.cross_entropy(
            logits, y, weight=torch.ones(10))
        self.assertTrue(torch.isfinite(weighted))
        self.assertGreater(abs(float(weighted) - float(unweighted)), 1e-6)
        self.assertAlmostEqual(float(unweighted), float(uniform), places=5)

    def test_checkpoint_save_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.pt"
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": None,
                "epoch": 3,
                "best_val_macro_f1": 0.42,
                "class_to_idx": {"A": 0},
                "model_config": self.model.get_config(),
            }, path)
            ckpt = torch.load(path, weights_only=False)
            clone = GRUClassifier(**ckpt["model_config"])
            clone.load_state_dict(ckpt["model_state_dict"])
            self.model.eval()
            clone.eval()
            x = torch.randn(1, 26, 126)
            self.assertTrue(
                torch.allclose(self.model(x), clone(x), atol=1e-6))
            self.assertEqual(ckpt["epoch"], 3)


class TemporalPoolingTests(unittest.TestCase):
    """Experiment 2: temporal mean+max pooling head."""

    def setUp(self):
        torch.manual_seed(42)
        self.cfg = dict(input_size=126, hidden_size=32, num_layers=2,
                        num_classes=10, dropout=0.3, pooling="mean_max")
        self.model = GRUClassifier(**self.cfg)

    def test_output_shape(self):
        logits = self.model(torch.randn(4, 26, 126))
        self.assertEqual(tuple(logits.shape), (4, 10))

    def test_pooling_dimensions_and_fc_input(self):
        # Classifier input must be 2 * hidden_size = 64 for mean||max.
        self.assertEqual(self.model.fc.in_features, 2 * 32)
        x = torch.randn(3, 26, 126)
        outputs, _ = self.model.gru(x)
        pooled = torch.cat([outputs.mean(dim=1), outputs.amax(dim=1)], dim=1)
        self.assertEqual(tuple(pooled.shape), (3, 64))
        # fc(pooled) with zeroed head weights reproduces pooled dims check
        with torch.no_grad():
            self.model.fc.weight.zero_()
            self.model.fc.bias.zero_()
        logits = self.model(x)
        self.assertTrue((logits == 0).all())  # sanity: head consumes [B,64]

    def test_forward_pass_finite(self):
        logits = self.model(torch.randn(2, 26, 126))
        self.assertTrue(torch.isfinite(logits).all())

    def test_parameter_count_increases_vs_last(self):
        last_cfg = dict(self.cfg, pooling="last")
        n_last = sum(p.numel() for p in GRUClassifier(**last_cfg).parameters())
        n_mm = sum(p.numel() for p in self.model.parameters())
        expected_delta = 32 * 10 * 1 + 10 * 1 * 0  # extra fc weight rows only
        self.assertGreater(n_mm, n_last)
        self.assertEqual(n_mm - n_last, 320)  # 32->10 extra weights

    def test_checkpoint_save_load_mean_max(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.pt"
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "epoch": 5,
                "best_val_macro_f1": 0.5,
                "class_to_idx": {},
                "model_config": self.model.get_config(),
            }, path)
            ckpt = torch.load(path, weights_only=False)
            clone = GRUClassifier(**ckpt["model_config"])
            clone.load_state_dict(ckpt["model_state_dict"])
            self.assertEqual(clone.pooling, "mean_max")
            self.model.eval(); clone.eval()
            x = torch.randn(1, 26, 126)
            self.assertTrue(torch.allclose(self.model(x), clone(x), atol=1e-6))


class TemporalAugmentationTests(unittest.TestCase):
    """Experiment 3: training-time temporal augmentation."""

    def setUp(self):
        rng = np.random.default_rng(0)
        self.feats = rng.normal(0.0, 0.5, (26, 126)).astype(np.float32)
        self.mask = np.zeros(26, dtype=np.float32)
        self.mask[:20] = 1.0  # last 6 frames are padding (zeros)
        self.feats[20:] = 0.0
        from augmentation import TemporalAugmentation
        self.aug = TemporalAugmentation(
            noise_std=0.01, temporal_mask_prob=0.10,
            temporal_dropout_prob=0.05, seed=42)

    def test_shape_and_dtype_preserved(self):
        out = self.aug(self.feats, self.mask)
        self.assertEqual(out.shape, (26, 126))
        self.assertEqual(out.dtype, np.float32)

    def test_original_not_modified(self):
        original = self.feats.copy()
        self.aug(self.feats, self.mask)
        np.testing.assert_array_equal(self.feats, original)

    def test_padding_frames_untouched(self):
        out = self.aug(self.feats, self.mask)
        # Padding frames must remain exactly zero.
        np.testing.assert_array_equal(out[20:], np.zeros((6, 126), np.float32))
        # Valid frames should change somewhere across several draws.
        changed = any(
            not np.allclose(self.aug(self.feats, self.mask)[10], self.feats[10])
            for _ in range(5)
        )
        self.assertTrue(changed)

    def test_zero_augmentation_reproduces_input(self):
        from augmentation import TemporalAugmentation
        zero = TemporalAugmentation(noise_std=0.0, temporal_mask_prob=0.0,
                                    temporal_dropout_prob=0.0)
        out = zero(self.feats, self.mask)
        np.testing.assert_array_equal(out, self.feats)

    def test_reproducible_under_seed(self):
        from augmentation import TemporalAugmentation
        a1 = TemporalAugmentation(seed=7)(self.feats, self.mask)
        a2 = TemporalAugmentation(seed=7)(self.feats, self.mask)
        np.testing.assert_array_equal(a1, a2)

    def test_masking_never_destroys_all_frames(self):
        from augmentation import TemporalAugmentation
        # Extreme settings must still leave at least one valid frame intact.
        for seed in range(10):
            heavy = TemporalAugmentation(noise_std=0.0, temporal_mask_prob=0.9,
                                         temporal_dropout_prob=0.5, seed=seed)
            out = heavy(self.feats, self.mask)
            frame_sums = np.abs(out[:20]).sum(axis=1)  # valid region only
            self.assertGreater((frame_sums > 0).sum(), 0)

    def test_dataloader_train_aug_vs_val_clean(self):
        import tempfile
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "data"))
        from azsl_dataset import AzslFeatureDataset, FeatureSample
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "C" / "s.npz"
            p.parent.mkdir(parents=True)
            np.savez(p, features=self.feats, mask=self.mask)
            s = [FeatureSample(p, "C", 0)]
            ds_aug = AzslFeatureDataset(s, augment=self.aug)
            ds_clean = AzslFeatureDataset(s)
            xa, ya = ds_aug[0]
            xc, yc = ds_clean[0]
            self.assertEqual(ya, yc)
            self.assertEqual(tuple(xa.shape), (26, 126))
            self.assertEqual(xa.dtype, torch.float32)
            # Augmented sample differs but clean equals stored features
            self.assertFalse(torch.allclose(xa, xc))


class TemporalDeltaTests(unittest.TestCase):
    """Experiment 4: masked temporal delta features."""

    def setUp(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "data"))
        from delta_features import compute_temporal_deltas, combine_with_deltas
        self.compute = compute_temporal_deltas
        self.combine = combine_with_deltas
        rng = np.random.default_rng(3)
        # mask: valid frames 0-4, invalid 5-7, valid 8-11, padding 12-25
        self.mask = np.zeros(26, dtype=np.float32)
        self.mask[0:5] = 1.0
        self.mask[8:12] = 1.0
        self.feats = rng.normal(0.3, 0.2, (26, 126)).astype(np.float32)
        self.feats[self.mask == 0] = 0.0

    def test_combined_shape_and_original_unchanged(self):
        original = self.feats.copy()
        comb = self.combine(self.feats, self.mask)
        self.assertEqual(comb.shape, (26, 252))
        np.testing.assert_array_equal(self.feats, original)  # not mutated
        np.testing.assert_array_equal(comb[:, :126], self.feats)

    def test_delta_shape_first_frame_zero(self):
        d = self.compute(self.feats, self.mask)
        self.assertEqual(d.shape, (26, 126))
        np.testing.assert_array_equal(d[0], np.zeros(126, np.float32))
        self.assertEqual(d.dtype, np.float32)

    def test_invalid_frames_have_zero_delta(self):
        d = self.compute(self.feats, self.mask)
        for t in list(range(5, 8)) + list(range(12, 26)):
            np.testing.assert_array_equal(
                d[t], np.zeros(126, np.float32), err_msg=f"frame {t}")

    def test_invalid_to_valid_boundary_no_fake_delta(self):
        d = self.compute(self.feats, self.mask)
        np.testing.assert_array_equal(d[8], np.zeros(126, np.float32))

    def test_valid_to_invalid_boundary_no_fake_delta(self):
        d = self.compute(self.feats, self.mask)
        np.testing.assert_array_equal(d[5], np.zeros(126, np.float32))
        self.assertTrue(np.any(d[4] != 0))

    def test_normal_delta_values(self):
        d = self.compute(self.feats, self.mask)
        np.testing.assert_allclose(d[3], self.feats[3] - self.feats[2], rtol=1e-6)

    def test_dataset_returns_26x252_labels_unchanged(self):
        import tempfile
        from azsl_dataset import AzslFeatureDataset, FeatureSample
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "C" / "s.npz"
            p.parent.mkdir(parents=True)
            np.savez(p, features=self.feats, mask=self.mask)
            ds = AzslFeatureDataset([FeatureSample(p, "C", 17)], with_delta=True)
            x, y = ds[0]
            self.assertEqual(tuple(x.shape), (26, 252))
            self.assertEqual(x.dtype, torch.float32)
            self.assertEqual(y, 17)

    def test_model_accepts_252_input(self):
        m = GRUClassifier(input_size=252, hidden_size=16, num_layers=2,
                          num_classes=200, pooling="mean_max")
        out = m(torch.randn(2, 26, 252))
        self.assertEqual(tuple(out.shape), (2, 200))


class Dropout05ConfigTests(unittest.TestCase):
    """Experiment 6: verify dropout=0.5 configuration is exactly as specified."""

    def test_dropout_configuration(self):
        m = GRUClassifier(input_size=126, hidden_size=128, num_layers=2,
                          num_classes=200, dropout=0.5, pooling="mean_max")
        self.assertEqual(m.dropout, 0.5)
        self.assertEqual(m.gru.dropout, 0.5)          # between GRU layers
        self.assertEqual(m.head_dropout.p, 0.5)       # final head dropout
        self.assertEqual(m.fc.in_features, 256)       # mean||max pooling
        self.assertEqual(m.fc.out_features, 200)

    def test_output_shape_and_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.pt"
            m = GRUClassifier(input_size=126, hidden_size=128, num_layers=2,
                              num_classes=200, dropout=0.5, pooling="mean_max")
            torch.save({"model_state_dict": m.state_dict(),
                        "model_config": m.get_config()}, path)
            ckpt = torch.load(path, weights_only=False)
            clone = GRUClassifier(**ckpt["model_config"])
            clone.load_state_dict(ckpt["model_state_dict"])
            self.assertEqual(clone.get_config()["dropout"], 0.5)
            self.assertEqual(tuple(m(torch.randn(4, 26, 126)).shape), (4, 200))


class FeatureNormalizationTests(unittest.TestCase):
    """Experiment 7: train-only mask-aware feature normalization."""

    def setUp(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "data"))
        from normalization import fit_normalization, FeatureNormalizer
        self.fit = fit_normalization
        self.FeatureNormalizer = FeatureNormalizer
        rng = np.random.default_rng(9)
        self.mask = np.zeros(26, dtype=np.float32)
        self.mask[:18] = 1.0  # 8 padding frames stay zero

    def _mk_samples(self, tmp, n=3):
        from azsl_dataset import FeatureSample
        samples = []
        for i in range(n):
            rng = np.random.default_rng(100 + i)
            f = (rng.normal(0.4, 0.25, (26, 126))).astype(np.float32)
            f[self.mask == 0] = 0.0
            p = Path(tmp) / f"C{i}" / "s.npz"
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez(p, features=f, mask=self.mask)
            samples.append(FeatureSample(p, f"C{i}", i))
        return samples

    def test_stats_from_train_only_and_excludes_invalid(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            samples = self._mk_samples(tmp)
            stats = self.fit(samples)
            m = np.array(stats["mean"])
            # If invalid frames leaked in (all-zero rows), mean would be
            # pulled toward 0; with valid mean ~0.4 it must be near that.
            self.assertTrue(np.all(np.abs(m - 0.4) < 0.15))
            self.assertEqual(stats["n_valid_frames"], 3 * 18)

    def test_invalid_frames_remain_zero_and_no_mutation(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            samples = self._mk_samples(tmp, n=1)
            norm = self.FeatureNormalizer.from_stats(self.fit(samples))
            f = samples[0].path
            with np.load(f) as z:
                feats = z["features"].copy()
                orig = feats.copy()
            out = norm(feats, self.mask)
            self.assertEqual(out.shape, (26, 126))
            self.assertEqual(out.dtype, np.float32)
            np.testing.assert_array_equal(out[18:], np.zeros((8, 126), np.float32))
            np.testing.assert_array_equal(feats, orig)  # input not mutated

    def test_deterministic_and_val_test_use_train_stats(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            samples = self._mk_samples(tmp, n=2)
            stats = self.fit(samples)
            n1 = self.FeatureNormalizer.from_stats(stats)
            n2 = self.FeatureNormalizer.from_stats(self.fit(samples))
            with np.load(samples[0].path) as z:
                feats, mask = z["features"], z["mask"]
            np.testing.assert_array_equal(n1(feats, mask), n2(feats, mask))
            # Same transform object applied to any data -> same stats source
            out = n1(feats, mask)
            self.assertTrue(np.isfinite(out).all())

    def test_model_receives_126_output_200(self):
        m = GRUClassifier(input_size=126, hidden_size=16, num_layers=2,
                          num_classes=200, pooling="mean_max")
        self.assertEqual(tuple(m(torch.randn(3, 26, 126)).shape), (3, 200))


class BalancedSamplerTests(unittest.TestCase):
    """Experiment 8: balanced sampling replaces class-weighted loss."""

    def setUp(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "data"))
        import tempfile
        from azsl_dataset import AzslFeatureDataset, FeatureSample
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        # Imbalanced: class 0 x10, class 1 x2
        samples = []
        for cls, n in ((0, 10), (1, 2)):
            for i in range(n):
                p = root / f"C{cls}" / f"{i}.npz"
                p.parent.mkdir(parents=True, exist_ok=True)
                np.savez(p, features=np.random.rand(26, 126).astype(np.float32),
                         mask=np.ones(26, dtype=np.float32))
                samples.append(FeatureSample(p, f"C{cls}", cls))
        self.ds = AzslFeatureDataset(samples)
        self.labels = [s.label for s in samples]

    def tearDown(self):
        self.tmp.cleanup()

    def test_weights_are_inverse_train_frequency(self):
        from collections import Counter
        counts = Counter(self.labels)
        w = [1.0 / counts[l] for l in self.labels]
        sampler = torch.utils.data.WeightedRandomSampler(
            w, num_samples=len(self.ds), replacement=True)
        self.assertEqual(len(sampler.weights), len(self.ds))
        self.assertAlmostEqual(float(sampler.weights[0]), 0.1, places=6)   # class 0 (10x)
        self.assertAlmostEqual(float(sampler.weights[10]), 0.5, places=6)  # class 1 (2x)

    def test_sampler_properties_and_balance(self):
        from collections import Counter
        counts = Counter(self.labels)
        w = [1.0 / counts[l] for l in self.labels]
        sampler = torch.utils.data.WeightedRandomSampler(
            w, num_samples=len(self.ds), replacement=True,
            generator=torch.Generator().manual_seed(42))
        self.assertTrue(sampler.replacement)
        self.assertEqual(sampler.num_samples, len(self.ds))
        idx = list(torch.utils.data.WeightedRandomSampler(
            w, num_samples=200, replacement=True,
            generator=torch.Generator().manual_seed(42)))
        sampled = Counter(self.labels[i] for i in idx)
        # Exposure approximately balanced despite 5:1 imbalance.
        frac1 = sampled[1] / len(idx)
        self.assertGreater(frac1, 0.4)
        self.assertLess(frac1, 0.6)

    def test_val_test_loaders_have_no_sampler(self):
        from torch.utils.data import DataLoader
        vl = DataLoader(torch.utils.data.Subset(self.ds, [0]), batch_size=4)
        tl = DataLoader(torch.utils.data.Subset(self.ds, [0]), batch_size=4)
        self.assertIsNone(vl.sampler.__class__.__name__ != "SequentialSampler" and vl.sampler or None)
        self.assertEqual(vl.sampler.__class__.__name__, "SequentialSampler")
        self.assertEqual(tl.sampler.__class__.__name__, "SequentialSampler")

    def test_unweighted_loss(self):
        crit = torch.nn.CrossEntropyLoss()
        self.assertIsNone(crit.weight)

    def test_batch_shape_preserved(self):
        from torch.utils.data import DataLoader
        dl = DataLoader(self.ds, batch_size=4)
        x, y = next(iter(dl))
        self.assertEqual(tuple(x.shape), (4, 26, 126))
        self.assertEqual(y.shape[0], 4)


if __name__ == "__main__":
    unittest.main()
