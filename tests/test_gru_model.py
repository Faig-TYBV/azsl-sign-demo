"""
Unit tests for the GRU baseline model (CPU only — no GPU required).

    python -m unittest discover -s tests -v
"""

import sys
import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
