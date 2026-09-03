"""
Error analysis on the production checkpoint over the full test set.

Uses pre-extracted .npz features under data/features/full/ (the same
features the model was trained/validated on). This is ~100x faster than
re-running MediaPipe on every raw .mp4 and produces identical numerical
results to evaluate_dataset.py (since both paths feed the GRU the same
preprocessed 26x126 tensor).

Outputs (under outputs/error_analysis/):
  per_class_metrics.csv     - per-class precision/recall/F1/support
  confusion_pairs.csv       - off-diagonal pairs sorted by frequency
  zero_recall_classes.csv   - the classes with recall == 0
  class_freq_vs_f1.csv      - (train_freq, val_f1) per class
  summary.json              - top-level numbers + lists
  REPORT.md                 - human-readable report (sections a-d)

Run:
  .venv/Scripts/python.exe -m src.inference.error_analysis
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA_PATH    = PROJECT_ROOT / "outputs" / "dataset_split.json"
DEFAULT_CHECKPOINT_PATH  = PROJECT_ROOT / "outputs" / "checkpoints" / "gru_temporal_pool_best.pt"
DEFAULT_FEATURES_ROOT    = PROJECT_ROOT / "data" / "features" / "full"
DEFAULT_OUTPUT_DIR       = PROJECT_ROOT / "outputs" / "error_analysis"

from src.features.preprocess_sequence import preprocess_sequence  # noqa: E402
from src.inference import predict  # noqa: E402


def _load_split(metadata_path: Path, split: str) -> list[Path]:
    """Return absolute paths to .npz files for the given split."""
    with open(metadata_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return [PROJECT_ROOT / p for p in d["splits"][split]]


def _load_features(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the pre-extracted (frame_features, frame_valid) arrays."""
    with np.load(npz_path) as d:
        return d["frame_features"], d["frame_valid"]


def _class_to_train_freq(metadata: dict, idx_to_class: dict) -> dict[str, int]:
    """Count how many training-set samples exist for each class."""
    counts: dict[str, int] = defaultdict(int)
    for p in metadata["splits"]["train"]:
        cls = Path(p).parent.name
        counts[cls] += 1
    return counts

def main(
    metadata_path: Path = DEFAULT_METADATA_PATH,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    features_root: Path = DEFAULT_FEATURES_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Error analysis output -> {output_dir}")

    print(f"Loading model from {checkpoint_path.name} ...")
    model, class_to_idx, idx_to_class, device = predict.load_model(checkpoint_path)
    model.to(device).eval()

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    train_counts = _class_to_train_freq(metadata, idx_to_class)

    splits_to_run = {
        "test":  _load_split(metadata_path, "test"),
        "val":   _load_split(metadata_path, "val"),
    }

    summary: dict = {
        "checkpoint": str(checkpoint_path),
        "model_config": predict.EXPECTED_CONFIG,
        "class_to_idx_size": len(class_to_idx),
    }

    # We do the deepest analysis on the test set; we also compute per-class
    # F1 on the val set for the train-frequency-vs-performance correlation.
    for split_name, paths in splits_to_run.items():
        print(f"\n=== {split_name.upper()} set: {len(paths)} videos ===")
        all_true, all_pred, all_conf = [], [], []
        skipped = 0

        for i, npz_path in enumerate(paths):
            if not npz_path.exists():
                skipped += 1
                continue
            try:
                frame_feats, frame_valid = _load_features(npz_path)
            except Exception:
                skipped += 1
                continue

            # Run the same preprocess_sequence the model was trained with,
            # then a single forward pass on GPU/CPU.
            seq, _mask = preprocess_sequence(frame_feats, frame_valid)
            x = torch.from_numpy(seq).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1)[0]
            top1 = int(torch.argmax(probs).item())
            top1_conf = float(probs[top1].item())

            true_cls = npz_path.parent.name
            if true_cls not in class_to_idx:
                skipped += 1
                continue
            true_idx = class_to_idx[true_cls]
            all_true.append(true_idx)
            all_pred.append(top1)
            all_conf.append(top1_conf)

            if (i + 1) % 200 == 0:
                print(f"  processed {i + 1}/{len(paths)}")

        all_true = np.array(all_true)
        all_pred = np.array(all_pred)
        all_conf = np.array(all_conf)

        acc = float(accuracy_score(all_true, all_pred))
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
            all_true, all_pred, average="macro", zero_division=0
        )
        prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
            all_true, all_pred, average="weighted", zero_division=0
        )

        prec_pc, rec_pc, f1_pc, support_pc = precision_recall_fscore_support(
            all_true, all_pred, average=None, zero_division=0, labels=list(range(len(class_to_idx)))
        )
        cm = confusion_matrix(all_true, all_pred, labels=list(range(len(class_to_idx))))

        print(f"  accuracy = {acc:.4f}  macro_F1 = {f1_macro:.4f}  weighted_F1 = {f1_w:.4f}")
        print(f"  skipped (missing/unreadable) = {skipped}")

        summary[split_name] = {
            "n_evaluated": int(len(all_true)),
            "n_skipped": int(skipped),
            "accuracy": acc,
            "macro_precision": float(prec_macro),
            "macro_recall": float(rec_macro),
            "macro_f1": float(f1_macro),
            "weighted_precision": float(prec_w),
            "weighted_recall": float(rec_w),
            "weighted_f1": float(f1_w),
        }

        if split_name == "test":
            # ---- (a) Top confused class pairs ----
            cm_off = cm.copy().astype(int)
            np.fill_diagonal(cm_off, 0)
            top_pairs = []
            for i in range(cm_off.shape[0]):
                for j in range(cm_off.shape[1]):
                    if i == j or cm_off[i, j] == 0:
                        continue
                    top_pairs.append({
                        "true_class": idx_to_class[i],
                        "predicted_class": idx_to_class[j],
                        "count": int(cm_off[i, j]),
                        "train_freq_true": int(train_counts.get(idx_to_class[i], 0)),
                        "train_freq_pred": int(train_counts.get(idx_to_class[j], 0)),
                    })
            top_pairs.sort(key=lambda r: -r["count"])
            top_pairs = top_pairs[:50]
            summary["top_confused_pairs_test"] = top_pairs

            # ---- (c) Zero-recall classes ----
            zero_recall = []
            for idx in range(len(class_to_idx)):
                if rec_pc[idx] == 0.0 and support_pc[idx] > 0:
                    zero_recall.append({
                        "class": idx_to_class[idx],
                        "support": int(support_pc[idx]),
                        "precision": float(prec_pc[idx]),
                        "recall": 0.0,
                        "f1": float(f1_pc[idx]),
                        "train_freq": int(train_counts.get(idx_to_class[idx], 0)),
                    })
            zero_recall.sort(key=lambda r: -r["train_freq"])
            summary["zero_recall_test_count"] = len(zero_recall)
            summary["zero_recall_test"] = zero_recall

            # ---- (b) Per-class metrics distribution ----
            per_class = []
            for idx in range(len(class_to_idx)):
                per_class.append({
                    "class": idx_to_class[idx],
                    "precision": float(prec_pc[idx]),
                    "recall":    float(rec_pc[idx]),
                    "f1":        float(f1_pc[idx]),
                    "support":   int(support_pc[idx]),
                    "train_freq": int(train_counts.get(idx_to_class[idx], 0)),
                })
            summary["per_class_metrics_test_count"] = len(per_class)

            _save_per_class_csv(per_class, output_dir / "per_class_metrics.csv")
            _save_pairs_csv(top_pairs, output_dir / "confusion_pairs.csv")
            _save_zero_recall_csv(zero_recall, output_dir / "zero_recall_classes.csv")

            correct_mask = all_true == all_pred
            summary["confidence_test"] = {
                "n_correct":   int(correct_mask.sum()),
                "n_incorrect": int((~correct_mask).sum()),
                "mean_conf_correct":   float(all_conf[correct_mask].mean())   if correct_mask.any()   else 0.0,
                "median_conf_correct": float(np.median(all_conf[correct_mask])) if correct_mask.any() else 0.0,
                "mean_conf_incorrect": float(all_conf[~correct_mask].mean())  if (~correct_mask).any() else 0.0,
                "median_conf_incorrect": float(np.median(all_conf[~correct_mask])) if (~correct_mask).any() else 0.0,
            }

        if split_name == "val":
            # ---- (d) Class frequency vs performance correlation ----
            per_class_val = []
            for idx in range(len(class_to_idx)):
                per_class_val.append({
                    "class": idx_to_class[idx],
                    "precision_val": float(prec_pc[idx]),
                    "recall_val":    float(rec_pc[idx]),
                    "f1_val":        float(f1_pc[idx]),
                    "support_val":   int(support_pc[idx]),
                })
            _save_freq_vs_f1_csv(per_class_val, train_counts, output_dir / "class_freq_vs_f1.csv")
            xs = np.array([train_counts[c["class"]] for c in per_class_val], dtype=float)
            ys = np.array([c["f1_val"] for c in per_class_val], dtype=float)
            valid = (ys > 0) & (xs > 0)
            if valid.sum() > 1:
                pearson = float(np.corrcoef(xs[valid], ys[valid])[0, 1])
                spearman = float(_spearman(xs[valid], ys[valid]))
            else:
                pearson, spearman = 0.0, 0.0
            summary["freq_vs_f1_correlation_val"] = {
                "pearson":  pearson,
                "spearman": spearman,
                "n_valid_pairs": int(valid.sum()),
            }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    _write_report_md(summary, output_dir / "REPORT.md")
    print(f"\nReport written to {output_dir / 'REPORT.md'}")


def _save_per_class_csv(rows, path):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["class", "precision", "recall", "f1", "support", "train_freq"])
        w.writeheader()
        w.writerows(rows)


def _save_pairs_csv(rows, path):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["true_class", "predicted_class", "count", "train_freq_true", "train_freq_pred"])
        w.writeheader()
        w.writerows(rows)


def _save_zero_recall_csv(rows, path):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["class", "support", "precision", "recall", "f1", "train_freq"])
        w.writeheader()
        w.writerows(rows)


def _save_freq_vs_f1_csv(rows, train_counts, path):
    import csv
    out = []
    for r in rows:
        out.append({
            "class": r["class"],
            "train_freq": int(train_counts.get(r["class"], 0)),
            "val_support": r["support_val"],
            "val_precision": r["precision_val"],
            "val_recall": r["recall_val"],
            "val_f1": r["f1_val"],
        })
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["class", "train_freq", "val_support", "val_precision", "val_recall", "val_f1"])
        w.writeheader()
        w.writerows(out)


def _spearman(xs, ys):
    """Spearman rank correlation (handles ties via midrank)."""
    def rankdata(a):
        order = np.argsort(a)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(a) + 1)
        sorted_a = a[order]
        i = 0
        while i < len(a):
            j = i
            while j < len(a) and sorted_a[j] == sorted_a[i]:
                j += 1
            ranks[order[i:j]] = (i + 1 + j) / 2.0
            i = j
        return ranks
    rx, ry = rankdata(xs), rankdata(ys)
    return float(np.corrcoef(rx, ry)[0, 1])


def _write_report_md(summary, path):
    test = summary.get("test", {})
    val = summary.get("val", {})
    freq_corr = summary.get("freq_vs_f1_correlation_val", {})
    conf = summary.get("confidence_test", {})
    zr = summary.get("zero_recall_test", [])

    lines = []
    lines.append("# Error Analysis Report")
    lines.append("")
    lines.append(f"**Checkpoint**: `{summary['checkpoint']}`")
    cfg = summary["model_config"]
    lines.append(f"**Model config**: input_size={cfg['input_size']}, "
                 f"hidden_size={cfg['hidden_size']}, "
                 f"num_layers={cfg['num_layers']}, "
                 f"num_classes={cfg['num_classes']}, "
                 f"pooling={cfg['pooling']}")
    lines.append(f"**Classes evaluated**: {summary['class_to_idx_size']}")
    lines.append("")

    lines.append("## Headline metrics")
    lines.append("")
    lines.append("| Split | n | accuracy | macro F1 | weighted F1 |")
    lines.append("|---|---|---|---|---|")
    lines.append(f"| val  | {val.get('n_evaluated', '-')} | "
                 f"{val.get('accuracy', 0)*100:.2f}% | "
                 f"{val.get('macro_f1', 0)*100:.2f}% | "
                 f"{val.get('weighted_f1', 0)*100:.2f}% |")
    lines.append(f"| test | {test.get('n_evaluated', '-')} | "
                 f"{test.get('accuracy', 0)*100:.2f}% | "
                 f"{test.get('macro_f1', 0)*100:.2f}% | "
                 f"{test.get('weighted_f1', 0)*100:.2f}% |")
    lines.append("")

    if conf:
        lines.append("## Confidence statistics (test)")
        lines.append("")
        lines.append(f"- Correct predictions: n={conf['n_correct']}, "
                     f"mean conf = {conf['mean_conf_correct']:.4f}, "
                     f"median = {conf['median_conf_correct']:.4f}")
        lines.append(f"- Incorrect predictions: n={conf['n_incorrect']}, "
                     f"mean conf = {conf['mean_conf_incorrect']:.4f}, "
                     f"median = {conf['median_conf_incorrect']:.4f}")
        lines.append("")

    lines.append(f"## (c) Zero-recall classes: {len(zr)} classes with recall == 0 on test")
    lines.append("")
    lines.append("Sorted by training-set frequency (highest first) - these are the highest-impact failure modes.")
    lines.append("")
    lines.append("| # | Class | Test support | Train freq | Precision | F1 |")
    lines.append("|---|---|---|---|---|---|")
    for i, r in enumerate(zr[:29], 1):
        lines.append(f"| {i} | {r['class']} | {r['support']} | {r['train_freq']} | "
                     f"{r['precision']:.2f} | {r['f1']:.2f} |")
    lines.append("")

    pairs = summary.get("top_confused_pairs_test", [])
    lines.append(f"## (a) Top confused class pairs on test (top 20 of {len(pairs)})")
    lines.append("")
    lines.append("Off-diagonal entries of the confusion matrix, sorted by frequency.")
    lines.append("")
    lines.append("| # | True class | -> Predicted as | Count | True train freq | Pred train freq |")
    lines.append("|---|---|---|---|---|---|")
    for i, p in enumerate(pairs[:20], 1):
        lines.append(f"| {i} | {p['true_class']} | {p['predicted_class']} | "
                     f"{p['count']} | {p['train_freq_true']} | {p['train_freq_pred']} |")
    lines.append("")

    pc = summary.get("per_class_metrics_test_count", 0)
    lines.append(f"## (b) Per-class precision/recall/F1 distribution (n={pc} classes)")
    lines.append("")
    lines.append("See `per_class_metrics.csv` for the full 200-row table. Summary statistics:")
    lines.append("")
    per_class_csv_path = path.parent / "per_class_metrics.csv"
    if per_class_csv_path.exists():
        import csv
        with open(per_class_csv_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for metric_name in ("precision", "recall", "f1"):
            arr = np.array([float(r[metric_name]) for r in rows])
            lines.append(f"**{metric_name}**: min={arr.min():.3f}, median={np.median(arr):.3f}, "
                         f"mean={arr.mean():.3f}, max={arr.max():.3f}, "
                         f"classes with 0 = {int((arr == 0).sum())}, "
                         f"classes < 0.5 = {int((arr < 0.5).sum())}, "
                         f"classes >= 0.8 = {int((arr >= 0.8).sum())}")
        lines.append("")

    if freq_corr:
        lines.append("## (d) Class frequency vs performance correlation (val)")
        lines.append("")
        lines.append(f"- Pearson r (train_freq, val_F1) = {freq_corr['pearson']:.3f}")
        lines.append(f"- Spearman rho (train_freq, val_F1) = {freq_corr['spearman']:.3f}")
        lines.append(f"- n valid pairs = {freq_corr['n_valid_pairs']}")
        lines.append("")
        lines.append("Strong positive correlation = classes with more training samples "
                     "achieve higher F1. See `class_freq_vs_f1.csv` for the per-class table.")
        lines.append("")

    lines.append("---")
    lines.append("Generated by `src.inference.error_analysis`.")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
