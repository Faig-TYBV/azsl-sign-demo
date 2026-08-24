"""
GRU baseline training for AzSLD word recognition.

Uses the EXISTING split from outputs/dataset_split.json (never re-splits),
the existing class mapping, and class weights computed from the TRAIN
split only.

Usage:
    python src/training/train_gru.py [--epochs 30] [--batch-size 32]
        [--lr 1e-3] [--weight-decay 1e-4] [--patience 7] [--num-workers 0]
"""

import argparse
import csv
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC / "data"))
sys.path.insert(0, str(SRC / "models"))
sys.path.insert(0, str(SRC / "training"))

from azsl_dataset import DEFAULT_NUM_WORKERS, load_split_metadata, AzslFeatureDataset  # noqa: E402
from gru_classifier import GRUClassifier  # noqa: E402
from metrics import confusion_matrix, metrics_from_confusion  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("train_gru")

DEFAULT_METADATA_PATH = Path("outputs/dataset_split.json")
DEFAULT_CHECKPOINT_PATH = Path("outputs/checkpoints/gru_baseline_best.pt")
DEFAULT_HISTORY_JSON = Path("outputs/training_history_gru_baseline.json")
DEFAULT_HISTORY_CSV = Path("outputs/training_history_gru_baseline.csv")

SEED: int = 42


def set_seed(seed: int) -> None:
    """Seed python/numpy/torch; keep cudnn benchmark ON for GPU speed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> dict:
    """Return loss + metrics for a full pass over ``loader``."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * y.size(0)
        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(y.cpu())

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()
    cm = confusion_matrix(y_true, y_pred, num_classes)
    result = metrics_from_confusion(cm)
    result["loss"] = total_loss / max(1, len(loader.dataset))
    return result


def print_environment(device: torch.device) -> None:
    print("=" * 60)
    print("  ENVIRONMENT")
    print("=" * 60)
    print(f"device          : {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"gpu             : {props.name}")
        print(f"vram_gb         : {props.total_memory / (1024 ** 3):.2f}")
    print(f"torch           : {torch.__version__}")
    print(f"torch_cuda      : {torch.version.cuda}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--pooling", choices=("last", "mean_max"), default="last",
                        help="'last' = final hidden state (baseline); "
                             "'mean_max' = temporal mean+max pooling")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--history-json", type=Path, default=None,
                        help="default: outputs/training_history_gru_baseline.json")
    parser.add_argument("--report", type=Path, default=None,
                        help="default: outputs/test_report_gru_baseline.json")
    parser.add_argument("--normalize", action="store_true",
                        help="apply train-only per-dim feature normalization")
    parser.add_argument("--balanced-sampler", action="store_true",
                        help="use WeightedRandomSampler on train loader with "
                             "unweighted CrossEntropyLoss (replaces weighted loss)")
    parser.add_argument("--norm-stats", type=Path, default=None,
                        help="where to save/load normalization statistics")
    parser.add_argument("--delta", action="store_true",
                        help="concatenate temporal delta features "
                             "(input_size 252)")
    parser.add_argument("--augment", action="store_true",
                        help="enable training-time temporal augmentation "
                             "(train split only)")
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--temporal-mask-prob", type=float, default=0.10)
    parser.add_argument("--temporal-dropout-prob", type=float, default=0.05)
    args = parser.parse_args()

    t_start = time.time()
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print_environment(device)

    # ---- Data: reuse the SAVED split ------------------------------------
    data = load_split_metadata(args.metadata)
    idx_to_class = data["idx_to_class"]
    num_classes = len(idx_to_class)
    augment = None
    if args.augment:
        from augmentation import TemporalAugmentation  # src/data
        augment = TemporalAugmentation(
            noise_std=args.noise_std,
            temporal_mask_prob=args.temporal_mask_prob,
            temporal_dropout_prob=args.temporal_dropout_prob,
            seed=SEED,
        )
        log.info("Temporal augmentation ENABLED (train only): noise_std=%s "
                 "mask_prob=%s dropout_prob=%s", args.noise_std,
                 args.temporal_mask_prob, args.temporal_dropout_prob)
    normalizer = None
    if args.normalize:
        from normalization import fit_normalization, FeatureNormalizer
        norm_stats_path = args.norm_stats or Path(
            "outputs/feature_normalization_stats.json")
        if norm_stats_path.exists():
            normalizer = FeatureNormalizer.load(norm_stats_path)
            log.info("Loaded normalization statistics from %s", norm_stats_path)
        else:
            log.info("Fitting normalization on TRAIN split only...")
            stats = fit_normalization(data["splits"]["train"])
            normalizer = FeatureNormalizer.from_stats(stats)
            normalizer.save(norm_stats_path, extra={
                "fitted_on": "train_split_only",
                "n_valid_frames": stats["n_valid_frames"],
                "epsilon": stats["epsilon"],
            })
            log.info("Normalization fitted: %d valid frames, saved to %s",
                     stats["n_valid_frames"], norm_stats_path)

    train_ds = AzslFeatureDataset(data["splits"]["train"], augment=augment,
                                  with_delta=args.delta,
                                  normalizer=normalizer)
    val_ds = AzslFeatureDataset(data["splits"]["val"], with_delta=args.delta,
                                normalizer=normalizer)
    test_ds = AzslFeatureDataset(data["splits"]["test"], with_delta=args.delta,
                                 normalizer=normalizer)
    pin_memory = device.type == "cuda"
    common = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                  pin_memory=pin_memory)

    sampler = None
    shuffle = True
    if getattr(args, "balanced_sampler", False):
        # Balanced sampling REPLACES class-weighted loss. Weights use TRAIN
        # labels only: w_i = 1 / count(class_of_i).
        from collections import Counter
        train_labels = [s.label for s in train_ds.samples]
        counts = Counter(train_labels)
        sample_weights = [1.0 / counts[l] for l in train_labels]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_ds),
            replacement=True,
            generator=torch.Generator().manual_seed(SEED),
        )
        shuffle = False  # sampler provides shuffling
        log.info("Balanced sampling ENABLED (replacement=True, num_samples=%d); "
                 "loss = unweighted CrossEntropyLoss", len(train_ds))
        # Deterministic sanity check: draw one epoch of indices.
        idx_list = list(torch.utils.data.WeightedRandomSampler(
            weights=sample_weights, num_samples=len(train_ds),
            replacement=True, generator=torch.Generator().manual_seed(SEED)))
        sampled_counts = Counter(train_labels[i] for i in idx_list)
        sc = list(sampled_counts.values())
        analysis = {
            "num_sampled": len(idx_list),
            "unique_sampled": len(set(idx_list)),
            "min_class_count": int(min(sc)),
            "max_class_count": int(max(sc)),
            "mean_class_count": float(np.mean(sc)),
            "std_class_count": float(np.std(sc)),
            "classes_sampled": len(sc),
            "seed": SEED,
        }
        Path("outputs").mkdir(exist_ok=True)
        with open("outputs/sampler_analysis.json", "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2)
        log.info("Sampler sanity check: %s", analysis)

    train_loader = DataLoader(train_ds, shuffle=shuffle, sampler=sampler,
                              **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)

    # ---- Model / loss / optimizer ----------------------------------------
    model_cfg = dict(
        input_size=252 if args.delta else 126,
        hidden_size=args.hidden_size, num_layers=args.num_layers,
        num_classes=num_classes, dropout=args.dropout, bidirectional=False,
        pooling=args.pooling,
    )
    model = GRUClassifier(**model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Trainable parameters: %d", n_params)

    class_weights = data["class_weights"].to(device)  # TRAIN-only weights
    if getattr(args, "balanced_sampler", False):
        criterion = nn.CrossEntropyLoss()  # NO class weights: sampling handles imbalance
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_f1, best_epoch = -1.0, -1
    history, epochs_no_improve = [], 0
    ckpt_path = args.checkpoint
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    print("-" * 60)
    header = f"{'epoch':>5} {'train_loss':>10} {'val_loss':>9} " \
             f"{'val_acc':>8} {'val_macroF1':>11} {'bestF1':>7} {'sec':>6}"
    print(header)

    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * y.size(0)
        train_loss = running / len(train_ds)

        val_metrics = evaluate_model(model, val_loader, criterion, device, num_classes)
        epoch_time = time.time() - t_epoch
        improved = val_metrics["macro_f1"] > best_f1
        if improved:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_macro_f1": best_f1,
                    "class_to_idx": data["class_to_idx"],
                    "model_config": model_cfg,
                },
                ckpt_path,
            )
        else:
            epochs_no_improve += 1

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_metrics["loss"], 6),
            "val_accuracy": round(val_metrics["accuracy"], 6),
            "val_macro_f1": round(val_metrics["macro_f1"], 6),
            "val_weighted_f1": round(val_metrics["weighted_f1"], 6),
            "val_macro_recall": round(val_metrics["macro_recall"], 6),
        })
        print(f"{epoch:>5} {train_loss:>10.4f} {val_metrics['loss']:>9.4f} "
              f"{val_metrics['accuracy']:>8.4f} {val_metrics['macro_f1']:>11.4f} "
              f"{best_f1:>7.4f} {epoch_time:>6.1f}"
              f"{' *' if improved else ''}")

        if epochs_no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch} "
                  f"(no improvement for {args.patience} epochs).")
            break

    # ---- Final test evaluation (BEST checkpoint only) ----------------------
    log.info("Loading best checkpoint from %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = evaluate_model(model, test_loader, criterion, device, num_classes)

    print("=" * 60)
    print("  TEST RESULTS (best checkpoint)")
    print("=" * 60)
    print(f"test_loss      : {test_metrics['loss']:.4f}")
    print(f"accuracy       : {test_metrics['accuracy']:.4f}")
    print(f"macro_f1       : {test_metrics['macro_f1']:.4f}")
    print(f"weighted_f1    : {test_metrics['weighted_f1']:.4f}")
    print(f"macro_recall   : {test_metrics['macro_recall']:.4f}")

    # Save per-class report + confusion matrix to JSON.
    report = {
        "test": {k: v for k, v in test_metrics.items()},
        "confusion_matrix": None,  # filled below (numpy -> list)
        "history": history,
        "model_config": model_cfg,
        "n_trainable_params": n_params,
        "seed": SEED,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    # Recompute CM for saving (evaluate_model discarded it).
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x.to(device))
            all_preds.append(logits.argmax(dim=1).cpu())
            all_labels.append(y)
    cm = confusion_matrix(torch.cat(all_labels).numpy(),
                          torch.cat(all_preds).numpy(), num_classes)
    report["confusion_matrix"] = cm.tolist()

    history_json = args.history_json or DEFAULT_HISTORY_JSON
    report_path = args.report or Path("outputs/test_report_gru_baseline.json")
    out_json = report_path
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False)

    with open(history_json, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    csv_path = history_json.with_suffix(".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    total_min = (time.time() - t_start) / 60
    gpu_mem = (
        f"{torch.cuda.max_memory_allocated(0) / (1024 ** 2):.0f} MiB"
        if device.type == "cuda" else "n/a"
    )
    print("-" * 60)
    print(f"best_epoch     : {ckpt['epoch']}")
    print(f"best_val_macroF1: {ckpt['best_val_macro_f1']:.4f}")
    print(f"total_time     : {total_min:.1f} min")
    print(f"gpu_memory_max : {gpu_mem}")
    print(f"checkpoint     : {ckpt_path}")
    print(f"history        : {history_json} / {csv_path}")
    print(f"test report    : {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

