"""
Training script for 126D GRU model on 20-word Azerbaijani Sign Language demo dataset.

Optimized for CUDA GPU acceleration (NVIDIA RTX 3060 Laptop GPU).
Saves trained checkpoint to outputs/checkpoints/gru_demo_20_words_best.pt.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / "src" / "models"))

from src.models.gru_classifier import GRUClassifier  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("train_demo_model")

SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DemoFeatureDataset(Dataset):
    """PyTorch Dataset loading 26x126 feature arrays from .npz files."""

    def __init__(self, samples: list[tuple[Path, int]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        npz_path, label = self.samples[idx]
        with np.load(npz_path) as data:
            if "features" in data:
                feats = data["features"]
            elif "frame_features" in data:
                feats = data["frame_features"]
            else:
                raise KeyError(f"No features found in {npz_path}")

        # Ensure float32 tensor of shape (26, D)
        feats_tensor = torch.from_numpy(feats).float()
        return feats_tensor, label


def prepare_demo_dataset(
    dataset_dir: Path,
) -> tuple[dict[str, list[tuple[Path, int]]], dict[int, str], dict[str, int]]:
    """Discover class folders, create class mappings, and split into train/val/test."""
    dataset_dir = Path(dataset_dir).resolve()
    class_folders = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
    if not class_folders:
        raise RuntimeError(f"No class folders found in {dataset_dir}")

    class_to_idx = {d.name: i for i, d in enumerate(class_folders)}
    idx_to_class = {i: d.name for i, d in enumerate(class_folders)}

    all_samples: list[tuple[Path, int]] = []
    labels: list[int] = []

    for folder in class_folders:
        label = class_to_idx[folder.name]
        files = list(folder.glob("*.npz"))
        for f in files:
            all_samples.append((f, label))
            labels.append(label)

    log.info(
        "Discovered %d total samples across %d classes in %s",
        len(all_samples),
        len(class_to_idx),
        dataset_dir,
    )

    # 80% train, 10% val, 10% test stratified split
    train_idx, temp_idx, _, temp_labels = train_test_split(
        range(len(all_samples)),
        labels,
        test_size=0.20,
        random_state=SEED,
        stratify=labels,
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_labels,
    )

    splits = {
        "train": [all_samples[i] for i in train_idx],
        "val": [all_samples[i] for i in val_idx],
        "test": [all_samples[i] for i in test_idx],
    }

    log.info(
        "Split counts: Train=%d, Val=%d, Test=%d",
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
    )
    return splits, idx_to_class, class_to_idx


def print_environment(device: torch.device) -> None:
    print("=" * 65)
    print("  ENVIRONMENT (20-Word Demo Training Pipeline)")
    print("=" * 65)
    print(f"device          : {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"gpu             : {props.name}")
        print(f"vram_gb         : {props.total_memory / (1024 ** 3):.2f}")
        print(f"GPU ACTIVATED   : PyTorch is actively using GPU -> {props.name}")
    else:
        print("GPU STATUS      : CUDA not available, using CPU")
    print(f"torch           : {torch.__version__}")
    print(f"torch_cuda      : {torch.version.cuda}")
    print("-" * 65)


@torch.no_grad()
def evaluate_model(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * y.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    avg_loss = total_loss / max(1, total)
    acc = correct / max(1, total)
    return avg_loss, acc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset/demo_20_words_features"),
        help="Path to 20-word features directory",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument(
        "--pooling", choices=("last", "mean_max"), default="mean_max"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/checkpoints/gru_demo_20_words_best.pt"),
    )
    parser.add_argument(
        "--split-metadata",
        type=Path,
        default=Path("outputs/demo_20_words_split.json"),
    )
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print_environment(device)

    splits, idx_to_class, class_to_idx = prepare_demo_dataset(args.dataset_dir)
    num_classes = len(class_to_idx)

    train_ds = DemoFeatureDataset(splits["train"])
    val_ds = DemoFeatureDataset(splits["val"])
    test_ds = DemoFeatureDataset(splits["test"])

    # Infer input feature size dynamically (typically 126)
    sample_x, _ = train_ds[0]
    input_dim = sample_x.shape[1]
    log.info("Detected input feature size: %dD", input_dim)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory
    )

    model_cfg = dict(
        input_size=input_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_classes=num_classes,
        dropout=args.dropout,
        bidirectional=False,
        pooling=args.pooling,
    )

    model = GRUClassifier(**model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Trainable model parameters: %d", n_params)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_acc = -1.0
    best_epoch = -1
    epochs_no_improve = 0

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.split_metadata.parent.mkdir(parents=True, exist_ok=True)

    # Save split metadata
    split_info = {
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "splits": {
            k: [str(p.relative_to(PROJECT_ROOT)) for p, _ in v]
            for k, v in splits.items()
        },
    }
    with open(args.split_metadata, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)

    print("-" * 65)
    header = f"{'epoch':>5} {'train_loss':>10} {'val_loss':>9} {'val_acc':>8} {'best_acc':>9} {'sec':>6}"
    print(header)

    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * y.size(0)

        train_loss = running_loss / len(train_ds)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        epoch_time = time.time() - t_epoch

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_acc": best_val_acc,
                    "class_to_idx": class_to_idx,
                    "idx_to_class": idx_to_class,
                    "model_config": model_cfg,
                },
                args.checkpoint,
            )
        else:
            epochs_no_improve += 1

        print(
            f"{epoch:>5} {train_loss:>10.4f} {val_loss:>9.4f} "
            f"{val_acc:>8.4f} {best_val_acc:>9.4f} {epoch_time:>6.1f}"
            f"{' *' if improved else ''}"
        )

        if epochs_no_improve >= args.patience:
            print(
                f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)."
            )
            break

    # Final evaluation on Test Set using best checkpoint
    print("=" * 65)
    print("  EVALUATING BEST CHECKPOINT ON TEST SET")
    print("=" * 65)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)

    total_duration = time.time() - t_start
    print(f"Best Epoch       : {best_epoch}")
    print(f"Val Accuracy     : {best_val_acc * 100:.2f}%")
    print(f"Test Accuracy    : {test_acc * 100:.2f}%")
    print(f"Test Loss        : {test_loss:.4f}")
    print(f"Total Time       : {total_duration:.1f} seconds")
    print(f"Saved Checkpoint : {args.checkpoint}")
    print("=" * 65)

    return 0


if __name__ == "__main__":
    sys.exit(main())
