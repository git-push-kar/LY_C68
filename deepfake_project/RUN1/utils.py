"""
utils.py
========
Shared utilities:
  - AverageMeter for running metrics
  - EarlyStopping
  - checkpoint save / load
  - training curve plots
  - seed setting
"""

import os
import json
import random
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# AverageMeter
# ---------------------------------------------------------------------------

class AverageMeter:
    """Tracks running mean of a scalar (e.g. loss, accuracy)."""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0.0

    def __repr__(self):
        return f"{self.name}: {self.avg:.4f}"


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Stop training when validation metric stops improving.

    Args:
        patience  : epochs to wait before stopping.
        min_delta : minimum improvement to count.
        mode      : 'min' for loss, 'max' for accuracy/AUC.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best = float("inf") if mode == "min" else float("-inf")
        self.stop = False

    def __call__(self, metric: float) -> bool:
        if self.mode == "min":
            improved = metric < self.best - self.min_delta
        else:
            improved = metric > self.best + self.min_delta

        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

        return self.stop


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    state: dict,
    is_best: bool,
    checkpoint_dir: str,
    filename: str = "checkpoint.pth",
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pth")
        shutil.copyfile(path, best_path)
        print(f"  ✓ New best model saved → {best_path}")


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> dict:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    print(f"Loaded checkpoint from {path} (epoch {checkpoint.get('epoch', '?')})")
    return checkpoint


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    all_labels: list,
    all_preds: list,
    all_probs: Optional[list] = None,
) -> dict:
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds, target_names=["real", "fake"], zero_division=0
    )

    metrics = {"accuracy": acc, "f1": f1, "confusion_matrix": cm.tolist(), "report": report}

    if all_probs is not None:
        try:
            auc = roc_auc_score(all_labels, all_probs)
            metrics["auc"] = auc
        except Exception:
            metrics["auc"] = None

    return metrics


# ---------------------------------------------------------------------------
# Training curve plots
# ---------------------------------------------------------------------------

def plot_training_curves(
    train_losses: list,
    val_losses: list,
    val_accuracies: list,
    save_dir: str,
):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(epochs, train_losses, label="Train Loss", color="#2196F3", linewidth=2)
    axes[0].plot(epochs, val_losses, label="Val Loss", color="#F44336", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, val_accuracies, label="Val Accuracy", color="#4CAF50", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Validation Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved → {save_path}")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class Logger:
    """Simple logger that writes to stdout and a log file simultaneously."""

    def __init__(self, log_dir: str, filename: str = "train.log"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, filename)
        self._fh = open(self.log_path, "a", encoding="utf-8")

    def log(self, msg: str):
        print(msg)
        self._fh.write(msg + "\n")
        self._fh.flush()

    def close(self):
        self._fh.close()

    def save_metrics(self, metrics: dict, filename: str = "metrics.json"):
        save_path = os.path.join(os.path.dirname(self.log_path), filename)
        # Convert numpy types to native Python for JSON serialisation
        def _convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        serialisable = json.loads(
            json.dumps(metrics, default=_convert)
        )
        with open(save_path, "w") as f:
            json.dump(serialisable, f, indent=2)
        print(f"Metrics saved → {save_path}")
