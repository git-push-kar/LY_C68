"""
utils.py
========
Shared utilities: metrics, checkpointing, early stopping,
plotting, logging.
"""

import os
import random
import shutil

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = self.count = 0.0

    def update(self, val: float, n: int = 1):
        self.sum   += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count else 0.0


class EarlyStopping:
    def __init__(self, patience: int = 8, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best      = float("-inf")
        self.stop      = False

    def __call__(self, metric: float) -> bool:
        if metric > self.best + self.min_delta:
            self.best    = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


def save_checkpoint(state: dict, is_best: bool,
                    checkpoint_dir: str,
                    filename: str = "checkpoint.pth"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    torch.save(state, path)
    if is_best:
        shutil.copyfile(
            path,
            os.path.join(checkpoint_dir, "best_model.pth"),
        )


def load_checkpoint(path: str, model,
                    optimizer=None, scheduler=None) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    print(f"Loaded checkpoint: {path}  "
          f"(epoch {ckpt.get('epoch', '?')})")
    return ckpt


def compute_metrics(labels: list, preds: list,
                    probs: list = None) -> dict:
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "f1":       f1_score(labels, preds, average="binary",
                             zero_division=0),
        "report":   classification_report(
                        labels, preds,
                        target_names=["real", "fake"],
                        zero_division=0),
    }
    if probs is not None:
        try:
            metrics["auc"] = roc_auc_score(labels, probs)
        except Exception:
            metrics["auc"] = None
    return metrics


def plot_training_curves(train_losses: list, val_losses: list,
                         val_accuracies: list, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, train_losses, label="Train",
                 color="#2196F3", linewidth=2)
    axes[0].plot(epochs, val_losses, label="Val",
                 color="#F44336", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, val_accuracies,
                 color="#4CAF50", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Val Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Curves saved → {path}")


class Logger:
    def __init__(self, log_dir: str, filename: str = "train.log"):
        os.makedirs(log_dir, exist_ok=True)
        self._fh = open(
            os.path.join(log_dir, filename), "a", encoding="utf-8")

    def log(self, msg: str):
        print(msg)
        self._fh.write(msg + "\n")
        self._fh.flush()

    def close(self):
        self._fh.close()