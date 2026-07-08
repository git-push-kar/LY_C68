"""
train.py
========
Training pipeline for HydraFakeNet.

Key changes vs original:
  - CombinedLoss replaced with pure CrossEntropyLoss (label_smoothing=0.1).
    GRU decoder is never trained — all gradients come from classification.
  - Mixup augmentation added (alpha=0.2): interpolates between random
    image pairs, forcing the model to learn soft decision boundaries
    rather than hard GAN-specific fingerprints.
  - Dual learning-rate optimizer updated to include fft_branch param group.
  - Early stopping now monitors AUC (more sensitive than accuracy for
    imbalanced or partially-collapsed predictions).
  - Collapse detection checks both train and val predictions each epoch.
  - GenLoss column removed from header (decoder no longer trained).

Usage:
    python train.py \\
        --dataset_root "C:/path/to/hydrafake" \\
        --json_root    "C:/path/to/hydrafake/jsons" \\
        --output_dir   "./runs/exp7" \\
        --batch_size   32 \\
        --epochs       60 \\
        --lr           1e-4 \\
        --warmup_epochs 5 \\
        --patience     20 \\
        --freeze_mode  partial \\
        --dropout      0.4 \\
        --cls_depth    2 \\
        --cls_queries  4 \\
        --weight_decay 0.05
"""

import argparse
import os
import math

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from dataset import build_dataloaders
from model import HydraFakeNet
from utils import (
    AverageMeter, EarlyStopping, Logger,
    compute_metrics, load_checkpoint,
    plot_training_curves, save_checkpoint, set_seed,
)


# ---------------------------------------------------------------------------
# Mixup
# ---------------------------------------------------------------------------

def mixup_batch(images: torch.Tensor,
                labels: torch.Tensor,
                alpha:  float = 0.2):
    """
    Returns mixed images, original labels_a, shuffled labels_b, and lambda.
    Loss = lam * CE(logits, labels_a) + (1-lam) * CE(logits, labels_b).

    WHY: Forces the model to learn smooth decision boundaries instead of
    memorizing sharp GAN-specific artifact thresholds. Particularly
    effective for domain generalization — mixed samples sit between
    training distributions, improving test-time generalization.
    """
    if alpha <= 0.0:
        return images, labels, labels, 1.0

    lam     = float(np.random.beta(alpha, alpha))
    B       = images.size(0)
    idx     = torch.randperm(B, device=images.device)
    mixed   = lam * images + (1.0 - lam) * images[idx]
    return mixed, labels, labels[idx], lam


# ---------------------------------------------------------------------------
# LR scheduler — linear warmup then cosine decay
# ---------------------------------------------------------------------------

def build_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = ((step - warmup_steps) /
                    max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Train one epoch
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scheduler,
                    criterion, scaler, device,
                    mixup_alpha: float = 0.2) -> dict:
    model.train()
    loss_m  = AverageMeter()
    preds_all, labels_all = [], []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        # Mixup augmentation
        images, labels_a, labels_b, lam = mixup_batch(
            images, labels, alpha=mixup_alpha)

        with autocast("cuda"):
            out        = model(images)
            cls_logits = out["cls_logits"]
            # Mixup loss: weighted combination of both label losses
            loss = (lam       * criterion(cls_logits, labels_a) +
                    (1 - lam) * criterion(cls_logits, labels_b))

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        bs = images.size(0)
        loss_m.update(loss.item(), bs)
        # For accuracy tracking use hard labels (labels_a = original)
        preds_all.extend(
            cls_logits.detach().argmax(-1).cpu().tolist())
        labels_all.extend(labels_a.cpu().tolist())

    # ── Collapse detection ───────────────────────────────────────────
    unique_preds = set(preds_all)
    if len(unique_preds) == 1:
        print(
            f"  !! COLLAPSE WARNING: model predicting only class "
            f"{list(unique_preds)[0]} for ALL {len(preds_all)} train "
            f"samples. Check data balance and learning rate."
        )

    metrics = compute_metrics(labels_all, preds_all)
    return {
        "loss":     loss_m.avg,
        "accuracy": metrics["accuracy"],
        "f1":       metrics["f1"],
    }


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, loader, criterion, device) -> dict:
    model.eval()
    loss_m  = AverageMeter()
    preds_all, labels_all, probs_all = [], [], []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with autocast("cuda"):
            out        = model(images)
            cls_logits = out["cls_logits"]
            loss       = criterion(cls_logits, labels)

        loss_m.update(loss.item(), images.size(0))
        probs_all.extend(
            torch.softmax(cls_logits, -1)[:, 1].cpu().tolist())
        preds_all.extend(cls_logits.argmax(-1).cpu().tolist())
        labels_all.extend(labels.cpu().tolist())

    # ── Val collapse detection ───────────────────────────────────────
    unique_preds = set(preds_all)
    if len(unique_preds) == 1:
        print(
            f"  !! COLLAPSE WARNING: model predicting only class "
            f"{list(unique_preds)[0]} for ALL {len(preds_all)} val "
            f"samples."
        )

    metrics         = compute_metrics(labels_all, preds_all, probs_all)
    metrics["loss"] = loss_m.avg
    return metrics


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root",     required=True)
    p.add_argument("--json_root",        required=True)
    p.add_argument("--output_dir",       default="./runs/exp1")
    p.add_argument("--tokenizer",        default="bert-base-uncased")
    p.add_argument("--image_size",       type=int,   default=224)
    p.add_argument("--max_text_len",     type=int,   default=128)
    p.add_argument("--batch_size",       type=int,   default=32)
    p.add_argument("--num_workers",      type=int,   default=8)
    p.add_argument("--clip_model",
                   default="openai/clip-vit-base-patch32")
    p.add_argument("--freeze_mode",      default="partial",
                   choices=["all", "partial", "none"])
    p.add_argument("--unfreeze_last_n",  type=int,   default=4)
    p.add_argument("--model_dim",        type=int,   default=512)
    p.add_argument("--cls_depth",        type=int,   default=2)
    p.add_argument("--cls_heads",        type=int,   default=8)
    p.add_argument("--cls_queries",      type=int,   default=4)
    p.add_argument("--dropout",          type=float, default=0.4)
    p.add_argument("--drop_path",        type=float, default=0.1)
    p.add_argument("--reasoning_hidden", type=int,   default=512)
    p.add_argument("--reasoning_layers", type=int,   default=2)
    p.add_argument("--epochs",           type=int,   default=60)
    p.add_argument("--lr",               type=float, default=1e-4)
    p.add_argument("--weight_decay",     type=float, default=0.05)
    p.add_argument("--warmup_epochs",    type=int,   default=5)
    p.add_argument("--patience",         type=int,   default=20)
    p.add_argument("--mixup_alpha",      type=float, default=0.2)
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--resume",           default=None)
    # loss_alpha kept for CLI compatibility but ignored (always 1.0 now)
    p.add_argument("--loss_alpha",       type=float, default=1.0,
                   help="Deprecated — decoder not trained. Keep at 1.0.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    logger = Logger(args.output_dir)

    if args.loss_alpha < 1.0:
        logger.log(
            f"WARNING: --loss_alpha={args.loss_alpha} has no effect. "
            f"The GRU decoder is no longer trained. "
            f"All gradient signal comes from classification loss."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    logger.log(f"Device : {device}")
    if device.type == "cuda":
        logger.log(f"GPU    : {torch.cuda.get_device_name(0)}")

    # ── Data ────────────────────────────────────────────────────────
    logger.log("\nLoading data …")
    loaders, tokenizer = build_dataloaders(
        dataset_root   = args.dataset_root,
        json_root      = args.json_root,
        tokenizer_name = args.tokenizer,
        image_size     = args.image_size,
        max_text_len   = args.max_text_len,
        batch_size     = args.batch_size,
        num_workers    = args.num_workers,
    )

    if "train" not in loaders:
        logger.log("ERROR: No train split found. Check --json_root.")
        return

    vocab_size = tokenizer.vocab_size

    # ── Model ────────────────────────────────────────────────────────
    logger.log("\nBuilding model …")
    model = HydraFakeNet(
        clip_model_name   = args.clip_model,
        freeze_mode       = args.freeze_mode,
        unfreeze_last_n   = args.unfreeze_last_n,
        model_dim         = args.model_dim,
        cls_depth         = args.cls_depth,
        cls_heads         = args.cls_heads,
        cls_queries       = args.cls_queries,
        vocab_size        = vocab_size,
        reasoning_hidden  = args.reasoning_hidden,
        reasoning_layers  = args.reasoning_layers,
        max_reasoning_len = args.max_text_len,
        dropout           = args.dropout,
        drop_path         = args.drop_path,
    ).to(device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"Total params     : {total:,}")
    logger.log(f"Trainable params : {trainable:,}")

    # ── Optimizer — separate LR groups ──────────────────────────────
    #
    # CLIP encoder:    base LR (small — already pretrained)
    # feature_proj:    10x base LR (from scratch, needs faster learning)
    # fft_branch:      10x base LR (from scratch, frequency features)
    # cls_head:        10x base LR (from scratch, classification)
    # reasoning_dec:   0 effective LR — decoder not trained, but included
    #                  so optimizer state is saved in checkpoints and
    #                  resuming works without errors.
    #
    optimizer = torch.optim.AdamW([
        {
            "params":       model.clip_encoder.parameters(),
            "lr":           args.lr,
            "weight_decay": args.weight_decay,
        },
        {
            "params":       model.feature_proj.parameters(),
            "lr":           args.lr * 10,
            "weight_decay": args.weight_decay,
        },
        {
            "params":       model.fft_branch.parameters(),
            "lr":           args.lr * 10,
            "weight_decay": args.weight_decay,
        },
        {
            "params":       model.cls_head.parameters(),
            "lr":           args.lr * 10,
            "weight_decay": args.weight_decay,
        },
        {
            "params":       model.reasoning_decoder.parameters(),
            "lr":           0.0,      # decoder frozen during training
            "weight_decay": 0.0,
        },
    ])

    steps_per_epoch = len(loaders["train"])
    total_steps     = args.epochs * steps_per_epoch
    warmup_steps    = args.warmup_epochs * steps_per_epoch
    scheduler       = build_scheduler(
        optimizer, warmup_steps, total_steps)

    # Label smoothing 0.1:
    #   Prevents the model from becoming overconfident on training GANs.
    #   Softens hard 0/1 targets to 0.05/0.95, reducing the gradient
    #   incentive to push logits to extreme values for known generators.
    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.1).to(device)

    scaler = GradScaler("cuda")

    # Early stopping on AUC (not accuracy):
    #   AUC is more sensitive to partial collapse. When the model starts
    #   predicting one class more often, accuracy can still look okay
    #   if the dataset is balanced, but AUC immediately drops.
    early_stopper = EarlyStopping(patience=args.patience)

    start_epoch   = 1
    best_val_auc  = 0.0
    train_losses: list = []
    val_losses:   list = []
    val_accs:     list = []

    if args.resume:
        ckpt        = load_checkpoint(
            args.resume, model, optimizer, scheduler)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_auc = ckpt.get("best_val_auc", 0.0)
        logger.log(
            f"Resumed from epoch {start_epoch - 1}, "
            f"best AUC so far: {best_val_auc:.4f}"
        )

    # ── Print header ─────────────────────────────────────────────────
    logger.log("\n" + "=" * 78)
    logger.log(
        f"{'Epoch':>6}  "
        f"{'TrainLoss':>10}  {'TrainAcc':>9}  "
        f"{'ValLoss':>8}  {'ValAcc':>7}  {'ValF1':>6}  "
        f"{'AUC':>6}  {'Best':>4}"
    )
    logger.log("=" * 78)

    # ── Training loop ────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):

        train_stats = train_one_epoch(
            model, loaders["train"], optimizer, scheduler,
            criterion, scaler, device,
            mixup_alpha=args.mixup_alpha,
        )

        val_stats = {}
        if "val" in loaders:
            val_stats = validate(
                model, loaders["val"], criterion, device)

        train_losses.append(train_stats["loss"])
        if val_stats:
            val_losses.append(val_stats["loss"])
            val_accs.append(val_stats["accuracy"])

        # Use AUC as the "best" metric — more robust than accuracy
        current_auc = val_stats.get("auc") or 0.0
        is_best     = current_auc > best_val_auc
        if is_best:
            best_val_auc = current_auc

        logger.log(
            f"{epoch:>6}  "
            f"{train_stats['loss']:>10.4f}  "
            f"{train_stats['accuracy']:>9.4f}  "
            f"{val_stats.get('loss', 0):>8.4f}  "
            f"{val_stats.get('accuracy', 0):>7.4f}  "
            f"{val_stats.get('f1', 0):>6.4f}  "
            f"{current_auc:>6.4f}  "
            f"{'*' if is_best else '':>4}"
        )

        save_checkpoint(
            state={
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_auc":         best_val_auc,
                "train_losses":         train_losses,
                "val_losses":           val_losses,
                "val_accuracies":       val_accs,
            },
            is_best        = is_best,
            checkpoint_dir = os.path.join(
                args.output_dir, "checkpoints"),
            filename       = f"checkpoint_ep{epoch:03d}.pth",
        )

        if val_stats and early_stopper(current_auc):
            logger.log(
                f"\nEarly stopping at epoch {epoch}. "
                f"Best val AUC: {best_val_auc:.4f}"
            )
            break

    # ── Done ─────────────────────────────────────────────────────────
    logger.log("=" * 78)
    logger.log(f"Best val AUC : {best_val_auc:.4f}")

    if train_losses and val_losses:
        plot_training_curves(
            train_losses, val_losses, val_accs, args.output_dir)

    logger.close()


if __name__ == "__main__":
    main()