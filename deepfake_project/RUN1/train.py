"""
train.py
========
Full training pipeline for HydraFakeNet.

Usage:
    python train.py \
        --dataset_root /path/to/hydrafake \
        --json_root    /path/to/hydrafake/jsons \
        --output_dir   ./runs/exp1

All major hyper-parameters are exposed as CLI flags.
"""

import argparse
import os
import time

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from dataset import build_dataloaders
from model import HydraFakeNet
from utils import (
    AverageMeter,
    EarlyStopping,
    Logger,
    compute_metrics,
    load_checkpoint,
    plot_training_curves,
    save_checkpoint,
    set_seed,
)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class CombinedLoss(nn.Module):
    """
    Weighted sum of classification loss and reasoning generation loss.

    L = alpha * CE(cls) + (1 - alpha) * CE(reasoning)
    """

    def __init__(self, vocab_size: int, pad_id: int = 0, alpha: float = 0.7):
        super().__init__()
        self.alpha = alpha
        self.cls_loss = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.gen_loss = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(
        self,
        cls_logits: torch.Tensor,
        labels: torch.Tensor,
        reasoning_logits: torch.Tensor | None = None,
        reasoning_targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        loss_cls = self.cls_loss(cls_logits, labels)
        losses = {"cls_loss": loss_cls.item()}

        if reasoning_logits is not None and reasoning_targets is not None:
            # Shift targets by one (predict next token)
            # reasoning_ids input  → [BOS, t1, t2, ..., tN]
            # reasoning_ids target → [t1, t2, ..., tN, EOS/PAD]
            logits_flat = reasoning_logits[:, :-1, :].reshape(-1, reasoning_logits.size(-1))
            targets_flat = reasoning_targets[:, 1:].reshape(-1)
            loss_gen = self.gen_loss(logits_flat, targets_flat)
            losses["gen_loss"] = loss_gen.item()
            total = self.alpha * loss_cls + (1 - self.alpha) * loss_gen
        else:
            total = loss_cls

        losses["total"] = total.item()
        return total, losses


# ---------------------------------------------------------------------------
# Single training epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: HydraFakeNet,
    loader,
    optimizer,
    criterion: CombinedLoss,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    logger: Logger,
    log_interval: int = 50,
    grad_clip: float = 1.0,
) -> dict:
    model.train()
    loss_meter = AverageMeter("loss")
    cls_loss_meter = AverageMeter("cls_loss")
    gen_loss_meter = AverageMeter("gen_loss")

    all_preds, all_labels = [], []
    t0 = time.time()

    for step, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        reasoning_ids = batch["reasoning_ids"].to(device, non_blocking=True)

        with autocast():
            outputs = model(images, reasoning_ids=reasoning_ids)
            cls_logits = outputs["cls_logits"]
            reasoning_logits = outputs.get("reasoning_logits")

            total_loss, sub_losses = criterion(
                cls_logits,
                labels,
                reasoning_logits,
                reasoning_ids,
            )

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        bs = images.size(0)
        loss_meter.update(sub_losses["total"], bs)
        cls_loss_meter.update(sub_losses["cls_loss"], bs)
        if "gen_loss" in sub_losses:
            gen_loss_meter.update(sub_losses["gen_loss"], bs)

        preds = cls_logits.argmax(dim=-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

        if (step + 1) % log_interval == 0:
            elapsed = time.time() - t0
            logger.log(
                f"  Epoch {epoch:03d} | Step {step+1:4d}/{len(loader)} | "
                f"Loss {loss_meter.avg:.4f} | ClsLoss {cls_loss_meter.avg:.4f} | "
                f"GenLoss {gen_loss_meter.avg:.4f} | {elapsed:.1f}s"
            )

    metrics = compute_metrics(all_labels, all_preds)
    return {
        "loss": loss_meter.avg,
        "cls_loss": cls_loss_meter.avg,
        "gen_loss": gen_loss_meter.avg,
        "accuracy": metrics["accuracy"],
        "f1": metrics["f1"],
    }


# ---------------------------------------------------------------------------
# Validation / evaluation epoch
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: HydraFakeNet,
    loader,
    criterion: CombinedLoss,
    device: torch.device,
) -> dict:
    model.eval()
    loss_meter = AverageMeter("val_loss")
    all_preds, all_labels, all_probs = [], [], []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        reasoning_ids = batch["reasoning_ids"].to(device, non_blocking=True)

        with autocast():
            outputs = model(images, reasoning_ids=reasoning_ids)
            cls_logits = outputs["cls_logits"]
            reasoning_logits = outputs.get("reasoning_logits")
            total_loss, _ = criterion(cls_logits, labels, reasoning_logits, reasoning_ids)

        loss_meter.update(total_loss.item(), images.size(0))

        probs = torch.softmax(cls_logits, dim=-1)[:, 1].cpu().tolist()
        preds = cls_logits.argmax(dim=-1).cpu().tolist()
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics["loss"] = loss_meter.avg
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="HydraFake Training")
    p.add_argument("--dataset_root", required=True, help="Path to hydrafake/ root.")
    p.add_argument("--json_root", required=True, help="Path to hydrafake/jsons/.")
    p.add_argument("--output_dir", default="./runs/exp1")
    p.add_argument("--tokenizer", default="bert-base-uncased")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--embed_dim", type=int, default=512)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--loss_alpha", type=float, default=0.7,
                   help="Weight of classification loss (1-alpha for reasoning loss).")
    p.add_argument("--patience", type=int, default=10,
                   help="Early stopping patience (epochs).")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from.")
    p.add_argument("--max_text_len", type=int, default=128)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--warmup_epochs", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    logger = Logger(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log(f"Device: {device}")
    if device.type == "cuda":
        logger.log(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Data ────────────────────────────────────────────────────────────
    logger.log("Building DataLoaders …")
    loaders, tokenizer = build_dataloaders(
        dataset_root=args.dataset_root,
        json_root=args.json_root,
        tokenizer_name=args.tokenizer,
        image_size=args.image_size,
        max_text_len=args.max_text_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    vocab_size = tokenizer.vocab_size
    pad_id = tokenizer.pad_token_id or 0
    bos_id = tokenizer.cls_token_id or 101
    eos_id = tokenizer.sep_token_id or 102

    # ── Model ────────────────────────────────────────────────────────────
    logger.log("Building HydraFakeNet …")
    model = HydraFakeNet(
        image_size=args.image_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        vocab_size=vocab_size,
        reasoning_hidden_dim=args.embed_dim,
        max_reasoning_len=args.max_text_len,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"Trainable parameters: {total_params:,}")

    # ── Optimiser & Scheduler ────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    total_steps = args.epochs * len(loaders.get("train", []))
    warmup_steps = args.warmup_epochs * len(loaders.get("train", []))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress)).item()))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    criterion = CombinedLoss(vocab_size, pad_id=pad_id, alpha=args.loss_alpha).to(device)
    scaler = GradScaler()
    early_stopper = EarlyStopping(patience=args.patience, mode="max")

    start_epoch = 1
    best_val_acc = 0.0
    train_losses, val_losses, val_accuracies = [], [], []

    # ── Resume ───────────────────────────────────────────────────────────
    if args.resume:
        ckpt = load_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)

    # ── Training Loop ────────────────────────────────────────────────────
    logger.log("\n" + "=" * 70)
    logger.log("Starting training …")
    logger.log("=" * 70 + "\n")

    for epoch in range(start_epoch, args.epochs + 1):
        logger.log(f"\n{'─'*60}")
        logger.log(f"Epoch {epoch}/{args.epochs}  lr={scheduler.get_last_lr()[0]:.2e}")

        train_stats = train_one_epoch(
            model, loaders["train"], optimizer, criterion, scaler,
            device, epoch, logger, args.log_interval, args.grad_clip,
        )
        scheduler.step()

        logger.log(
            f"  TRAIN → loss={train_stats['loss']:.4f} "
            f"acc={train_stats['accuracy']:.4f} f1={train_stats['f1']:.4f}"
        )

        val_stats = evaluate(model, loaders["val"], criterion, device)
        logger.log(
            f"  VAL   → loss={val_stats['loss']:.4f} "
            f"acc={val_stats['accuracy']:.4f} f1={val_stats['f1']:.4f} "
            f"auc={val_stats.get('auc', 'N/A')}"
        )

        train_losses.append(train_stats["loss"])
        val_losses.append(val_stats["loss"])
        val_accuracies.append(val_stats["accuracy"])

        is_best = val_stats["accuracy"] > best_val_acc
        if is_best:
            best_val_acc = val_stats["accuracy"]

        save_checkpoint(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_acc": best_val_acc,
                "val_stats": val_stats,
            },
            is_best=is_best,
            checkpoint_dir=os.path.join(args.output_dir, "checkpoints"),
            filename=f"checkpoint_ep{epoch:03d}.pth",
        )

        if early_stopper(val_stats["accuracy"]):
            logger.log(f"Early stopping triggered at epoch {epoch}.")
            break

    # ── Save training curves ─────────────────────────────────────────────
    plot_training_curves(
        train_losses, val_losses, val_accuracies,
        save_dir=args.output_dir,
    )

    logger.log(f"\nBest validation accuracy: {best_val_acc:.4f}")
    logger.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
