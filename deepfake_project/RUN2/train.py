"""
train.py
========
Training pipeline.
Prints one summary line per epoch only.
Val split is used for monitoring and early stopping only —
weights are never updated from val data.
Test split is never touched here — use evaluate.py for that.

Usage:
    python train.py \
        --dataset_root /path/to/hydrafake \
        --json_root    /path/to/hydrafake/jsons \
        --output_dir   ./runs/exp1
"""

import argparse
import os
import math

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
# Combined loss
# ---------------------------------------------------------------------------

class CombinedLoss(nn.Module):
    """
    alpha       * CrossEntropy(classification)
    (1 - alpha) * CrossEntropy(reasoning generation)
    """

    def __init__(self, vocab_size: int, pad_id: int = 0,
                 alpha: float = 0.7):
        super().__init__()
        self.alpha    = alpha
        self.cls_loss = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.gen_loss = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, cls_logits, labels,
                reasoning_logits=None, reasoning_targets=None):
        loss_cls = self.cls_loss(cls_logits, labels)
        if reasoning_logits is not None and \
                reasoning_targets is not None:
            lf = reasoning_logits[:, :-1, :].reshape(
                     -1, reasoning_logits.size(-1))
            tf = reasoning_targets[:, 1:].reshape(-1)
            loss_gen = self.gen_loss(lf, tf)
            total    = (self.alpha * loss_cls +
                        (1 - self.alpha) * loss_gen)
            return total, loss_cls.item(), loss_gen.item()
        return loss_cls, loss_cls.item(), 0.0


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
# Train one epoch  (no per-step printing)
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scheduler,
                    criterion, scaler, device) -> dict:
    model.train()
    loss_m = AverageMeter()
    cls_m  = AverageMeter()
    gen_m  = AverageMeter()
    preds_all, labels_all = [], []

    for batch in loader:
        images        = batch["image"].to(device, non_blocking=True)
        labels        = batch["label"].to(device, non_blocking=True)
        reasoning_ids = batch["reasoning_ids"].to(
                            device, non_blocking=True)

        with autocast("cuda"):
            out              = model(images,
                                     reasoning_ids=reasoning_ids)
            cls_logits       = out["cls_logits"]
            reasoning_logits = out.get("reasoning_logits")
            total, lc, lg    = criterion(
                cls_logits, labels,
                reasoning_logits, reasoning_ids,
            )

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        bs = images.size(0)
        loss_m.update(total.item(), bs)
        cls_m.update(lc, bs)
        gen_m.update(lg, bs)
        preds_all.extend(
            cls_logits.detach().argmax(-1).cpu().tolist())
        labels_all.extend(labels.cpu().tolist())

    metrics = compute_metrics(labels_all, preds_all)
    return {
        "loss":     loss_m.avg,
        "cls_loss": cls_m.avg,
        "gen_loss": gen_m.avg,
        "accuracy": metrics["accuracy"],
        "f1":       metrics["f1"],
    }


# ---------------------------------------------------------------------------
# Validate  (no gradient updates — monitoring only)
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, loader, criterion, device) -> dict:
    model.eval()
    loss_m = AverageMeter()
    preds_all, labels_all, probs_all = [], [], []

    for batch in loader:
        images        = batch["image"].to(device, non_blocking=True)
        labels        = batch["label"].to(device, non_blocking=True)
        reasoning_ids = batch["reasoning_ids"].to(
                            device, non_blocking=True)

        with autocast("cuda"):
            out              = model(images,
                                     reasoning_ids=reasoning_ids)
            cls_logits       = out["cls_logits"]
            reasoning_logits = out.get("reasoning_logits")
            total, _, _      = criterion(
                cls_logits, labels,
                reasoning_logits, reasoning_ids,
            )

        loss_m.update(total.item(), images.size(0))
        probs_all.extend(
            torch.softmax(cls_logits, -1)[:, 1].cpu().tolist())
        preds_all.extend(cls_logits.argmax(-1).cpu().tolist())
        labels_all.extend(labels.cpu().tolist())

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
    p.add_argument("--batch_size",       type=int,   default=64)
    p.add_argument("--num_workers",      type=int,   default=8)
    p.add_argument("--clip_model",
                   default="openai/clip-vit-base-patch32")
    p.add_argument("--freeze_mode",      default="partial",
                   choices=["all", "partial", "none"])
    p.add_argument("--unfreeze_last_n",  type=int,   default=4)
    p.add_argument("--model_dim",        type=int,   default=512)
    p.add_argument("--cls_depth",        type=int,   default=4)
    p.add_argument("--cls_heads",        type=int,   default=8)
    p.add_argument("--cls_queries",      type=int,   default=8)
    p.add_argument("--dropout",          type=float, default=0.1)
    p.add_argument("--reasoning_hidden", type=int,   default=512)
    p.add_argument("--reasoning_layers", type=int,   default=2)
    p.add_argument("--epochs",           type=int,   default=30)
    p.add_argument("--lr",               type=float, default=3e-5)
    p.add_argument("--weight_decay",     type=float, default=0.01)
    p.add_argument("--warmup_epochs",    type=int,   default=3)
    p.add_argument("--patience",         type=int,   default=8)
    p.add_argument("--loss_alpha",       type=float, default=0.7)
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--resume",           default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    logger = Logger(args.output_dir)

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
    vocab_size = tokenizer.vocab_size
    pad_id     = tokenizer.pad_token_id or 0

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
    ).to(device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    logger.log(f"Total params     : {total:,}")
    logger.log(f"Trainable params : {trainable:,}")

    # ── Optimizer — dual LR ──────────────────────────────────────────
    # CLIP layers update at base LR
    # Scratch heads update at 10x base LR
    optimizer = torch.optim.AdamW([
        {"params": model.clip_encoder.parameters(),
         "lr": args.lr},
        {"params": model.feature_proj.parameters(),
         "lr": args.lr * 10},
        {"params": model.cls_head.parameters(),
         "lr": args.lr * 10},
        {"params": model.reasoning_decoder.parameters(),
         "lr": args.lr * 10},
    ], weight_decay=args.weight_decay)

    steps_per_epoch = len(loaders["train"])
    total_steps     = args.epochs * steps_per_epoch
    warmup_steps    = args.warmup_epochs * steps_per_epoch
    scheduler       = build_scheduler(
                          optimizer, warmup_steps, total_steps)

    criterion     = CombinedLoss(vocab_size, pad_id,
                                 args.loss_alpha).to(device)
    scaler = torch.amp.GradScaler("cuda")
    early_stopper = EarlyStopping(patience=args.patience)

    start_epoch    = 1
    best_val_acc   = 0.0
    train_losses:  list = []
    val_losses:    list = []
    val_accs:      list = []

    if args.resume:
        ckpt         = load_checkpoint(
                           args.resume, model, optimizer, scheduler)
        start_epoch  = ckpt.get("epoch", 0) + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)

    # ── Print header ─────────────────────────────────────────────────
    logger.log("\n" + "=" * 80)
    logger.log(
        f"{'Epoch':>6}  "
        f"{'TrainLoss':>10}  {'ClsLoss':>8}  {'GenLoss':>8}  "
        f"{'TrainAcc':>9}  "
        f"{'ValLoss':>8}  {'ValAcc':>7}  {'ValF1':>6}  "
        f"{'AUC':>6}  {'Best':>4}"
    )
    logger.log("=" * 80)

    # ── Training loop ────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):

        # Train — weights updated here only
        train_stats = train_one_epoch(
            model, loaders["train"], optimizer, scheduler,
            criterion, scaler, device,
        )

        # Val — monitoring only, zero weight updates
        val_stats = {}
        if "val" in loaders:
            val_stats = validate(
                model, loaders["val"], criterion, device)

        train_losses.append(train_stats["loss"])
        if val_stats:
            val_losses.append(val_stats["loss"])
            val_accs.append(val_stats["accuracy"])

        is_best = val_stats.get("accuracy", 0) > best_val_acc
        if is_best:
            best_val_acc = val_stats["accuracy"]

        # One line per epoch
        logger.log(
            f"{epoch:>6}  "
            f"{train_stats['loss']:>10.4f}  "
            f"{train_stats['cls_loss']:>8.4f}  "
            f"{train_stats['gen_loss']:>8.4f}  "
            f"{train_stats['accuracy']:>9.4f}  "
            f"{val_stats.get('loss', 0):>8.4f}  "
            f"{val_stats.get('accuracy', 0):>7.4f}  "
            f"{val_stats.get('f1', 0):>6.4f}  "
            f"{val_stats.get('auc') or 0:>6.4f}  "
            f"{'*' if is_best else '':>4}"
        )

        save_checkpoint(
            state={
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_acc":         best_val_acc,
                "train_losses":         train_losses,
                "val_losses":           val_losses,
                "val_accuracies":       val_accs,
            },
            is_best        = is_best,
            checkpoint_dir = os.path.join(
                                 args.output_dir, "checkpoints"),
            filename       = f"checkpoint_ep{epoch:03d}.pth",
        )

        if val_stats and early_stopper(val_stats["accuracy"]):
            logger.log(
                f"\nEarly stopping at epoch {epoch}. "
                f"Best val acc: {best_val_acc:.4f}"
            )
            break

    # ── Done ─────────────────────────────────────────────────────────
    logger.log("=" * 80)
    logger.log(f"Best val accuracy : {best_val_acc:.4f}")

    if train_losses:
        plot_training_curves(
            train_losses, val_losses, val_accs, args.output_dir)

    logger.close()


if __name__ == "__main__":
    main()