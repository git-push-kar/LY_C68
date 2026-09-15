# train.py
# ========
# Training pipeline for DeepfakeReasoningModel (InternVL3-2B + LoRA).

# Fixes applied (vs previous version):
#   #4  Tokenization removed from train loop — dataset now pre-tokenizes.
#       train_epoch reads reasoning_ids/attention_mask directly from batch.
#   #5  Eval script was broken (collate_fn, build_transform imports).
#       eval.py is a separate fixed file.
#   #11 lm_seq_len default 512 (covers full tagged trace + <answer> + eos; longer targets are dropped not truncated).
#   #12 epochs_joint default lowered to 2 (reduces overfitting risk).

# Logging added:
#   - All stdout output also written to <output_dir>/logs/train.log
#   - Per-step CSV written to <output_dir>/logs/steps.csv
#   - Per-epoch CSV written to <output_dir>/logs/epochs.csv
#   - Training curves PNG saved after each epoch to <output_dir>/logs/curves.png

# Usage:
#     python train.py ^
#         --dataset_root "C:\\path\\to\\hydrafake" ^
#         --json_root    "C:\\path\\to\\hydrafake\\jsons" ^
#         --model_path   "./models/InternVL3-2B" ^
#         --tokenizer_path "./models/InternVL3-2B" ^
#         --output_dir   "./runs/intern_exp2" ^
#         --batch_size   8 ^
#         --grad_accum   4 ^
#         --epochs_cls   3 ^
#         --epochs_joint 2 ^
#         --lm_seq_len   192 ^
#         --lr           5e-5 ^
#         --num_workers  8

# Corrective fine-tune from ep010 (one line, cmd):
# python train.py --dataset_root "C:\Users\Admin\Desktop\ly project c 68\deepfake_project\datasets\hydrafake" --json_root "C:\Users\Admin\Desktop\ly project c 68\deepfake_project\datasets\hydrafake\jsons" --output_dir ./runs/intern_exp3_corrective --weights_from ./runs/intern_exp2/checkpoints/ep010.pth --epochs_cls 0 --epochs_joint 2 --lr 1e-5 --lm_loss_weight 0.3 --lm_seq_len 512 --batch_size 4 --grad_accum 8 --num_workers 8 --freeze_cls_head --freeze_projector

#python train.py --dataset_root "C:\Users\Admin\Desktop\ly project c 68\deepfake_project\datasets\hydrafake" --json_root "C:\Users\Admin\Desktop\ly project c 68\deepfake_project\datasets\hydrafake\jsons" --output_dir ./runs/intern_exp3_corrective --weights_from ./runs/intern_exp3_corrective/checkpoints/best.pth --epochs_cls 0 --epochs_joint 2 --lr 5e-06 --lm_loss_weight 0.12 --lm_seq_len 512 --batch_size 4 --grad_accum 8 --num_workers 8 --freeze_projector

# Resume
# python train.py --dataset_root "C:\Users\Admin\Desktop\ly project c 68\deepfake_project\datasets\hydrafake" --json_root "C:\Users\Admin\Desktop\ly project c 68\deepfake_project\datasets\hydrafake\jsons" --output_dir ./runs/intern_exp3_corrective --resume ./runs/intern_exp3_corrective/checkpoints/ep001.pth --epochs_cls 0 --epochs_joint 2 --lr 1e-5 --lm_loss_weight 0.3 --lm_seq_len 512 --batch_size 4 --grad_accum 8 --num_workers 8 --freeze_cls_head --freeze_projector
import argparse
import csv
import itertools
import logging
import math
import os
import sys

import torch
import torch.nn as nn
from torch.amp import autocast
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from dataset import build_dataloaders, build_separated_loaders
from model import DeepfakeReasoningModel


# ── Logger ────────────────────────────────────────────────────────────────────

def setup_logger(log_dir: str) -> logging.Logger:
    """
    Write every log() call to stdout AND <log_dir>/train.log simultaneously.
    Append mode — safe to call again on resume.
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(os.path.join(log_dir, "train.log"),
                             mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ── CSV step / epoch loggers ──────────────────────────────────────────────────

class StepCSV:
    """Appends one row per logged step to steps.csv."""
    def __init__(self, path: str):
        self.path    = path
        self.created = os.path.exists(path)

    def write(self, row: dict):
        is_new = not self.created
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if is_new:
                w.writeheader()
                self.created = True
            w.writerow(row)


class EpochCSV:
    """Appends one row per epoch to epochs.csv."""
    def __init__(self, path: str):
        self.path    = path
        self.created = os.path.exists(path)

    def write(self, row: dict):
        is_new = not self.created
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if is_new:
                w.writeheader()
                self.created = True
            w.writerow(row)


# ── Training curves ───────────────────────────────────────────────────────────

def save_curves(epoch_csv_path: str, out_path: str):
    """
    Read epochs.csv and save a training-curves PNG.
    Silently skips if matplotlib is not installed or file has <2 rows.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs, tr_loss, va_loss, tr_acc, va_acc, va_auc = [], [], [], [], [], []
        with open(epoch_csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                epochs.append(int(row["epoch"]))
                tr_loss.append(float(row["tr_loss"]))
                va_loss.append(float(row.get("va_loss", 0)))
                tr_acc.append(float(row["tr_acc"]))
                va_acc.append(float(row.get("va_acc", 0)))
                va_auc.append(float(row.get("va_auc", 0)))

        if len(epochs) < 2:
            return

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(epochs, tr_loss, "o-", label="train loss")
        axes[0].plot(epochs, va_loss, "s--", label="val loss")
        axes[0].set_title("Loss"); axes[0].legend(); axes[0].set_xlabel("epoch")

        axes[1].plot(epochs, tr_acc, "o-", label="train acc")
        axes[1].plot(epochs, va_acc, "s--", label="val acc")
        axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].set_xlabel("epoch")

        axes[2].plot(epochs, va_auc, "o-", color="green", label="val AUC")
        axes[2].set_title("Val AUC"); axes[2].legend(); axes[2].set_xlabel("epoch")

        plt.tight_layout()
        plt.savefig(out_path, dpi=120)
        plt.close()
    except Exception:
        pass   # non-fatal — matplotlib may not be installed


# ── Utils ─────────────────────────────────────────────────────────────────────

class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val   = val
        self.sum  += val * n
        self.count += n
        self.avg   = self.sum / self.count


def compute_metrics(labels, preds, probs=None):
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="binary", zero_division=0)
    auc = 0.0
    if probs and len(set(labels)) > 1:
        try:
            auc = roc_auc_score(labels, probs)
        except Exception:
            pass
    return {"accuracy": acc, "f1": f1, "auc": auc}


def save_checkpoint(state, path, logger):
    """
    Atomic save: write to .tmp first, then os.replace().
    os.replace() is atomic on Windows and Linux — if Ctrl+C fires
    during torch.save() the .tmp is trashed, the previous good
    checkpoint at `path` is untouched. No more corrupted .pth files.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)
    logger.info(f"  Saved: {path}")


def build_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Train one epoch ───────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler,
                device, epoch, logger, step_csv,
                use_lm_loss=False, grad_accum=1):
    """
    Fix #4: reasoning_ids and attention_mask come directly from the batch
    (pre-tokenized at dataset load time). No tokenizer call here.
    """
    model.train()
    loss_m = AverageMeter()
    cls_m  = AverageMeter()
    lm_m   = AverageMeter()
    preds_all, labels_all = [], []

    optimizer.zero_grad(set_to_none=True)
    global_step = (epoch - 1) * len(loader)

    for step, batch in enumerate(loader):
        pv     = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        B      = pv.size(0)

        # Fix #4: read pre-tokenized tensors directly — no CPU tokenizer call
        reasoning_ids  = None
        reasoning_mask = None
        if use_lm_loss:
            reasoning_ids  = batch["reasoning_ids"].to(device, non_blocking=True)
            reasoning_mask = batch["attention_mask"].to(device, non_blocking=True)

        with autocast("cuda", dtype=torch.bfloat16):
            out  = model(pv, labels=labels,
                         reasoning_tokens=reasoning_ids,
                         reasoning_attention_mask=reasoning_mask)
            loss = out["loss"] / grad_accum

        # No GradScaler: it is FP16-only machinery (no BF16 unscale kernel)
        # and pointless under bfloat16 autocast — bf16 shares fp32's
        # exponent range, so gradients never need scaling.
        loss.backward()

        if (step + 1) % grad_accum == 0:
            grads = [p for p in model.parameters() if p.grad is not None]
            nn.utils.clip_grad_norm_(grads, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        loss_m.update(out["loss"].item(), B)
        if out["cls_loss"] is not None:
            cls_m.update(out["cls_loss"].item(), B)
        if out["lm_loss"] is not None:
            lm_m.update(out["lm_loss"].item(), B)

        preds_all.extend(out["cls_logits"].detach().argmax(-1).cpu().tolist())
        labels_all.extend(labels.cpu().tolist())

        # ── Step logging (every 200 steps) ────────────────────────────
        if step % 200 == 0:
            lm_str = (f" | LM {out['lm_loss'].item():.4f}"
                      if out["lm_loss"] is not None else "")
            logger.info(
                f"Ep {epoch} | Step {step}/{len(loader)} "
                f"| Loss {out['loss'].item():.4f} "
                f"| Cls {out['cls_loss'].item():.4f}"
                f"{lm_str}"
            )
            step_csv.write({
                "epoch":    epoch,
                "step":     step,
                "global_step": global_step + step,
                "loss":     round(out["loss"].item(), 6),
                "cls_loss": round(out["cls_loss"].item(), 6) if out["cls_loss"] is not None else "",
                "lm_loss":  round(out["lm_loss"].item(), 6) if out["lm_loss"] is not None else "",
            })

    unique = set(preds_all)
    if len(unique) == 1:
        logger.info(f"  !! COLLAPSE: model predicts only class {list(unique)[0]}")

    m = compute_metrics(labels_all, preds_all)
    return {
        "loss":     loss_m.avg,
        "cls_loss": cls_m.avg,
        "lm_loss":  lm_m.avg,
        "accuracy": m["accuracy"],
        "f1":       m["f1"],
    }


def train_joint_epoch(model, cls_loader, sft_loader, optimizer, scheduler,
                      device, epoch, logger, step_csv,
                      grad_accum=1, lm_loss_weight=0.3):
    """
    Joint epoch with separated pools (Option A):
      joint forward steps = len(sft_loader) = 8,765
      optimizer updates   = joint steps // grad_accum = 1,095 (grad_accum=8)
    Every forward step consumes 1 cls batch (all.json, 4 images) for cls_loss
    and 1 sft batch (sft_36k.json, 4 images) for lm_loss, then
      loss = cls_loss + lm_loss_weight * lm_loss
    CLS loader is cycled (48,320 → 73% coverage per epoch, 145% over 2 epochs).
    """
    model.train()
    loss_m = AverageMeter()
    cls_m  = AverageMeter()
    lm_m   = AverageMeter()
    preds_all, labels_all = [], []

    optimizer.zero_grad(set_to_none=True)
    # Joint epoch defined by SFT length
    joint_steps = len(sft_loader)
    cls_steps_total = len(cls_loader)
    global_step = (epoch - 1) * joint_steps

    cls_iter = iter(cls_loader)
    # We will iterate sft_loader directly and cycle cls_loader as needed
    for step, sft_batch in enumerate(sft_loader):
        # Cycle cls_loader
        try:
            cls_batch = next(cls_iter)
        except StopIteration:
            cls_iter = iter(cls_loader)
            cls_batch = next(cls_iter)

        pv_cls = cls_batch["pixel_values"].to(device, non_blocking=True)
        labels_cls = cls_batch["labels"].to(device, non_blocking=True)

        pv_sft = sft_batch["pixel_values"].to(device, non_blocking=True)
        labels_sft = sft_batch["labels"].to(device, non_blocking=True)
        reasoning_ids = sft_batch["reasoning_ids"].to(device, non_blocking=True)
        reasoning_mask = sft_batch["attention_mask"].to(device, non_blocking=True)
        B = pv_cls.size(0)  # for meters; both batches same B

        with autocast("cuda", dtype=torch.bfloat16):
            out_cls = model(pv_cls, labels=labels_cls)
            out_sft = model(pv_sft, labels=labels_sft,
                            reasoning_tokens=reasoning_ids,
                            reasoning_attention_mask=reasoning_mask)
            cls_loss = out_cls["cls_loss"]
            lm_loss = out_sft["lm_loss"]
            # Preserve exact formulation: L = cls(all.json) + 0.3 * lm(sft_36k)
            loss = cls_loss + lm_loss_weight * lm_loss
            loss = loss / grad_accum

        loss.backward()

        if (step + 1) % grad_accum == 0:
            grads = [p for p in model.parameters() if p.grad is not None]
            nn.utils.clip_grad_norm_(grads, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        loss_m.update((cls_loss + lm_loss_weight * lm_loss).item(), B)
        cls_m.update(cls_loss.item(), B)
        lm_m.update(lm_loss.item(), B)

        preds_all.extend(out_cls["cls_logits"].detach().argmax(-1).cpu().tolist())
        labels_all.extend(labels_cls.cpu().tolist())

        if step % 200 == 0:
            logger.info(
                f"Ep {epoch} | Step {step}/{joint_steps} "
                f"| Loss {(cls_loss + lm_loss_weight*lm_loss).item():.4f} "
                f"| Cls {cls_loss.item():.4f} | LM {lm_loss.item():.4f}"
            )
            step_csv.write({
                "epoch": epoch,
                "step": step,
                "global_step": global_step + step,
                "loss": round((cls_loss + lm_loss_weight*lm_loss).item(), 6),
                "cls_loss": round(cls_loss.item(), 6),
                "lm_loss": round(lm_loss.item(), 6),
            })

    # Flush remainder grads if steps not divisible by grad_accum
    if joint_steps % grad_accum != 0:
        grads = [p for p in model.parameters() if p.grad is not None]
        if grads:
            nn.utils.clip_grad_norm_(grads, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

    unique = set(preds_all)
    if len(unique) == 1:
        logger.info(f"  !! COLLAPSE: model predicts only class {list(unique)[0]}")

    m = compute_metrics(labels_all, preds_all)
    # Log epoch-level consumption — forward pairs vs optimizer updates (grad_accum)
    opt_updates = (joint_steps + grad_accum - 1) // grad_accum
    logger.info(f"  Joint steps: {joint_steps} forward pairs "
                f"({joint_steps * 4} SFT images, {joint_steps * 4} CLS images) | "
                f"Optimizer updates: {opt_updates} (forward pairs // grad_accum={grad_accum}) | "
                f"CLS coverage ~{joint_steps*4/48320*100:.1f}% per epoch — both loaders in every pair, thus every optimizer update")

    return {
        "loss": loss_m.avg,
        "cls_loss": cls_m.avg,
        "lm_loss": lm_m.avg,
        "accuracy": m["accuracy"],
        "f1": m["f1"],
    }


# ── Validate ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, device, logger):
    """Cls-only validation — fast, no LM generation."""
    model.eval()
    loss_m = AverageMeter()
    preds_all, labels_all, probs_all = [], [], []

    for batch in loader:
        pv     = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with autocast("cuda", dtype=torch.bfloat16):
            out = model(pv, labels=labels)   # cls-only forward

        if out["loss"] is not None:
            loss_m.update(out["loss"].item(), pv.size(0))

        logits = out["cls_logits"]
        probs_all.extend(torch.softmax(logits.float(), -1)[:, 1].cpu().tolist())
        preds_all.extend(logits.argmax(-1).cpu().tolist())
        labels_all.extend(labels.cpu().tolist())

    unique = set(preds_all)
    if len(unique) == 1:
        logger.info(f"  !! VAL COLLAPSE: only class {list(unique)[0]}")

    m = compute_metrics(labels_all, preds_all, probs_all)
    m["loss"] = loss_m.avg
    return m


# ── Args ──────────────────────────────────────────────────────────────────────

def _load_config(path="config.yaml"):
    """Load config.yaml if present; CLI always wins. No hard dependency on yaml."""
    cfg = {}
    if path and os.path.isfile(path):
        try:
            import yaml  # type: ignore
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                for k, v in data.items():
                    if not isinstance(v, dict) and v is not None and not str(k).startswith("_"):
                        cfg[k] = v
        except ImportError:
            # yaml not installed — try json fallback (comments in yaml will break json, so just warn)
            pass
        except Exception:
            pass
    return cfg


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",         default="config.yaml",
                   help="Central config file; CLI args override it")
    p.add_argument("--dataset_root",   required=False, default=None)
    p.add_argument("--json_root",      required=False, default=None)
    p.add_argument("--model_path",     default="./models/InternVL3-2B")
    p.add_argument("--tokenizer_path", default="./models/InternVL3-2B")
    p.add_argument("--output_dir",     default="./runs/intern_exp2")
    p.add_argument("--batch_size",     type=int,   default=8)
    p.add_argument("--grad_accum",     type=int,   default=4)
    p.add_argument("--epochs_cls",     type=int,   default=3)
    p.add_argument("--epochs_joint",   type=int,   default=2)  # fix #12
    p.add_argument("--lm_seq_len",     type=int,   default=512,
                   help="Reasoning token length. 512 covers ~95%% of the "
                        "full tagged traces; longer DROPPED "
                        "(never truncated) by the dataset.")
    p.add_argument("--lr",             type=float, default=5e-5)
    p.add_argument("--weight_decay",   type=float, default=0.01)
    p.add_argument("--lora_rank",      type=int,   default=64)
    p.add_argument("--lora_alpha",     type=int,   default=128)
    p.add_argument("--lm_loss_weight", type=float, default=0.3)   # fix #9
    p.add_argument("--num_workers",    type=int,   default=8)
    p.add_argument("--warmup_ratio",   type=float, default=0.05)
    p.add_argument("--patience",       type=int,   default=5)
    p.add_argument("--resume",         default=None)
    p.add_argument("--weights_from",   default=None,
                   help="Init MODEL WEIGHTS ONLY from this checkpoint "
                        "(no optimizer/scheduler restore). Use for "
                        "corrective fine-tuning from a trusted epoch.")
    p.add_argument("--freeze_cls_head",   action="store_true",
                   help="Freeze the classification head (protects "
                        "classifier during corrective fine-tuning).")
    p.add_argument("--freeze_projector", action="store_true",
                   help="Freeze the visual projector (stabilises the "
                        "LLM prefix during corrective fine-tuning).")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    # ── Config file (CLI priority) ────────────────────────────────────
    _cfg = _load_config(args.config)
    # dataset/json are now optional CLI — fill from config if missing
    if not args.dataset_root and "dataset_root" in _cfg:
        args.dataset_root = _cfg["dataset_root"]
    if not args.json_root and "json_root" in _cfg:
        args.json_root = _cfg["json_root"]
    if not args.dataset_root or not args.json_root:
        sys.exit("ERROR: --dataset_root and --json_root required (via CLI or config.yaml)")
    # generic scalar overrides: only if flag not in sys.argv
    for _k, _v in _cfg.items():
        if not hasattr(args, _k):
            continue
        _f1, _f2 = f"--{_k}", f"--{_k.replace('_','-')}"
        if _f1 not in sys.argv and _f2 not in sys.argv:
            setattr(args, _k, _v)
    # boolean flags (store_true): set if config true and not passed
    if not args.freeze_cls_head and _cfg.get("freeze_cls_head"):
        args.freeze_cls_head = True
    if not args.freeze_projector and _cfg.get("freeze_projector"):
        args.freeze_projector = True

    # ── Directories ───────────────────────────────────────────────────
    log_dir  = os.path.join(args.output_dir, "logs")
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(log_dir,  exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    logger    = setup_logger(log_dir)
    step_csv  = StepCSV(os.path.join(log_dir, "steps.csv"))
    epoch_csv = EpochCSV(os.path.join(log_dir, "epochs.csv"))
    curves_path = os.path.join(log_dir, "curves.png")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU   : {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # ── Data ──────────────────────────────────────────────────────────
    # Separated pools: cls (all.json 48,320) / sft (sft_36k.json 35,367) / pref (mipo 3,480 pairs, inactive)
    # pgrpo_8k.json is intentionally never loaded.
    logger.info("Loading data ...")
    try:
        loaders, tokenizer = build_separated_loaders(
            args.dataset_root, args.json_root,
            tokenizer_name=args.tokenizer_path,
            max_text_len=args.lm_seq_len,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        if "cls_train" in loaders:
            logger.info(f"  CLS pool: {len(loaders['cls_train'].dataset)} samples (all.json, bare allowed)")
        if "train" in loaders:
            logger.info(f"  SFT pool: {len(loaders['train'].dataset)} samples (sft_36k, rich only)")
        if "pref" in loaders:
            logger.info(f"  PREF pool: {len(loaders['pref'].dataset)} pairs (mipo_3k, inactive until DPO)")
    except Exception as e:
        logger.info(f"Separated loaders failed ({e}), falling back to legacy build_dataloaders")
        loaders, tokenizer = build_dataloaders(
            args.dataset_root, args.json_root,
            tokenizer_name=args.tokenizer_path,
            max_text_len=args.lm_seq_len,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    if "train" not in loaders:
        logger.info("ERROR: no train split found"); return

    # ── Model ─────────────────────────────────────────────────────────
    logger.info("Building model ...")
    model = DeepfakeReasoningModel(
        model_path=args.model_path,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lm_loss_weight=args.lm_loss_weight,
    ).to(device)

    # ── Corrective init: model weights ONLY ───────────────────────────
    # Unlike --resume, this deliberately does NOT restore optimizer or
    # scheduler state. The old schedule was re-stretched across many
    # resumes (epochs_joint 2→5→7→10), so its moment/schedule state is
    # not trustworthy for a new objective.
    if args.weights_from:
        if not os.path.isfile(args.weights_from):
            raise FileNotFoundError(f"weights_from not found: {args.weights_from}")
        logger.info(f"Loading model weights from: {args.weights_from}")
        ckpt = torch.load(args.weights_from, map_location=device,
                          weights_only=False)
        missing, unexpected = model.load_state_dict(
            ckpt.get("model", ckpt), strict=False)

        critical_prefixes = ("cls_head.", "custom_projector.", "lora_")
        critical_missing = [k for k in missing
                            if k.startswith(critical_prefixes)]
        if critical_missing:
            raise RuntimeError(
                "weights_from checkpoint incompatible — critical keys "
                "missing:\n" + "\n".join(critical_missing[:10]))
        logger.info(f"  epoch={ckpt.get('epoch', '?')}  "
                    f"missing={len(missing)}  unexpected={len(unexpected)}")

    # ── Corrective freezes (protect classifier + prefix semantics) ────
    frozen_names = []
    for n, p_ in model.named_parameters():
        if args.freeze_cls_head and n.startswith("cls_head."):
            p_.requires_grad = False
            frozen_names.append(n)
        elif args.freeze_projector and n.startswith("custom_projector."):
            p_.requires_grad = False
            frozen_names.append(n)
    if frozen_names:
        logger.info(f"Froze {len(frozen_names)} tensors "
                    f"(cls_head={args.freeze_cls_head}, "
                    f"projector={args.freeze_projector})")
        trainable = sum(p_.numel() for p_ in model.parameters()
                        if p_.requires_grad)
        total     = sum(p_.numel() for p_ in model.parameters())
        logger.info(f"Trainable after freeze: {trainable:,} "
                    f"({100*trainable/total:.2f}%)")

    # ── Optimizer ─────────────────────────────────────────────────────
    lora_params   = [p for n, p in model.named_parameters()
                     if "lora_" in n and p.requires_grad]
    custom_params = [p for n, p in model.named_parameters()
                     if any(x in n for x in
                            ["custom_projector", "cls_head", "domain_router"])
                     and p.requires_grad]

    if not lora_params:
        raise RuntimeError("No LoRA params found in optimizer — check model build.")

    logger.info(f"Optimizer groups: "
                f"LoRA={len(lora_params)} tensors, "
                f"custom={len(custom_params)} tensors")

    optimizer = torch.optim.AdamW([
        {"params": lora_params,   "lr": args.lr,      "weight_decay": args.weight_decay},
        {"params": custom_params, "lr": args.lr * 2,  "weight_decay": args.weight_decay},
    ])

    total_epochs = args.epochs_cls + args.epochs_joint
    # Joint forward steps = SFT length (Option A); optimizer updates account for grad_accum
    if "sft_train" in loaders:
        joint_forward_steps = len(loaders["sft_train"])
    else:
        joint_forward_steps = len(loaders["train"])
    optimizer_steps_per_epoch = (joint_forward_steps + args.grad_accum - 1) // args.grad_accum
    steps_per_ep = joint_forward_steps  # for display/logging as forward steps
    total_steps  = total_epochs * steps_per_ep
    total_optimizer_steps = total_epochs * optimizer_steps_per_epoch
    warmup_steps = int(total_optimizer_steps * args.warmup_ratio)
    scheduler    = build_scheduler(optimizer, warmup_steps, total_optimizer_steps)

    best_auc   = 0.0
    no_improve = 0
    start_epoch = 1

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        best_auc    = ckpt.get("best_auc", 0.0)
        start_epoch = ckpt.get("epoch", 0) + 1
        logger.info(f"Resumed from epoch {start_epoch-1}, best AUC={best_auc:.4f}")

    # ── Log config ────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info(f"  batch_size={args.batch_size}  grad_accum={args.grad_accum}  "
                f"eff_batch={args.batch_size * args.grad_accum}")
    logger.info(f"  lm_seq_len={args.lm_seq_len}  lm_loss_weight={args.lm_loss_weight}")
    logger.info(f"  epochs_cls={args.epochs_cls}  epochs_joint={args.epochs_joint}")
    logger.info(f"  lr={args.lr}  lora_rank={args.lora_rank}")
    logger.info(f"  steps/epoch (forward)={steps_per_ep}  total forward steps={total_steps}")
    logger.info(f"  optimizer updates/epoch={optimizer_steps_per_epoch}  total optimizer steps={total_optimizer_steps}  warmup={warmup_steps}")
    logger.info(f"  logs → {log_dir}")
    logger.info("=" * 70)

    hdr = (f"{'Ep':>4} {'Stage':>8} {'TrLoss':>8} {'TrAcc':>7} "
           f"{'VaLoss':>8} {'VaAcc':>7} {'VaAUC':>7} {'Best':>5}")
    logger.info(hdr)
    logger.info("=" * 70)

    # ── Epoch loop ────────────────────────────────────────────────────
    for epoch in range(start_epoch, total_epochs + 1):
        use_lm = epoch > args.epochs_cls
        stage  = "joint" if use_lm else "cls_only"

        # Joint with separated pools: every forward step uses 1 cls batch
        # (all.json, 4 images) + 1 sft batch (sft_36k, 4 images) → L=cls+0.3*lm.
        # Joint forward steps = len(sft_loader) = 8,765; optimizer updates = 8,765 // grad_accum = 1,095 (+tail).
        # CLS loader (12,080 steps) is cycled; each epoch sees 100% of SFT and 73% of CLS (145% over 2 epochs).
        if use_lm and "cls_train" in loaders and "sft_train" in loaders:
            tr = train_joint_epoch(
                model, loaders["cls_train"], loaders["sft_train"],
                optimizer, scheduler, device, epoch, logger, step_csv,
                grad_accum=args.grad_accum, lm_loss_weight=args.lm_loss_weight,
            )
        elif use_lm:
            tr = train_epoch(
                model, loaders["train"], optimizer, scheduler,
                device, epoch, logger, step_csv,
                use_lm_loss=True, grad_accum=args.grad_accum,
            )
        else:
            # cls_only stage — use classification pool if available
            cls_loader = loaders.get("cls_train", loaders["train"])
            tr = train_epoch(
                model, cls_loader, optimizer, scheduler,
                device, epoch, logger, step_csv,
                use_lm_loss=False, grad_accum=args.grad_accum,
            )

        va = {}
        if "val" in loaders:
            va = validate(model, loaders["val"], device, logger)

        cur_auc = va.get("auc", 0.0)
        is_best = cur_auc > best_auc
        if is_best:
            best_auc   = cur_auc
            no_improve = 0
        else:
            no_improve += 1

        # ── Epoch log line ─────────────────────────────────────────────
        line = (f"{epoch:>4} {stage:>8} {tr['loss']:>8.4f} {tr['accuracy']:>7.4f} "
                f"{va.get('loss', 0):>8.4f} {va.get('accuracy', 0):>7.4f} "
                f"{cur_auc:>7.4f} {'*' if is_best else '':>5}")
        logger.info(line)

        # ── Write epoch CSV + update curves ───────────────────────────
        epoch_csv.write({
            "epoch":    epoch,
            "stage":    stage,
            "tr_loss":  round(tr["loss"], 6),
            "tr_cls":   round(tr["cls_loss"], 6),
            "tr_lm":    round(tr["lm_loss"], 6),
            "tr_acc":   round(tr["accuracy"], 6),
            "tr_f1":    round(tr["f1"], 6),
            "va_loss":  round(va.get("loss", 0), 6),
            "va_acc":   round(va.get("accuracy", 0), 6),
            "va_f1":    round(va.get("f1", 0), 6),
            "va_auc":   round(cur_auc, 6),
            "is_best":  int(is_best),
        })
        save_curves(
            os.path.join(log_dir, "epochs.csv"), curves_path
        )

        # ── Checkpoints ───────────────────────────────────────────────
        ckpt_state = {
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_auc":  best_auc,
            "stage":     stage,
        }
        save_checkpoint(
            ckpt_state,
            os.path.join(ckpt_dir, f"ep{epoch:03d}.pth"),
            logger,
        )
        if is_best:
            save_checkpoint(
                {"epoch": epoch, "model": model.state_dict(),
                 "best_auc": best_auc},
                os.path.join(ckpt_dir, "best.pth"),
                logger,
            )

        if no_improve >= args.patience:
            logger.info(f"\nEarly stopping at epoch {epoch}. "
                        f"Best AUC: {best_auc:.4f}")
            break

    logger.info("=" * 70)
    logger.info(f"Done. Best val AUC: {best_auc:.4f}")
    logger.info(f"Checkpoints : {ckpt_dir}")
    logger.info(f"Logs        : {log_dir}")
    logger.info(f"Curves      : {curves_path}")


if __name__ == "__main__":
    main()