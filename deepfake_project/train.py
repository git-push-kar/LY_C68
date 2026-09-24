"""
train.py
========
Training pipeline for DeepfakeReasoningModel v2 (InternVL3-2B + LoRA + Frequency + Reasoning).

Architecture v2 Features & Fixes:
  #1  Vision LoRA              — InternViT attention/MLP layers fine-tuned with LoRA (r=16, a=32).
  #2  Attention Pooling        — Patch tokens pooled via learned cross-attention query.
  #3  Frequency Branch         — 2D FFT log-magnitude CNN features fused into cls_head.
  #4  Native Visual Tokens     — Full multi-token visual representation for LLM reasoning.
  #5  Self-Consistency Loss    — Bidirectional KL alignment between classifier and LLM <answer> token.
  #6  LM Loss Curriculum       — Linear ramp for lm_loss_weight from start (0.3) to end (0.6).
  #7  Per-Family Diagnostics   — Validation logs per-generator breakdown (cd, cf, cm, id).
  #8  Arch Versioning Guard    — Strict checks on v1 vs v2 checkpoint compatibility.
  #9  Time & Step Tracking     — Precise per-step timing, ETA, speed (s/step), and epoch duration logging.

Logging:
  - All stdout output written to <output_dir>/logs/train.log
  - Per-step metrics written to <output_dir>/logs/steps.csv
  - Per-epoch metrics written to <output_dir>/logs/epochs.csv
  - Training curves PNG saved to <output_dir>/logs/curves.png
"""

import argparse
import csv
import itertools
import logging
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from dataset import build_dataloaders, build_separated_loaders


def _env_preflight():
    """Fail fast with a clear message when run under the wrong interpreter.

    Training requires the `veritas` conda env (CUDA torch + transformers 4.x
    + peft). Running under `venv` (CPU torch, transformers 5.x) or `base`
    (no peft) produces cryptic tracebacks minutes into startup.
    """
    try:
        import transformers
        _major = int(transformers.__version__.split(".")[0])
    except Exception:
        _major = 0
    if _major >= 5:
        sys.exit(
            f"ERROR: transformers {transformers.__version__} detected. "
            "This repo's InternVL3 modeling code requires transformers 4.x. "
            "Run with the veritas env: `conda activate veritas` "
            f"(current interpreter: {sys.executable})"
        )
    try:
        import peft  # noqa: F401
    except ImportError:
        sys.exit(
            "ERROR: `peft` is not installed for this interpreter. "
            "Run with the veritas env: `conda activate veritas` "
            f"(current interpreter: {sys.executable})"
        )
    import torch
    if not torch.cuda.is_available():
        print(
            "WARNING: CUDA not available under this interpreter "
            f"({sys.executable}). Full training needs the veritas env "
            "(`conda activate veritas`) on the RTX A5000 box. "
            "Continuing in CPU prep mode ...",
            flush=True,
        )


_env_preflight()

from model import DeepfakeReasoningModel


# ── Logger ────────────────────────────────────────────────────────────────────

def setup_logger(log_dir: str) -> logging.Logger:
    """Write every log() call to stdout AND <log_dir>/train.log simultaneously."""
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


# ── CSV Loggers ───────────────────────────────────────────────────────────────

class StepCSV:
    """Appends one row per logged step to steps.csv."""
    def __init__(self, path: str):
        self.path = path
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
        self.path = path
        self.created = os.path.exists(path)

    def write(self, row: dict):
        is_new = not self.created
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if is_new:
                w.writeheader()
                self.created = True
            w.writerow(row)


# ── Training Curves ───────────────────────────────────────────────────────────

def save_curves(epoch_csv_path: str, out_path: str):
    """Read epochs.csv and save training curves PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs, tr_loss, va_loss, tr_acc, va_acc, va_auc = [], [], [], [], [], []
        cons_losses = []
        with open(epoch_csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                epochs.append(int(row["epoch"]))
                tr_loss.append(float(row["tr_loss"]))
                va_loss.append(float(row.get("va_loss", 0)))
                tr_acc.append(float(row["tr_acc"]))
                va_acc.append(float(row.get("va_acc", 0)))
                va_auc.append(float(row.get("va_auc", 0)))
                if "tr_consistency" in row and row["tr_consistency"]:
                    try:
                        cons_losses.append(float(row["tr_consistency"]))
                    except ValueError:
                        cons_losses.append(0.0)

        if len(epochs) < 2:
            return

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

        axes[0].plot(epochs, tr_loss, "o-", label="train loss")
        axes[0].plot(epochs, va_loss, "s--", label="val loss")
        axes[0].set_title("Total Loss")
        axes[0].legend()
        axes[0].set_xlabel("Epoch")
        axes[0].grid(True, linestyle="--", alpha=0.5)

        axes[1].plot(epochs, tr_acc, "o-", label="train acc")
        axes[1].plot(epochs, va_acc, "s--", label="val acc")
        axes[1].set_title("Classification Accuracy")
        axes[1].legend()
        axes[1].set_xlabel("Epoch")
        axes[1].grid(True, linestyle="--", alpha=0.5)

        axes[2].plot(epochs, va_auc, "o-", color="green", label="val AUC")
        if any(c > 0 for c in cons_losses):
            axes[2].plot(epochs[:len(cons_losses)], cons_losses, "^-.", color="purple", label="consistency loss")
        axes[2].set_title("Validation AUC & Consistency")
        axes[2].legend()
        axes[2].set_xlabel("Epoch")
        axes[2].grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(out_path, dpi=120)
        plt.close()
    except Exception:
        pass


# ── Time & Metric Utils ───────────────────────────────────────────────────────

def format_time(seconds: float) -> str:
    """Format seconds into a clean human-readable duration (e.g. '1h 24m 10s', '45m 12s', '35.4s')."""
    if seconds < 0:
        return "0.0s"
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        if val is None: return
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(1, self.count)


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
    """Atomic save via .tmp and os.replace()."""
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


def get_answer_token_ids(tokenizer):
    """Detect token IDs for 'real' and 'fake' for self-consistency loss."""
    real_id, fake_id = None, None
    for cand in ["real", " real", "REAL", " Real"]:
        ids = tokenizer.encode(cand, add_special_tokens=False)
        if len(ids) == 1:
            real_id = ids[0]; break
        elif len(ids) > 1:
            real_id = ids[-1]; break

    for cand in ["fake", " fake", "FAKE", " Fake"]:
        ids = tokenizer.encode(cand, add_special_tokens=False)
        if len(ids) == 1:
            fake_id = ids[0]; break
        elif len(ids) > 1:
            fake_id = ids[-1]; break

    return real_id, fake_id


# ── Train Single Epoch (Stage 1: Cls warmup) ──────────────────────────────────

def train_epoch(
    model, loader, optimizer, scheduler,
    device, epoch, logger, step_csv,
    use_lm_loss=False, grad_accum=1, lm_loss_weight=0.3
):
    model.train()
    start_time = time.time()
    loss_m = AverageMeter()
    cls_m  = AverageMeter()
    lm_m   = AverageMeter()
    cons_m = AverageMeter()
    preds_all, labels_all = [], []

    optimizer.zero_grad(set_to_none=True)
    total_steps = len(loader)
    global_step = (epoch - 1) * total_steps

    for step, batch in enumerate(loader):
        pv     = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        freq   = batch["freq_features"].to(device, non_blocking=True) if "freq_features" in batch else None
        B      = pv.size(0)

        reasoning_ids = None
        reasoning_mask = None
        ans_pos = None
        if use_lm_loss:
            reasoning_ids  = batch["reasoning_ids"].to(device, non_blocking=True)
            reasoning_mask = batch["attention_mask"].to(device, non_blocking=True)
            ans_pos        = batch["answer_token_pos"].to(device, non_blocking=True) if "answer_token_pos" in batch else None

        autocast_ctx = autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.nullcontext()
        with autocast_ctx:
            out = model(
                pv, labels=labels,
                reasoning_tokens=reasoning_ids,
                reasoning_attention_mask=reasoning_mask,
                freq_features=freq,
                answer_token_pos=ans_pos
            )
            loss = out["loss"] / grad_accum

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
        if out["consistency_loss"] is not None:
            cons_m.update(out["consistency_loss"].item(), B)

        preds_all.extend(out["cls_logits"].detach().argmax(-1).cpu().tolist())
        labels_all.extend(labels.cpu().tolist())

        if (step + 1) % 200 == 0 or (step + 1) == total_steps or step == 0:
            elapsed = time.time() - start_time
            steps_done = step + 1
            s_per_step = elapsed / max(1, steps_done)
            eta_sec = (total_steps - steps_done) * s_per_step
            progress_pct = (steps_done / total_steps) * 100

            lm_str = (f" | LM {out['lm_loss'].item():.4f}" if out["lm_loss"] is not None else "")
            cons_str = (f" | Cons {out['consistency_loss'].item():.4f}" if out["consistency_loss"] is not None else "")
            logger.info(
                f"Ep {epoch} | Step {steps_done:>5}/{total_steps} ({progress_pct:>5.1f}%) "
                f"| {s_per_step:.2f}s/step | Elapsed {format_time(elapsed)} | ETA {format_time(eta_sec)} "
                f"| Loss {out['loss'].item():.4f} | Cls {out['cls_loss'].item():.4f}"
                f"{lm_str}{cons_str}"
            )
            step_csv.write({
                "epoch":       epoch,
                "step":        step,
                "global_step": global_step + step,
                "step_time_s": round(s_per_step, 3),
                "loss":        round(out["loss"].item(), 6),
                "cls_loss":    round(out["cls_loss"].item(), 6) if out["cls_loss"] is not None else "",
                "lm_loss":     round(out["lm_loss"].item(), 6) if out["lm_loss"] is not None else "",
                "consistency": round(out["consistency_loss"].item(), 6) if out["consistency_loss"] is not None else "",
            })

    total_epoch_time = time.time() - start_time
    unique = set(preds_all)
    if len(unique) == 1:
        logger.warning(f"  !! COLLAPSE WARNING: model predicts only class {list(unique)[0]}")

    if cons_m.avg > 1.5 and cls_m.avg < 0.3:
        logger.warning(
            f"  !! CONSISTENCY DIVERGENCE: consistency_loss ({cons_m.avg:.4f}) is high "
            f"while cls_loss ({cls_m.avg:.4f}) is low. Classifier and LLM answer predictions disagree systematically."
        )

    m = compute_metrics(labels_all, preds_all)
    logger.info(f"  Stage 1 Epoch {epoch} completed: {total_steps}/{total_steps} steps | "
                f"Duration: {format_time(total_epoch_time)} ({total_epoch_time:.1f}s) | "
                f"Avg Speed: {total_epoch_time/max(1, total_steps):.3f}s/step")

    return {
        "loss":             loss_m.avg,
        "cls_loss":         cls_m.avg,
        "lm_loss":          lm_m.avg,
        "consistency_loss": cons_m.avg if cons_m.count > 0 else 0.0,
        "accuracy":         m["accuracy"],
        "f1":               m["f1"],
        "time_sec":         total_epoch_time,
        "time_str":         format_time(total_epoch_time),
        "steps":            total_steps,
    }


# ── Train Joint Epoch (Stage 2: Joint cls + SFT reasoning + Consistency) ──────

def train_joint_epoch(
    model, cls_loader, sft_loader, optimizer, scheduler,
    device, epoch, logger, step_csv,
    grad_accum=1, lm_loss_weight=0.3, consistency_loss_weight=0.05
):
    """
    Joint training epoch consuming paired batches:
      - 1 CLS batch (all.json, 48k pool) for classification loss
      - 1 SFT batch (sft_36k.json, 35k pool) for reasoning loss & consistency
    Loss = cls_loss + lm_loss_weight * lm_loss + consistency_loss_weight * consistency_loss
    """
    model.train()
    start_time = time.time()
    loss_m = AverageMeter()
    cls_m  = AverageMeter()
    lm_m   = AverageMeter()
    cons_m = AverageMeter()
    preds_all, labels_all = [], []

    optimizer.zero_grad(set_to_none=True)
    joint_steps = len(sft_loader)
    global_step = (epoch - 1) * joint_steps

    cls_iter = iter(cls_loader)
    for step, sft_batch in enumerate(sft_loader):
        # Cycle CLS loader
        try:
            cls_batch = next(cls_iter)
        except StopIteration:
            cls_iter = iter(cls_loader)
            cls_batch = next(cls_iter)

        pv_cls     = cls_batch["pixel_values"].to(device, non_blocking=True)
        labels_cls = cls_batch["labels"].to(device, non_blocking=True)
        freq_cls   = cls_batch["freq_features"].to(device, non_blocking=True) if "freq_features" in cls_batch else None

        pv_sft         = sft_batch["pixel_values"].to(device, non_blocking=True)
        labels_sft     = sft_batch["labels"].to(device, non_blocking=True)
        freq_sft       = sft_batch["freq_features"].to(device, non_blocking=True) if "freq_features" in sft_batch else None
        reasoning_ids  = sft_batch["reasoning_ids"].to(device, non_blocking=True)
        reasoning_mask = sft_batch["attention_mask"].to(device, non_blocking=True)
        ans_pos        = sft_batch["answer_token_pos"].to(device, non_blocking=True) if "answer_token_pos" in sft_batch else None
        B = pv_cls.size(0)

        autocast_ctx = autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.nullcontext()
        with autocast_ctx:
            # 1. Classification forward on CLS batch
            out_cls = model(pv_cls, labels=labels_cls, freq_features=freq_cls)
            cls_loss = out_cls["cls_loss"]

            # 2. Reasoning forward on SFT batch (with consistency loss)
            out_sft = model(
                pv_sft, labels=labels_sft,
                reasoning_tokens=reasoning_ids,
                reasoning_attention_mask=reasoning_mask,
                freq_features=freq_sft,
                answer_token_pos=ans_pos
            )
            lm_loss = out_sft["lm_loss"]
            consistency_loss = out_sft["consistency_loss"]

            # Combined joint objective
            step_loss = cls_loss + lm_loss_weight * lm_loss
            if consistency_loss is not None and consistency_loss_weight > 0:
                step_loss = step_loss + consistency_loss_weight * consistency_loss

            loss = step_loss / grad_accum

        loss.backward()

        if (step + 1) % grad_accum == 0:
            grads = [p for p in model.parameters() if p.grad is not None]
            nn.utils.clip_grad_norm_(grads, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        loss_m.update(step_loss.item(), B)
        cls_m.update(cls_loss.item(), B)
        lm_m.update(lm_loss.item(), B)
        if consistency_loss is not None:
            cons_m.update(consistency_loss.item(), B)

        preds_all.extend(out_cls["cls_logits"].detach().argmax(-1).cpu().tolist())
        labels_all.extend(labels_cls.cpu().tolist())

        if (step + 1) % 200 == 0 or (step + 1) == joint_steps or step == 0:
            elapsed = time.time() - start_time
            steps_done = step + 1
            s_per_step = elapsed / max(1, steps_done)
            eta_sec = (joint_steps - steps_done) * s_per_step
            progress_pct = (steps_done / joint_steps) * 100

            cons_str = f" | Cons {consistency_loss.item():.4f}" if consistency_loss is not None else ""
            logger.info(
                f"Ep {epoch} | Step {steps_done:>5}/{joint_steps} ({progress_pct:>5.1f}%) "
                f"| {s_per_step:.2f}s/step | Elapsed {format_time(elapsed)} | ETA {format_time(eta_sec)} "
                f"| Loss {step_loss.item():.4f} "
                f"| Cls {cls_loss.item():.4f} | LM {lm_loss.item():.4f} (w={lm_loss_weight:.2f})"
                f"{cons_str}"
            )
            step_csv.write({
                "epoch":       epoch,
                "step":        step,
                "global_step": global_step + step,
                "step_time_s": round(s_per_step, 3),
                "loss":        round(step_loss.item(), 6),
                "cls_loss":    round(cls_loss.item(), 6),
                "lm_loss":     round(lm_loss.item(), 6),
                "consistency": round(consistency_loss.item(), 6) if consistency_loss is not None else "",
            })

    # Flush remainder grads
    if joint_steps % grad_accum != 0:
        grads = [p for p in model.parameters() if p.grad is not None]
        if grads:
            nn.utils.clip_grad_norm_(grads, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

    total_epoch_time = time.time() - start_time
    unique = set(preds_all)
    if len(unique) == 1:
        logger.warning(f"  !! COLLAPSE WARNING: model predicts only class {list(unique)[0]}")

    if cons_m.avg > 1.5 and cls_m.avg < 0.3:
        logger.warning(
            f"  !! CONSISTENCY DIVERGENCE: consistency_loss ({cons_m.avg:.4f}) is diverging from cls_loss ({cls_m.avg:.4f})."
        )

    m = compute_metrics(labels_all, preds_all)
    opt_updates = (joint_steps + grad_accum - 1) // grad_accum
    logger.info(f"  Stage 2 Joint Epoch {epoch} completed: {joint_steps}/{joint_steps} pairs | "
                f"Optimizer updates: {opt_updates} | Duration: {format_time(total_epoch_time)} ({total_epoch_time:.1f}s) | "
                f"Avg Speed: {total_epoch_time/max(1, joint_steps):.3f}s/step | "
                f"Curriculum LM weight: {lm_loss_weight:.3f}")

    return {
        "loss":             loss_m.avg,
        "cls_loss":         cls_m.avg,
        "lm_loss":          lm_m.avg,
        "consistency_loss": cons_m.avg if cons_m.count > 0 else 0.0,
        "accuracy":         m["accuracy"],
        "f1":               m["f1"],
        "time_sec":         total_epoch_time,
        "time_str":         format_time(total_epoch_time),
        "steps":            joint_steps,
    }


# ── Validation with Per-Family Diagnostics (2.8) ──────────────────────────────

@torch.no_grad()
def validate(model, loader, device, logger) -> Dict[str, Union[float, dict]]:
    """Cls-only validation with per-generator-family diagnostics."""
    model.eval()
    val_start = time.time()
    loss_m = AverageMeter()
    preds_all, labels_all, probs_all = [], [], []
    family_stats = {}  # ftype -> {"preds": [], "labels": []}

    for batch in loader:
        pv     = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        freq   = batch["freq_features"].to(device, non_blocking=True) if "freq_features" in batch else None
        ftypes = batch.get("forgery_type", ["unknown"] * pv.size(0))

        autocast_ctx = autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.nullcontext()
        with autocast_ctx:
            out = model(pv, labels=labels, freq_features=freq)

        if out["loss"] is not None:
            loss_m.update(out["loss"].item(), pv.size(0))

        logits = out["cls_logits"].float()
        probs = F.softmax(logits, -1)[:, 1].cpu().tolist()
        preds = logits.argmax(-1).cpu().tolist()
        targets = labels.cpu().tolist()

        probs_all.extend(probs)
        preds_all.extend(preds)
        labels_all.extend(targets)

        for p, t, ft in zip(preds, targets, ftypes):
            if ft not in family_stats:
                family_stats[ft] = {"preds": [], "labels": []}
            family_stats[ft]["preds"].append(p)
            family_stats[ft]["labels"].append(t)

    unique = set(preds_all)
    if len(unique) == 1:
        logger.warning(f"  !! VAL COLLAPSE: only class {list(unique)[0]}")

    val_time = time.time() - val_start
    m = compute_metrics(labels_all, preds_all, probs_all)
    m["loss"] = loss_m.avg
    m["val_time_sec"] = val_time

    # Log per-generator family diagnostic breakdown
    logger.info("  " + "-" * 60)
    logger.info(f"  Validation Per-Family Diagnostics ({len(family_stats)} categories, elapsed {format_time(val_time)}):")
    for ft, d in sorted(family_stats.items()):
        f_acc = accuracy_score(d["labels"], d["preds"])
        f_n = len(d["labels"])
        f_r = d["labels"].count(0)
        f_f = d["labels"].count(1)
        logger.info(f"    - Family {ft:<12} : Acc={f_acc:>6.2%} | N={f_n:>4} (Real={f_r}, Fake={f_f})")
    logger.info("  " + "-" * 60)

    return m


# ── Args & Config ─────────────────────────────────────────────────────────────

def _load_config(path="config.yaml"):
    """Load config.yaml if present; CLI always wins."""
    cfg = {}
    if path and os.path.isfile(path):
        try:
            import yaml
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                for k, v in data.items():
                    if not isinstance(v, dict) and v is not None and not str(k).startswith("_"):
                        cfg[k] = v
        except ImportError:
            pass
        except Exception:
            pass
    return cfg


def parse_args():
    p = argparse.ArgumentParser(description="Train DeepfakeReasoningModel v2")
    p.add_argument("--config",                  default="config.yaml")
    p.add_argument("--dataset_root",            required=False, default=None)
    p.add_argument("--json_root",               required=False, default=None)
    p.add_argument("--model_path",              default="./models/InternVL3-2B")
    p.add_argument("--tokenizer_path",          default="./models/InternVL3-2B")
    p.add_argument("--output_dir",              default="./runs/intern_v2")
    p.add_argument("--batch_size",              type=int,   default=4)
    p.add_argument("--grad_accum",              type=int,   default=8)
    p.add_argument("--epochs_cls",              type=int,   default=3)
    p.add_argument("--epochs_joint",            type=int,   default=2)
    p.add_argument("--lm_seq_len",              type=int,   default=512)
    p.add_argument("--lr",                      type=float, default=5e-5)
    p.add_argument("--weight_decay",            type=float, default=0.01)
    p.add_argument("--lora_rank",               type=int,   default=64)
    p.add_argument("--lora_alpha",              type=int,   default=128)
    p.add_argument("--lora_dropout",            type=float, default=0.05)
    p.add_argument("--vision_lora_rank",        type=int,   default=16)
    p.add_argument("--vision_lora_alpha",       type=int,   default=32)
    p.add_argument("--vision_lora_dropout",     type=float, default=0.05)
    p.add_argument("--lm_loss_weight_start",    type=float, default=0.3)
    p.add_argument("--lm_loss_weight_end",      type=float, default=0.6)
    p.add_argument("--lm_loss_weight",          type=float, default=None,
                   help="Legacy flat LM loss weight. If set, overrides curriculum.")
    p.add_argument("--consistency_loss_weight", type=float, default=0.05)
    p.add_argument("--use_focal_loss",          action="store_true")
    p.add_argument("--focal_gamma",             type=float, default=2.0)
    p.add_argument("--arch_version",            default="v2", choices=["v1", "v2"])
    p.add_argument("--num_workers",             type=int,   default=8)
    p.add_argument("--warmup_ratio",            type=float, default=0.05)
    p.add_argument("--patience",                type=int,   default=5)
    p.add_argument("--resume",                  default=None)
    p.add_argument("--weights_from",            default=None)
    p.add_argument("--freeze_cls_head",         action="store_true")
    p.add_argument("--freeze_projector",        action="store_true")
    p.add_argument("--attn_implementation",     default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    p.add_argument("--no_gradient_checkpointing", action="store_true", help="Disable gradient checkpointing on LLM")
    p.add_argument("--use_gradient_checkpointing", type=lambda x: str(x).lower() in ('true', '1', 'yes'), default=True)
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    _cfg = _load_config(args.config)

    if not args.dataset_root and "dataset_root" in _cfg:
        args.dataset_root = _cfg["dataset_root"]
    if not args.json_root and "json_root" in _cfg:
        args.json_root = _cfg["json_root"]
    if not args.dataset_root or not args.json_root:
        sys.exit("ERROR: --dataset_root and --json_root required (via CLI or config.yaml)")

    for _k, _v in _cfg.items():
        if not hasattr(args, _k):
            continue
        _f1, _f2 = f"--{_k}", f"--{_k.replace('_','-')}"
        if _f1 not in sys.argv and _f2 not in sys.argv:
            setattr(args, _k, _v)

    if not args.freeze_cls_head and _cfg.get("freeze_cls_head"):
        args.freeze_cls_head = True
    if not args.freeze_projector and _cfg.get("freeze_projector"):
        args.freeze_projector = True

    # Directories & Logging
    log_dir  = os.path.join(args.output_dir, "logs")
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(log_dir,  exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    logger    = setup_logger(log_dir)
    step_csv  = StepCSV(os.path.join(log_dir, "steps.csv"))
    epoch_csv = EpochCSV(os.path.join(log_dir, "epochs.csv"))
    curves_path = os.path.join(log_dir, "curves.png")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("=" * 86)
    logger.info(f"DeepfakeReasoningModel v2 Training Pipeline")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU   : {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    else:
        logger.info("Running on CPU (Prep mode for GPU execution)")
    logger.info("=" * 86)

    # ── Tokenizer & Data ──────────────────────────────────────────────
    logger.info("Loading tokenizer & data ...")
    try:
        loaders, tokenizer = build_separated_loaders(
            args.dataset_root, args.json_root,
            tokenizer_name=args.tokenizer_path,
            max_text_len=args.lm_seq_len,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        if "cls_train" in loaders:
            logger.info(f"  CLS pool : {len(loaders['cls_train'].dataset):,} samples (all.json, bare allowed)")
        if "sft_train" in loaders:
            logger.info(f"  SFT pool : {len(loaders['sft_train'].dataset):,} samples (sft_36k.json, rich only)")
        if "pref" in loaders:
            logger.info(f"  PREF pool: {len(loaders['pref'].dataset):,} pairs (mipo_3k.json, ready for DPO)")
    except Exception as e:
        logger.info(f"Separated loaders fallback ({e}) -> legacy build_dataloaders")
        loaders, tokenizer = build_dataloaders(
            args.dataset_root, args.json_root,
            tokenizer_name=args.tokenizer_path,
            max_text_len=args.lm_seq_len,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    if "train" not in loaders:
        logger.error("ERROR: no train loader resolved."); return

    # Detect answer token IDs for consistency loss
    real_id, fake_id = get_answer_token_ids(tokenizer)
    logger.info(f"Answer token IDs: real={real_id}, fake={fake_id}")

    use_grad_ckpt = getattr(args, "use_gradient_checkpointing", True) and not getattr(args, "no_gradient_checkpointing", False)
    model = DeepfakeReasoningModel(
        model_path=args.model_path,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        vision_lora_rank=args.vision_lora_rank,
        vision_lora_alpha=args.vision_lora_alpha,
        vision_lora_dropout=args.vision_lora_dropout,
        cls_dropout=0.3,
        lm_loss_weight=args.lm_loss_weight_start if args.lm_loss_weight is None else args.lm_loss_weight,
        consistency_loss_weight=args.consistency_loss_weight,
        use_focal_loss=args.use_focal_loss,
        focal_gamma=args.focal_gamma,
        arch_version=args.arch_version,
        real_token_id=real_id,
        fake_token_id=fake_id,
        attn_implementation=getattr(args, "attn_implementation", "sdpa"),
        use_gradient_checkpointing=use_grad_ckpt,
    ).to(device)

    # ── Checkpoint Loading Guard (3.0) ────────────────────────────────
    if args.weights_from:
        if not os.path.isfile(args.weights_from):
            raise FileNotFoundError(f"weights_from checkpoint not found: {args.weights_from}")
        logger.info(f"Loading weights from: {args.weights_from}")
        ckpt = torch.load(args.weights_from, map_location=device, weights_only=False)
        state_dict = ckpt.get("model", ckpt)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        v2_new_prefixes = ("attn_pool.", "freq_branch.", "backbone.vision_model.")
        v2_missing = [k for k in missing if any(k.startswith(p) or p in k for p in v2_new_prefixes)]

        if v2_missing and args.arch_version == "v2":
            logger.warning("=" * 86)
            logger.warning("  [CHECKPOINT COMPATIBILITY NOTICE]")
            logger.warning("  Loaded a checkpoint lacking v2 keys into a v2 model.")
            logger.warning(f"  Missing v2 keys (will be randomly initialized): {len(v2_missing)} tensors")
            logger.warning("    - Examples: " + ", ".join(v2_missing[:4]))
            logger.warning("  RECOMMENDATION: Because v2 updates both classification-feature and")
            logger.warning("  reasoning-input paths, a full retrain of Stage 1 (cls warm-up) and")
            logger.warning("  Stage 2 (joint) is strongly recommended to avoid inconsistent partially-adapted weights.")
            logger.warning("=" * 86)

    # ── Corrective Freezes ────────────────────────────────────────────
    frozen_names = []
    for n, p_ in model.named_parameters():
        if args.freeze_cls_head and n.startswith("cls_head."):
            p_.requires_grad = False
            frozen_names.append(n)
        elif args.freeze_projector and n.startswith("custom_projector."):
            p_.requires_grad = False
            frozen_names.append(n)
    if frozen_names:
        logger.info(f"Froze {len(frozen_names)} tensors (cls_head={args.freeze_cls_head}, projector={args.freeze_projector})")

    # ── Optimizer ─────────────────────────────────────────────────────
    lora_params = [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]
    custom_params = [
        p for n, p in model.named_parameters()
        if any(x in n for x in ["custom_projector", "cls_head", "domain_router", "attn_pool", "freq_branch"])
        and p.requires_grad
    ]

    if not lora_params:
        raise RuntimeError("FATAL: No LoRA params found in optimizer.")

    logger.info(f"Optimizer param groups: LoRA={len(lora_params)} tensors, Custom={len(custom_params)} tensors")
    optimizer = torch.optim.AdamW([
        {"params": lora_params,   "lr": args.lr,     "weight_decay": args.weight_decay},
        {"params": custom_params, "lr": args.lr * 2, "weight_decay": args.weight_decay},
    ])

    total_epochs = args.epochs_cls + args.epochs_joint
    joint_forward_steps = len(loaders.get("sft_train", loaders["train"]))
    optimizer_steps_per_epoch = (joint_forward_steps + args.grad_accum - 1) // args.grad_accum
    total_optimizer_steps = total_epochs * optimizer_steps_per_epoch
    warmup_steps = int(total_optimizer_steps * args.warmup_ratio)
    scheduler = build_scheduler(optimizer, warmup_steps, total_optimizer_steps)

    best_auc = 0.0
    no_improve = 0
    start_epoch = 1

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        best_auc = ckpt.get("best_auc", 0.0)
        start_epoch = ckpt.get("epoch", 0) + 1
        logger.info(f"Resumed from epoch {start_epoch-1}, best AUC={best_auc:.4f}")

    # ── Run Configuration Summary ─────────────────────────────────────
    logger.info("=" * 86)
    logger.info(f"  batch_size={args.batch_size}  grad_accum={args.grad_accum}  eff_batch={args.batch_size * args.grad_accum}")
    logger.info(f"  lm_seq_len={args.lm_seq_len}  consistency_weight={args.consistency_loss_weight}")
    logger.info(f"  lm_loss_curriculum: start={args.lm_loss_weight_start} -> end={args.lm_loss_weight_end}")
    logger.info(f"  epochs_cls={args.epochs_cls}  epochs_joint={args.epochs_joint}  total_epochs={total_epochs}")
    logger.info(f"  lr={args.lr}  llm_lora_rank={args.lora_rank}  vision_lora_rank={args.vision_lora_rank}")
    logger.info(f"  attn_impl={getattr(args, 'attn_implementation', 'sdpa')}  grad_ckpt={use_grad_ckpt}")
    logger.info(f"  steps/epoch={joint_forward_steps}  optimizer_updates/epoch={optimizer_steps_per_epoch}")
    logger.info(f"  logs -> {log_dir}")
    logger.info("=" * 86)

    hdr = (f"{'Ep':>4} {'Stage':>8} {'Steps':>11} {'Time':>10} {'TrLoss':>8} {'TrAcc':>7} "
           f"{'VaLoss':>8} {'VaAcc':>7} {'VaAUC':>7} {'LM_W':>6} {'Best':>5}")
    logger.info(hdr)
    logger.info("=" * 86)

    # ── Epoch Loop ────────────────────────────────────────────────────
    train_start_wall = time.time()
    for epoch in range(start_epoch, total_epochs + 1):
        use_lm = epoch > args.epochs_cls
        stage  = "joint" if use_lm else "cls_only"

        # Linear LM weight curriculum across joint epochs (2.6)
        if use_lm:
            if args.lm_loss_weight is not None:
                cur_lm_weight = args.lm_loss_weight
            else:
                joint_idx = epoch - args.epochs_cls - 1
                total_joint_span = max(1, args.epochs_joint - 1)
                cur_lm_weight = args.lm_loss_weight_start + (
                    args.lm_loss_weight_end - args.lm_loss_weight_start
                ) * (joint_idx / total_joint_span)
                cur_lm_weight = min(max(cur_lm_weight, args.lm_loss_weight_start), args.lm_loss_weight_end)
        else:
            cur_lm_weight = 0.0

        model.lm_loss_weight = cur_lm_weight

        if use_lm and "cls_train" in loaders and "sft_train" in loaders:
            tr = train_joint_epoch(
                model, loaders["cls_train"], loaders["sft_train"],
                optimizer, scheduler, device, epoch, logger, step_csv,
                grad_accum=args.grad_accum,
                lm_loss_weight=cur_lm_weight,
                consistency_loss_weight=args.consistency_loss_weight,
            )
        elif use_lm:
            tr = train_epoch(
                model, loaders["train"], optimizer, scheduler,
                device, epoch, logger, step_csv,
                use_lm_loss=True, grad_accum=args.grad_accum,
                lm_loss_weight=cur_lm_weight,
            )
        else:
            cls_loader = loaders.get("cls_train", loaders["train"])
            tr = train_epoch(
                model, cls_loader, optimizer, scheduler,
                device, epoch, logger, step_csv,
                use_lm_loss=False, grad_accum=args.grad_accum,
            )

        va = {}
        if "val" in loaders:
            va = validate(model, loaders["val"], device, logger)
        else:
            logger.info("  [val] no val loader (val images absent) — tracking best by train acc")

        # When val is absent (val images not in this checkout), fall back to
        # train accuracy so best.pth is still saved and early stopping works.
        cur_auc = va.get("auc", 0.0) if "val" in loaders else float(tr.get("accuracy", 0.0))
        is_best = cur_auc > best_auc
        if is_best:
            best_auc   = cur_auc
            no_improve = 0
        else:
            no_improve += 1

        steps_display = f"{tr.get('steps', 0)}/{tr.get('steps', 0)}"
        time_display = tr.get("time_str", "N/A")
        line = (f"{epoch:>4} {stage:>8} {steps_display:>11} {time_display:>10} "
                f"{tr['loss']:>8.4f} {tr['accuracy']:>7.4f} "
                f"{va.get('loss', 0):>8.4f} {va.get('accuracy', 0):>7.4f} "
                f"{cur_auc:>7.4f} {cur_lm_weight:>6.2f} {'*' if is_best else '':>5}")
        logger.info(line)

        epoch_csv.write({
            "epoch":          epoch,
            "stage":          stage,
            "steps":          tr.get("steps", 0),
            "time_sec":       round(tr.get("time_sec", 0.0), 2),
            "time_str":       time_display,
            "tr_loss":        round(tr["loss"], 6),
            "tr_cls":         round(tr["cls_loss"], 6),
            "tr_lm":          round(tr["lm_loss"], 6),
            "tr_consistency": round(tr.get("consistency_loss", 0.0), 6),
            "tr_acc":         round(tr["accuracy"], 6),
            "tr_f1":          round(tr["f1"], 6),
            "va_loss":        round(va.get("loss", 0), 6),
            "va_acc":         round(va.get("accuracy", 0), 6),
            "va_f1":          round(va.get("f1", 0), 6),
            "va_auc":         round(cur_auc, 6),
            "lm_loss_weight": round(cur_lm_weight, 4),
            "is_best":        int(is_best),
        })
        save_curves(os.path.join(log_dir, "epochs.csv"), curves_path)

        ckpt_state = {
            "epoch":        epoch,
            "model":        model.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "scheduler":    scheduler.state_dict(),
            "best_auc":     best_auc,
            "stage":        stage,
            "arch_version": args.arch_version,
            "epoch_time_s": tr.get("time_sec", 0.0),
        }
        save_checkpoint(ckpt_state, os.path.join(ckpt_dir, f"ep{epoch:03d}.pth"), logger)
        if is_best:
            save_checkpoint(
                {"epoch": epoch, "model": model.state_dict(), "best_auc": best_auc, "arch_version": args.arch_version},
                os.path.join(ckpt_dir, "best.pth"), logger
            )

        if no_improve >= args.patience:
            logger.info(f"\nEarly stopping at epoch {epoch}. Best AUC: {best_auc:.4f}")
            break

    total_training_wall = time.time() - train_start_wall
    logger.info("=" * 86)
    logger.info(f"Training Complete in {format_time(total_training_wall)} ({total_training_wall:.1f}s). Best val AUC: {best_auc:.4f}")
    logger.info(f"Checkpoints : {ckpt_dir}")
    logger.info(f"Logs        : {log_dir}")
    logger.info(f"Curves      : {curves_path}")


if __name__ == "__main__":
    main()