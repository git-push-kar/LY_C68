"""
train_dpo.py
============
Direct Preference Optimization (DPO) Stage for DeepfakeReasoningModel v2.
Activates the `mipo_3k.json` preference dataset (3,480 chosen/rejected pairs).

Objective:
  Aligns the structured reasoning outputs (<planning>/<reasoning>/<reflection>/<conclusion>/<answer>)
  with high-quality forensic traces using DPO.

Architecture:
  - Policy Model (pi_theta) : Trainable DeepfakeReasoningModel v2 (initialized from joint checkpoint)
  - Reference Model (pi_ref): Frozen DeepfakeReasoningModel v2 (requires_grad=False, eval())
  - Both models are conditioned on the same visual input via InternVL3's native multi-token representation.

Usage:
  python train_dpo.py ^
      --dataset_root "C:\\path\\to\\hydrafake" ^
      --json_root    "C:\\path\\to\\hydrafake\\jsons" ^
      --model_path   "./models/InternVL3-2B" ^
      --checkpoint   "./runs/intern_v2/checkpoints/best.pth" ^
      --output_dir   "./runs/intern_v2_dpo" ^
      --dpo_epochs   2 ^
      --dpo_lr       1e-6 ^
      --dpo_beta     0.1 ^
      --batch_size   4 ^
      --grad_accum   8
"""

import argparse
import csv
import logging
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import build_preference_loader, HydraFakePreferenceDataset
from model import DeepfakeReasoningModel


# ── Logger ────────────────────────────────────────────────────────────────────

def setup_logger(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("dpo")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(os.path.join(log_dir, "dpo_train.log"), mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ── Time formatting helper ───────────────────────────────────────────────────

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


# ── CSV Loggers ───────────────────────────────────────────────────────────────

class DPOStepCSV:
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


class DPOEpochCSV:
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


class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        if val is None: return
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(1, self.count)


# ── DPO Log-Likelihood Computation ────────────────────────────────────────────

def compute_sequence_logps(
    model: DeepfakeReasoningModel,
    pixel_values: torch.Tensor,
    freq_features: Optional[torch.Tensor],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Computes per-sequence conditional log-likelihood log P(input_ids | pixel_values).
    Routes pixel_values through InternVL3's native visual tokens.
    """
    B = pixel_values.size(0)
    device = pixel_values.device

    # Extract native visual token sequence
    vis_tokens = model._get_native_visual_tokens(pixel_values)  # (B, N_vis, D)
    N_vis = vis_tokens.size(1)

    token_embeds = model.backbone.language_model.get_input_embeddings()(input_ids)
    inputs_embeds = torch.cat([vis_tokens.to(token_embeds.dtype), token_embeds], dim=1)

    prefix_mask = torch.ones((B, N_vis), dtype=attention_mask.dtype, device=device)
    full_mask = torch.cat([prefix_mask, attention_mask], dim=1)

    out = model.backbone.language_model(
        inputs_embeds=inputs_embeds,
        attention_mask=full_mask,
    )
    # Logits for text tokens: predicted starting from (N_vis - 1) to (-1)
    logits = out.logits[:, N_vis - 1 : -1, :]  # (B, L, vocab_size)

    log_probs = F.log_softmax(logits.float(), dim=-1)
    # Gather log prob of actual target tokens
    per_token_logps = torch.gather(log_probs, dim=2, index=input_ids.unsqueeze(2)).squeeze(2)

    # Mask padding tokens
    per_token_logps = per_token_logps * attention_mask.float()
    return per_token_logps.sum(dim=1)  # (B,)


# ── DPO Epoch ─────────────────────────────────────────────────────────────────

def train_dpo_epoch(
    policy_model: DeepfakeReasoningModel,
    ref_model: DeepfakeReasoningModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    logger: logging.Logger,
    step_csv: DPOStepCSV,
    dpo_beta: float = 0.1,
    grad_accum: int = 8,
):
    policy_model.train()
    ref_model.eval()
    start_time = time.time()

    loss_m = AverageMeter()
    chosen_rewards_m = AverageMeter()
    rejected_rewards_m = AverageMeter()
    reward_acc_m = AverageMeter()

    optimizer.zero_grad(set_to_none=True)
    num_steps = len(loader)
    global_step = (epoch - 1) * num_steps

    for step, batch in enumerate(loader):
        pv            = batch["pixel_values"].to(device, non_blocking=True)
        freq          = batch["freq_features"].to(device, non_blocking=True) if "freq_features" in batch else None
        chosen_ids    = batch["chosen_ids"].to(device, non_blocking=True)
        chosen_mask   = batch["chosen_mask"].to(device, non_blocking=True)
        rejected_ids  = batch["rejected_ids"].to(device, non_blocking=True)
        rejected_mask = batch["rejected_mask"].to(device, non_blocking=True)
        B = pv.size(0)

        autocast_ctx = autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.nullcontext()

        # 1. Policy model logps
        with autocast_ctx:
            pi_chosen_logps = compute_sequence_logps(policy_model, pv, freq, chosen_ids, chosen_mask)
            pi_rejected_logps = compute_sequence_logps(policy_model, pv, freq, rejected_ids, rejected_mask)

        # 2. Reference model logps (frozen)
        with torch.no_grad():
            with autocast_ctx:
                ref_chosen_logps = compute_sequence_logps(ref_model, pv, freq, chosen_ids, chosen_mask)
                ref_rejected_logps = compute_sequence_logps(ref_model, pv, freq, rejected_ids, rejected_mask)

        # 3. DPO Loss computation
        pi_logratios = pi_chosen_logps - pi_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        logits = dpo_beta * (pi_logratios - ref_logratios)
        dpo_loss = -F.logsigmoid(logits).mean()

        # Compute implicit rewards for monitoring
        chosen_rewards = dpo_beta * (pi_chosen_logps - ref_chosen_logps).detach()
        rejected_rewards = dpo_beta * (pi_rejected_logps - ref_rejected_logps).detach()
        reward_acc = (chosen_rewards > rejected_rewards).float().mean()

        loss = dpo_loss / grad_accum
        loss.backward()

        if (step + 1) % grad_accum == 0:
            grads = [p for p in policy_model.parameters() if p.grad is not None]
            nn.utils.clip_grad_norm_(grads, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        loss_m.update(dpo_loss.item(), B)
        chosen_rewards_m.update(chosen_rewards.mean().item(), B)
        rejected_rewards_m.update(rejected_rewards.mean().item(), B)
        reward_acc_m.update(reward_acc.item(), B)

        if (step + 1) % 50 == 0 or (step + 1) == num_steps or step == 0:
            elapsed = time.time() - start_time
            steps_done = step + 1
            s_per_step = elapsed / max(1, steps_done)
            eta_sec = (num_steps - steps_done) * s_per_step
            progress_pct = (steps_done / num_steps) * 100

            logger.info(
                f"DPO Ep {epoch} | Step {steps_done:>4}/{num_steps} ({progress_pct:>5.1f}%) "
                f"| {s_per_step:.2f}s/step | Elapsed {format_time(elapsed)} | ETA {format_time(eta_sec)} "
                f"| Loss {dpo_loss.item():.4f} "
                f"| Margin {(chosen_rewards.mean() - rejected_rewards.mean()).item():.4f} "
                f"| Acc {reward_acc.item():.2%}"
            )
            step_csv.write({
                "epoch":           epoch,
                "step":            step,
                "global_step":     global_step + step,
                "step_time_s":     round(s_per_step, 3),
                "dpo_loss":        round(dpo_loss.item(), 6),
                "chosen_reward":   round(chosen_rewards.mean().item(), 6),
                "rejected_reward": round(rejected_rewards.mean().item(), 6),
                "reward_margin":   round((chosen_rewards.mean() - rejected_rewards.mean()).item(), 6),
                "reward_acc":      round(reward_acc.item(), 4),
            })

    if num_steps % grad_accum != 0:
        grads = [p for p in policy_model.parameters() if p.grad is not None]
        if grads:
            nn.utils.clip_grad_norm_(grads, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

    total_epoch_time = time.time() - start_time
    logger.info(f"  DPO Epoch {epoch} completed: {num_steps}/{num_steps} steps | "
                f"Duration: {format_time(total_epoch_time)} ({total_epoch_time:.1f}s) | "
                f"Avg Speed: {total_epoch_time/max(1, num_steps):.3f}s/step")

    return {
        "dpo_loss":        loss_m.avg,
        "chosen_reward":   chosen_rewards_m.avg,
        "rejected_reward": rejected_rewards_m.avg,
        "reward_margin":   chosen_rewards_m.avg - rejected_rewards_m.avg,
        "reward_acc":      reward_acc_m.avg,
        "time_sec":        total_epoch_time,
        "time_str":        format_time(total_epoch_time),
        "steps":           num_steps,
    }


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DPO Training for DeepfakeReasoningModel v2")
    p.add_argument("--dataset_root",  required=True, help="Path to HydraFake root")
    p.add_argument("--json_root",     required=True, help="Path to JSON annotations root")
    p.add_argument("--model_path",    default="./models/InternVL3-2B")
    p.add_argument("--checkpoint",    required=True, help="Checkpoint from joint training stage (e.g. best.pth)")
    p.add_argument("--output_dir",    default="./runs/intern_v2_dpo")
    p.add_argument("--dpo_epochs",    type=int,   default=2)
    p.add_argument("--dpo_lr",        type=float, default=1e-6)
    p.add_argument("--dpo_beta",      type=float, default=0.1)
    p.add_argument("--batch_size",    type=int,   default=4)
    p.add_argument("--grad_accum",    type=int,   default=8)
    p.add_argument("--lm_seq_len",    type=int,   default=512)
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--warmup_ratio",  type=float, default=0.1)
    p.add_argument("--arch_version",  default="v2", choices=["v1", "v2"])
    p.add_argument("--attn_implementation", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    p.add_argument("--no_gradient_checkpointing", action="store_true", help="Disable gradient checkpointing on LLM")
    p.add_argument("--use_gradient_checkpointing", type=lambda x: str(x).lower() in ('true', '1', 'yes'), default=True)
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    log_dir  = os.path.join(args.output_dir, "logs")
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(log_dir,  exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    logger    = setup_logger(log_dir)
    step_csv  = DPOStepCSV(os.path.join(log_dir, "dpo_steps.csv"))
    epoch_csv = DPOEpochCSV(os.path.join(log_dir, "dpo_epochs.csv"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("=" * 86)
    logger.info("DeepfakeReasoningModel v2 — DPO Preference Alignment Stage")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU   : {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    logger.info(f"Checkpoint source : {args.checkpoint}")
    logger.info(f"DPO Beta: {args.dpo_beta} | DPO LR: {args.dpo_lr} | Epochs: {args.dpo_epochs}")
    logger.info("=" * 86)

    # ── Tokenizer & Preference DataLoader ─────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading preference dataset (mipo_3k.json) ...")
    pref_loader, pref_ds = build_preference_loader(
        dataset_root=args.dataset_root,
        json_root=args.json_root,
        tokenizer=tokenizer,
        max_text_len=args.lm_seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    logger.info(f"Loaded {len(pref_ds):,} chosen/rejected pairs for DPO.")

    # ── Policy & Reference Models ─────────────────────────────────────
    use_grad_ckpt = getattr(args, "use_gradient_checkpointing", True) and not getattr(args, "no_gradient_checkpointing", False)
    logger.info("Building Policy Model (trainable) ...")
    policy_model = DeepfakeReasoningModel(
        model_path=args.model_path,
        arch_version=args.arch_version,
        attn_implementation=getattr(args, "attn_implementation", "sdpa"),
        use_gradient_checkpointing=use_grad_ckpt,
    ).to(device)

    logger.info(f"Loading checkpoint weights into Policy Model: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    policy_model.load_state_dict(ckpt.get("model", ckpt), strict=False)

    logger.info("Building Reference Model (frozen copy) ...")
    ref_model = DeepfakeReasoningModel(
        model_path=args.model_path,
        arch_version=args.arch_version,
        attn_implementation=getattr(args, "attn_implementation", "sdpa"),
        use_gradient_checkpointing=False,
    ).to(device)
    ref_model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # Freeze non-LoRA parameters during DPO
    for n, p in policy_model.named_parameters():
        if not ("lora_" in n):
            p.requires_grad = False

    lora_params = [p for n, p in policy_model.named_parameters() if "lora_" in n and p.requires_grad]
    logger.info(f"Policy model trainable LoRA parameters: {sum(p.numel() for p in lora_params):,}")

    optimizer = torch.optim.AdamW(lora_params, lr=args.dpo_lr, weight_decay=0.01)
    steps_per_epoch = len(pref_loader)
    total_steps = args.dpo_epochs * ((steps_per_epoch + args.grad_accum - 1) // args.grad_accum)
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_reward_acc = 0.0

    hdr = f"{'Ep':>4} {'Steps':>10} {'Time':>10} {'DPOLoss':>9} {'RewardMargin':>14} {'RewardAcc':>11} {'Best':>5}"
    logger.info(hdr)
    logger.info("=" * 86)

    # ── DPO Training Loop ─────────────────────────────────────────────
    dpo_start_wall = time.time()
    for epoch in range(1, args.dpo_epochs + 1):
        tr = train_dpo_epoch(
            policy_model=policy_model,
            ref_model=ref_model,
            loader=pref_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            logger=logger,
            step_csv=step_csv,
            dpo_beta=args.dpo_beta,
            grad_accum=args.grad_accum,
        )

        is_best = tr["reward_acc"] > best_reward_acc
        if is_best:
            best_reward_acc = tr["reward_acc"]

        time_display = tr.get("time_str", "N/A")
        steps_display = f"{tr.get('steps', 0)}/{tr.get('steps', 0)}"

        line = (f"{epoch:>4} {steps_display:>10} {time_display:>10} "
                f"{tr['dpo_loss']:>9.4f} {tr['reward_margin']:>14.4f} "
                f"{tr['reward_acc']:>10.2%} {'*' if is_best else '':>5}")
        logger.info(line)

        epoch_csv.write({
            "epoch":           epoch,
            "steps":           tr.get("steps", 0),
            "time_sec":        round(tr.get("time_sec", 0.0), 2),
            "time_str":        time_display,
            "dpo_loss":        round(tr["dpo_loss"], 6),
            "chosen_reward":   round(tr["chosen_reward"], 6),
            "rejected_reward": round(tr["rejected_reward"], 6),
            "reward_margin":   round(tr["reward_margin"], 6),
            "reward_acc":      round(tr["reward_acc"], 4),
            "is_best":         int(is_best),
        })

        ckpt_path = os.path.join(ckpt_dir, f"dpo_ep{epoch:03d}.pth")
        torch.save({
            "epoch":           epoch,
            "model":           policy_model.state_dict(),
            "dpo_loss":        tr["dpo_loss"],
            "reward_acc":      tr["reward_acc"],
            "arch_version":    args.arch_version,
            "epoch_time_s":    tr.get("time_sec", 0.0),
        }, ckpt_path)
        logger.info(f"Saved DPO checkpoint: {ckpt_path}")

        if is_best:
            best_path = os.path.join(ckpt_dir, "dpo_best.pth")
            torch.save({
                "epoch":        epoch,
                "model":        policy_model.state_dict(),
                "reward_acc":   tr["reward_acc"],
                "arch_version": args.arch_version,
            }, best_path)
            logger.info(f"Saved best DPO checkpoint: {best_path}")

    total_dpo_wall = time.time() - dpo_start_wall
    logger.info("=" * 86)
    logger.info(f"DPO Preference Training Complete in {format_time(total_dpo_wall)} ({total_dpo_wall:.1f}s). Best Reward Accuracy: {best_reward_acc:.2%}")
    logger.info(f"Checkpoints directory: {ckpt_dir}")


if __name__ == "__main__":
    main()
