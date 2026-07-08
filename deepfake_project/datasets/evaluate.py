"""
evaluate.py
===========
Final benchmark on the TEST split only.
Never run on val — that was already used during training for
early stopping. Only the test split gives a truly unseen result.

Usage:
    python evaluate.py \
        --checkpoint   ./runs/exp3/checkpoints/best_model.pth \
        --dataset_root /path/to/hydrafake \
        --json_root    /path/to/hydrafake/jsons
"""

import argparse
import os
import torch
from torch.amp import autocast

from dataset import build_dataloaders
from model import HydraFakeNet
from utils import Logger, compute_metrics, load_checkpoint, set_seed


@torch.no_grad()
def evaluate_test(model, loader, device, logger: Logger):
    model.eval()
    preds_all, labels_all, probs_all = [], [], []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with autocast("cuda"):
            out        = model(images)
            cls_logits = out["cls_logits"]

        probs_all.extend(
            torch.softmax(cls_logits, -1)[:, 1].cpu().tolist())
        preds_all.extend(cls_logits.argmax(-1).cpu().tolist())
        labels_all.extend(labels.cpu().tolist())

    m = compute_metrics(labels_all, preds_all, probs_all)
    logger.log("\n── Test results ──")
    logger.log(f"  Accuracy : {m['accuracy']:.4f}")
    logger.log(f"  F1       : {m['f1']:.4f}")
    if m.get("auc"):
        logger.log(f"  AUC-ROC  : {m['auc']:.4f}")
    logger.log(f"\n{m['report']}")
    return m


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",       required=True)
    p.add_argument("--dataset_root",     required=True)
    p.add_argument("--json_root",        required=True)
    p.add_argument("--output_dir",       default="./eval_results")
    p.add_argument("--tokenizer",        default="bert-base-uncased")
    p.add_argument("--image_size",       type=int, default=224)
    p.add_argument("--batch_size",       type=int, default=64)
    p.add_argument("--num_workers",      type=int, default=8)
    p.add_argument("--clip_model",
                   default="openai/clip-vit-base-patch32")
    p.add_argument("--freeze_mode",      default="none",
                   choices=["all", "partial", "none"])
    p.add_argument("--model_dim",        type=int, default=512)
    p.add_argument("--cls_depth",        type=int, default=4)
    p.add_argument("--cls_heads",        type=int, default=8)
    p.add_argument("--cls_queries",      type=int, default=8)
    p.add_argument("--reasoning_hidden", type=int, default=512)
    p.add_argument("--reasoning_layers", type=int, default=2)
    p.add_argument("--seed",             type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    logger = Logger(args.output_dir, "eval.log")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    logger.log(f"Device : {device}")

    loaders, tokenizer = build_dataloaders(
        dataset_root   = args.dataset_root,
        json_root      = args.json_root,
        tokenizer_name = args.tokenizer,
        image_size     = args.image_size,
        batch_size     = args.batch_size,
        num_workers    = args.num_workers,
    )

    if "test" not in loaders:
        logger.log("No test split found. Check --json_root path.")
        return

    model = HydraFakeNet(
        clip_model_name   = args.clip_model,
        freeze_mode       = args.freeze_mode,
        model_dim         = args.model_dim,
        cls_depth         = args.cls_depth,
        cls_heads         = args.cls_heads,
        cls_queries       = args.cls_queries,
        vocab_size        = tokenizer.vocab_size,
        reasoning_hidden  = args.reasoning_hidden,
        reasoning_layers  = args.reasoning_layers,
    ).to(device)

    load_checkpoint(args.checkpoint, model)
    evaluate_test(model, loaders["test"], device, logger)
    logger.close()


if __name__ == "__main__":
    main()