"""
evaluate.py
===========
Evaluate a trained HydraFakeNet checkpoint on the test split(s).

Usage:
    python evaluate.py \
        --checkpoint ./runs/exp1/checkpoints/best_model.pth \
        --dataset_root /path/to/hydrafake \
        --json_root    /path/to/hydrafake/jsons \
        --output_dir   ./runs/exp1/eval
"""

import argparse
import os

import torch
from torch.cuda.amp import autocast

from dataset import build_dataloaders
from model import HydraFakeNet
from utils import Logger, compute_metrics, load_checkpoint, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="HydraFake Evaluation")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--json_root", required=True)
    p.add_argument("--output_dir", default="./eval_results")
    p.add_argument("--tokenizer", default="bert-base-uncased")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--embed_dim", type=int, default=512)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_text_len", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def evaluate_split(model, loader, device, split_name: str, logger: Logger):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with autocast():
            outputs = model(images)
            cls_logits = outputs["cls_logits"]

        probs = torch.softmax(cls_logits, dim=-1)[:, 1].cpu().tolist()
        preds = cls_logits.argmax(dim=-1).cpu().tolist()

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

    metrics = compute_metrics(all_labels, all_preds, all_probs)

    logger.log(f"\n{'='*60}")
    logger.log(f"Results for split: {split_name}")
    logger.log(f"  Accuracy : {metrics['accuracy']:.4f}")
    logger.log(f"  F1       : {metrics['f1']:.4f}")
    if metrics.get("auc") is not None:
        logger.log(f"  AUC-ROC  : {metrics['auc']:.4f}")
    logger.log(f"\n{metrics['report']}")

    return metrics


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    logger = Logger(args.output_dir, filename="eval.log")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log(f"Device: {device}")

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

    model = HydraFakeNet(
        image_size=args.image_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        vocab_size=vocab_size,
        reasoning_hidden_dim=args.embed_dim,
        max_reasoning_len=args.max_text_len,
    ).to(device)

    load_checkpoint(args.checkpoint, model)

    all_metrics = {}
    for split in ("val", "test"):
        if split in loaders:
            metrics = evaluate_split(model, loaders[split], device, split, logger)
            all_metrics[split] = metrics

    logger.save_metrics(all_metrics, "eval_metrics.json")
    logger.close()


if __name__ == "__main__":
    main()
