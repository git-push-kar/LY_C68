"""
eval_by_generator.py
====================
Per-generator breakdown of one test split (default: CD), which `eval.py` cannot
produce because it only logs four split totals -- so the 0.7660 on CD has never
been attributable to a particular generator.

This is a separate harness, not an edit to `eval.py`/`dataset.py`: it reuses
`HydraFakeDataset` and `DeepfakeReasoningModel` exactly as `eval.py` does, keeps
per-sample probabilities, and groups them by the generator directory that the
image path already carries (`.../test/cd/<generator>/img.jpg`).

It also reports the split total, so the numbers can be reconciled against the
archived `eval_best.log` before anything is believed.

Usage:
    python eval_by_generator.py --checkpoint runs/intern_fused_ground/checkpoints/best.pth \
        --split cd --out runs/timing/eval_by_generator_cd.json
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter, OrderedDict

import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score
from transformers import AutoTokenizer

from dataset import HydraFakeDataset
from eval_metrics import hanley_mcneil, wilson
from model import DeepfakeReasoningModel

SPLITS = {"id": "id", "cm": "cm", "cf": "cf", "cd": "cd"}


def setup_logger(log_path):
    logger = logging.getLogger("eval_by_generator")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout),
              logging.FileHandler(log_path, mode="w", encoding="utf-8")):
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def generator_of(path, split):
    """`.../test/<split>/<generator>/file.jpg` -> `<generator>`.

    Falls back to the parent directory name so an unexpected layout still groups
    by something meaningful instead of silently collapsing into one bucket.
    """
    parts = [p for p in path.replace("\\", "/").split("/") if p]
    for i, part in enumerate(parts):
        if part.lower() == "test" and i + 2 < len(parts) and parts[i + 1].lower() == split:
            return parts[i + 2]
    return os.path.basename(os.path.dirname(path)) or "unknown"


def metrics(labels, probs, threshold):
    preds = [1 if p >= threshold else 0 for p in probs]
    n = len(labels)
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    tn = n - tp - fp - fn
    n_pos, n_neg = labels.count(1), labels.count(0)
    correct = tp + tn
    acc = correct / n if n else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if prec + rec else 0.0
    try:
        auc = roc_auc_score(labels, probs) if n_pos and n_neg else float("nan")
    except Exception:
        auc = float("nan")
    real = [p for y, p in zip(labels, probs) if y == 0]
    fake = [p for y, p in zip(labels, probs) if y == 1]
    return {
        "n": n, "n_real": n_neg, "n_fake": n_pos,
        "pred_real": tn + fn, "pred_fake": tp + fp,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": acc, "accuracy_ci": wilson(correct, n),
        "precision": prec, "precision_ci": wilson(tp, tp + fp),
        "recall": rec, "recall_ci": wilson(tp, tp + fn),
        "f1": f1, "f1_sklearn": (f1_score(labels, preds, average="binary",
                                          zero_division=0) if n else 0.0),
        "auc": auc, "auc_ci": hanley_mcneil(auc, n_pos, n_neg),
        "mean_p_fake_real": (sum(real) / len(real)) if real else float("nan"),
        "mean_p_fake_fake": (sum(fake) / len(fake)) if fake else float("nan"),
    }


def load_config(path):
    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--dataset_root", default=None)
    p.add_argument("--json_root", default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--model_path", default="./models/InternVL3-2B")
    p.add_argument("--split", default="cd", choices=list(SPLITS))
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--image_size", type=int, default=448)
    p.add_argument("--lm_seq_len", type=int, default=512)
    p.add_argument("--arch_version", default="v2", choices=["v1", "v2"])
    p.add_argument("--attn_implementation", default="sdpa")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--out", default=None)
    p.add_argument("--log_dir", default="runs/timing")
    return p.parse_args()


@torch.no_grad()
def run_split(model, loader, device, threshold):
    labels_all, probs_all = [], []
    for batch in loader:
        pv = batch["pixel_values"].to(device, non_blocking=True)
        freq = batch.get("freq_features")
        if freq is not None:
            freq = freq.to(device, non_blocking=True)
        ctx = autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.nullcontext()
        with ctx:
            out = model(pv, freq_features=freq)
        probs = F.softmax(out["cls_logits"].float(), dim=-1)[:, 1]
        labels_all.extend(int(x) for x in batch["labels"].tolist())
        probs_all.extend(probs.cpu().tolist())
    return labels_all, probs_all


def main():
    args = parse_args()
    cfg = load_config(args.config)
    args.dataset_root = args.dataset_root or cfg.get("dataset_root")
    args.json_root = args.json_root or cfg.get("json_root")
    args.checkpoint = args.checkpoint or cfg.get("checkpoint")
    if not all((args.dataset_root, args.json_root, args.checkpoint)):
        sys.exit("ERROR: --dataset_root, --json_root and --checkpoint are required")

    os.makedirs(args.log_dir, exist_ok=True)
    ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    log_path = os.path.join(args.log_dir, f"eval_by_generator_{ckpt_name}_{args.split}.log")
    log = setup_logger(log_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Split {args.split.upper()} | checkpoint {args.checkpoint} | device {device}")
    if device.type == "cuda":
        log.info(f"GPU {torch.cuda.get_device_name(0)} | threshold {args.threshold}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = DeepfakeReasoningModel(
        model_path=args.model_path, arch_version=args.arch_version,
        attn_implementation=args.attn_implementation,
        use_gradient_checkpointing=False,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    log.info(f"epoch={ckpt.get('epoch', '?')} stage={ckpt.get('stage', '?')}")
    model.eval()

    json_dir = os.path.join(args.json_root, "test", args.split)
    ds = HydraFakeDataset(
        dataset_root=args.dataset_root, json_dir=json_dir, tokenizer=tokenizer,
        split="test", image_size=args.image_size, max_text_len=args.lm_seq_len,
    )
    groups = [generator_of(s["image_path"], args.split) for s in ds.samples]
    log.info(f"samples {len(ds.samples)} | groups {dict(Counter(groups))}")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    labels, probs = run_split(model, loader, device, args.threshold)

    per_group = OrderedDict()
    for g in OrderedDict.fromkeys(groups):                      # stable order
        idx = [i for i, x in enumerate(groups) if x == g]
        per_group[g] = metrics([labels[i] for i in idx], [probs[i] for i in idx],
                               args.threshold)

    total = metrics(labels, probs, args.threshold)

    sep = "=" * 108
    hdr = (f"{'Generator':<22} {'N':>6} {'GT R/F':>10} {'Pred R/F':>11} "
           f"{'Acc [95% CI]':>22} {'P':>6} {'R':>6} {'F1':>7} {'AUC [95% CI]':>20}")
    log.info("\n" + sep)
    log.info(f"Per-generator breakdown -- split {args.split.upper()} "
             f"(positive = FAKE, threshold {args.threshold})")
    log.info(sep)
    log.info(hdr)
    log.info("-" * len(hdr))
    for g, r in per_group.items():
        log.info(
            f"{g:<22} {r['n']:>6} {r['n_real']:>4}/{r['n_fake']:<5} "
            f"{r['pred_real']:>4}/{r['pred_fake']:<6} "
            f"{r['accuracy']:>8.4f} [{r['accuracy_ci'][0]:.4f},{r['accuracy_ci'][1]:.4f}] "
            f"{r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>7.4f} "
            f"{r['auc']:>7.4f} [{r['auc_ci'][0]:.4f},{r['auc_ci'][1]:.4f}]")
    log.info("-" * len(hdr))
    log.info(
        f"{'SPLIT TOTAL':<22} {total['n']:>6} {total['n_real']:>4}/{total['n_fake']:<5} "
        f"{total['pred_real']:>4}/{total['pred_fake']:<6} "
        f"{total['accuracy']:>8.4f} [{total['accuracy_ci'][0]:.4f},{total['accuracy_ci'][1]:.4f}] "
        f"{total['precision']:>6.3f} {total['recall']:>6.3f} {total['f1']:>7.4f} "
        f"{total['auc']:>7.4f} [{total['auc_ci'][0]:.4f},{total['auc_ci'][1]:.4f}]")
    log.info(sep)
    log.info(f"{'Generator':<22} {'mean p_fake | REAL':>20} {'mean p_fake | FAKE':>20}")
    for g, r in per_group.items():
        log.info(f"{g:<22} {r['mean_p_fake_real']:>20.4f} {r['mean_p_fake_fake']:>20.4f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        payload = {
            "split": args.split, "checkpoint": args.checkpoint,
            "threshold": args.threshold, "n": len(ds.samples),
            "epoch": ckpt.get("epoch"), "stage": ckpt.get("stage"),
            "total": total, "per_generator": per_group,
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        log.info(f"JSON written to {args.out}")
    log.info(f"log saved: {log_path}")


if __name__ == "__main__":
    main()
