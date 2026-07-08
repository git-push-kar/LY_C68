
# eval.py
# =======
# Evaluate a checkpoint across all 4 HydraFake test splits: ID, CM, CF, CD.
# Works for both cls-only (ep001–ep003) and joint checkpoints.
# No reasoning generation — cls_head only, fast.

# Logging: all output written to stdout AND <log_dir>/eval_<checkpoint>.log

# Usage:
#     python eval.py ^
#         --dataset_root "C:\Users\Admin\Desktop\ly project c 68\deepfake_project\datasets\hydrafake" ^
#         --json_root    "C:\Users\Admin\Desktop\ly project c 68\deepfake_project\datasets\hydrafake\jsons" ^
#         --model_path   "./models/InternVL3-2B" ^
#         --checkpoint   "./runs/intern_exp2/checkpoints/ep001.pth" ^
#         --batch_size   8 ^
#         --num_workers  8

import argparse
import logging
import os
import sys

import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from dataset import HydraFakeDataset
from model import DeepfakeReasoningModel


# ── Logger ────────────────────────────────────────────────────────────────────

def setup_logger(log_path: str) -> logging.Logger:
    """Write to stdout and log file simultaneously."""
    logger = logging.getLogger("eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--json_root",    required=True)
    p.add_argument("--model_path",   default="./models/InternVL3-2B")
    p.add_argument("--checkpoint",   required=True)
    p.add_argument("--batch_size",   type=int, default=8)
    p.add_argument("--num_workers",  type=int, default=8)
    p.add_argument("--image_size",   type=int, default=448)
    p.add_argument("--lm_seq_len",   type=int, default=192)
    p.add_argument("--log_dir",      default=None,
                   help="Where to write eval log. Defaults to "
                        "<checkpoint_dir>/../logs/")
    return p.parse_args()


# ── Eval loop ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_split(model, loader, device):
    model.eval()
    preds_all, labels_all, probs_all = [], [], []

    for batch in loader:
        pv     = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"]

        with autocast("cuda", dtype=torch.bfloat16):
            out = model(pv)   # cls-only forward, no labels needed

        logits = out["cls_logits"].float()
        probs  = F.softmax(logits, dim=-1)[:, 1]
        preds  = logits.argmax(dim=-1)

        preds_all.extend(preds.cpu().tolist())
        labels_all.extend(labels.tolist())
        probs_all.extend(probs.cpu().tolist())

    acc = accuracy_score(labels_all, preds_all)
    f1  = f1_score(labels_all, preds_all, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(labels_all, probs_all)
    except Exception:
        auc = 0.0

    n_real = labels_all.count(0); n_fake = labels_all.count(1)
    p_real = preds_all.count(0);  p_fake = preds_all.count(1)

    return {
        "n": len(labels_all), "n_real": n_real, "n_fake": n_fake,
        "p_real": p_real, "p_fake": p_fake,
        "accuracy": acc, "f1": f1, "auc": auc,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Log path: sits in <run>/logs/eval_<ckptname>.log by default ───
    ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    if args.log_dir:
        log_dir = args.log_dir
    else:
        log_dir = os.path.join(
            os.path.dirname(os.path.dirname(args.checkpoint)), "logs"
        )
    log_path = os.path.join(log_dir, f"eval_{ckpt_name}.log")
    logger   = setup_logger(log_path)

    logger.info(f"Eval log : {log_path}")
    logger.info(f"Device   : {device}")
    if device.type == "cuda":
        logger.info(f"GPU      : {torch.cuda.get_device_name(0)}")

    # ── Tokenizer ─────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model ─────────────────────────────────────────────────────────
    logger.info("Loading model ...")
    model = DeepfakeReasoningModel(model_path=args.model_path).to(device)

    logger.info(f"Loading checkpoint: {args.checkpoint}")
    ckpt      = torch.load(args.checkpoint, map_location=device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    epoch_num = ckpt.get("epoch", "?")
    stage     = ckpt.get("stage", "cls_only")
    logger.info(f"  epoch={epoch_num}  stage={stage}  "
                f"missing={len(missing)}  unexpected={len(unexpected)}")
    if missing:
        logger.info(f"  Missing   : {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        logger.info(f"  Unexpected: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    model.eval()

    # ── Test splits ───────────────────────────────────────────────────
    splits = {
        "ID  (In-Domain)":     "id",
        "CM  (Cross-Model)":   "cm",
        "CF  (Cross-Forgery)": "cf",
        "CD  (Cross-Domain)":  "cd",
    }

    sep = "=" * 74
    hdr = (f"{'Split':<24} {'N':>6} {'GT R/F':>9} {'Pred R/F':>10} "
           f"{'Acc':>7} {'F1':>7} {'AUC':>7}")
    logger.info(f"\nCheckpoint: {os.path.basename(args.checkpoint)}  "
                f"(epoch {epoch_num}, {stage})")
    logger.info(sep)
    logger.info(hdr)
    logger.info(sep)

    all_results = {}
    for name, subdir in splits.items():
        json_dir = os.path.join(args.json_root, "test", subdir)
        if not os.path.isdir(json_dir):
            json_dir = os.path.join(args.json_root, "test")
        if not os.path.isdir(json_dir):
            logger.info(f"{name:<24}  SKIPPED (dir not found)")
            continue

        try:
            ds = HydraFakeDataset(
                dataset_root=args.dataset_root,
                json_dir=json_dir,
                tokenizer=tokenizer,
                split="test",
                image_size=args.image_size,
                max_text_len=args.lm_seq_len,
            )
            if not ds.samples:
                logger.info(f"{name:<24}  NO DATA")
                continue

            loader = DataLoader(
                ds, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True,
            )
            r = eval_split(model, loader, device)
            all_results[name] = r

            flag = "  !! COLLAPSE" if r["p_real"] == 0 or r["p_fake"] == 0 else ""
            logger.info(
                f"{name:<24} {r['n']:>6} "
                f"{r['n_real']:>4}/{r['n_fake']:<4} "
                f"{r['p_real']:>4}/{r['p_fake']:<5} "
                f"{r['accuracy']:>7.4f} {r['f1']:>7.4f} {r['auc']:>7.4f}"
                f"{flag}"
            )
        except Exception as e:
            logger.info(f"{name:<24}  ERROR: {e}")

    if all_results:
        logger.info(sep)
        avg_acc = sum(r["accuracy"] for r in all_results.values()) / len(all_results)
        avg_f1  = sum(r["f1"]       for r in all_results.values()) / len(all_results)
        avg_auc = sum(r["auc"]      for r in all_results.values()) / len(all_results)
        logger.info(
            f"{'MEAN':<24} {'':>6} {'':>9} {'':>10} "
            f"{avg_acc:>7.4f} {avg_f1:>7.4f} {avg_auc:>7.4f}"
        )
        logger.info(sep)
        logger.info(f"Log saved: {log_path}")


if __name__ == "__main__":
    main()
