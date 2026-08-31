"""
inference.py
============
Run classification + reasoning generation on custom images.
Logs everything to the run's logs directory.

Aligned with corrective training (keep-richest, verbatim tagged
targets + <answer> + EOS, seq 512, deterministic decoding):

  Training (corrective):  --weights_from ep010 --freeze_cls_head
                          --freeze_projector --lm_seq_len 512
  Inference: greedy (do_sample=False) + use_cache + trim after
             </answer>; max_new_tokens should match lm_seq_len.

Usage:
    python inference.py ^
        --model_path  "./models/InternVL3-2B" ^
        --checkpoint  "./runs/intern_exp3_corrective/checkpoints/best.pth" ^
        --images      1.jpg 2.jpg 3.jpg 4.jpg
    # or:  --checkpoint ./runs/intern_exp2/checkpoints/ep010.pth
"""

import argparse
import logging
import os
import re
import sys
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.amp import autocast
from PIL import Image
from transformers import AutoTokenizer
import torchvision.transforms as T

from model import DeepfakeReasoningModel


# ── Logger ────────────────────────────────────────────────────────────────────

def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("inference")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(message)s")

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
    p.add_argument("--model_path",     default="./models/InternVL3-2B")
    p.add_argument("--checkpoint",     required=True)
    p.add_argument("--images",         nargs="+",
                   default=["1.jpg", "2.jpg", "3.jpg", "4.jpg"])
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--image_size",     type=int, default=448)
    p.add_argument("--log_dir",        default=None,
                   help="Defaults to <project>/runs/intern_exp2/logs")
    return p.parse_args()


# ── Image preprocessing ───────────────────────────────────────────────────────

def build_transform(image_size: int):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


# ── Reasoning parser ──────────────────────────────────────────────────────────

_TAG_RE = re.compile(
    r"<(fast|planning|reasoning|reflection|conclusion)>(.*?)</\1>",
    re.DOTALL | re.IGNORECASE,
)
_ANSWER_RE = re.compile(
    r"<answer>\s*(real|fake)\s*</answer>", re.IGNORECASE
)

def parse_reasoning(text: str) -> dict:
    tags = {m.group(1).lower(): m.group(2).strip()
            for m in _TAG_RE.finditer(text)}
    m = _ANSWER_RE.search(text)
    tags["answer"] = m.group(1).lower() if m else "unknown"
    return tags


def trim_after_answer(text: str) -> str:
    """
    Cut everything after the LAST '</answer>' tag. Checkpoints trained
    before the eos-append fix never learned when to stop and ramble until
    max_new_tokens is exhausted. New checkpoints emit <eos> right after
    </answer>, so this is a no-op for them but keeps output clean for
    older checkpoints. Also strips <|im_end|>/<|endoftext|> artifacts.
    """
    # strip leftover special-token strings that survive skip_special_tokens
    text = text.replace("<|im_end|>", "").replace("<|endoftext|>", "").replace("<|im_start|>", "")
    end = text.rfind("</answer>")
    if end != -1:
        return text[:end + len("</answer>")].strip()
    return text.strip()


# ── Result formatter ──────────────────────────────────────────────────────────

def format_result(img_path: str, cls_label: int, cls_conf: float,
                  raw_text: str, tags: dict) -> str:
    verdict = "FAKE" if cls_label == 1 else "REAL"
    sep = "=" * 60
    lines = [
        f"\n{sep}",
        f"  Image             : {os.path.basename(img_path)}",
        f"  Classifier verdict: {verdict}  (conf {cls_conf:.1%})",
        sep,
    ]

    order  = ["fast", "planning", "reasoning", "reflection", "conclusion"]
    labels = {
        "fast":       "Quick take",
        "planning":   "Planning",
        "reasoning":  "Reasoning",
        "reflection": "Reflection",
        "conclusion": "Conclusion",
    }
    has_tags = any(k in tags and tags[k] for k in order)

    if has_tags:
        for key in order:
            if key in tags and tags[key]:
                lines.append(f"\n  [{labels[key]}]")
                for line in tags[key].split("\n"):
                    line = line.strip()
                    if line:
                        lines.append(f"    {line}")
    else:
        lines.append("\n  [Raw LM output]")
        for line in raw_text.strip().split("\n"):
            line = line.strip()
            if line:
                lines.append(f"    {line}")

    # Surface the generated <answer> and flag classifier/reasoning drift
    gen_answer = tags.get("answer", "unknown")
    lines.append(f"\n  Generated answer : {gen_answer.upper()}")
    if gen_answer in ("real", "fake"):
        gen_verdict = "FAKE" if gen_answer == "fake" else "REAL"
        if gen_verdict != verdict:
            lines.append(f"  [WARNING] reasoning answer ({gen_verdict}) disagrees with classifier ({verdict})")

    lines.append(sep)
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args       = parse_args()
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    transform  = build_transform(args.image_size)

    # ── Log file: inference_<ckpt>_<timestamp>.log ────────────────────
    ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.log_dir is None:
        args.log_dir = os.path.join(
            script_dir, "runs", "intern_exp2", "logs")
    os.makedirs(args.log_dir, exist_ok=True)
    log_path  = os.path.join(args.log_dir, f"inference_{ckpt_name}_{ts}.log")
    logger    = setup_logger(log_path)

    logger.info(f"Inference log : {log_path}")
    logger.info(f"Checkpoint    : {args.checkpoint}")
    logger.info(f"Images        : {args.images}")
    logger.info(f"Device        : {device}")
    if device.type == "cuda":
        logger.info(f"GPU           : {torch.cuda.get_device_name(0)}")
    logger.info("")

    # ── Tokenizer ─────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model ─────────────────────────────────────────────────────────
    logger.info("Loading model ...")
    model = DeepfakeReasoningModel(model_path=args.model_path).to(device)

    logger.info(f"Loading checkpoint: {args.checkpoint}")
    ckpt    = torch.load(args.checkpoint, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    epoch   = ckpt.get("epoch", "?")
    stage   = ckpt.get("stage", "?")
    logger.info(f"  epoch={epoch}  stage={stage}  "
                f"missing={len(missing)}  unexpected={len(unexpected)}")
    model.eval()

    summary = []

    logger.info(f"\nProcessing {len(args.images)} image(s) ...\n")

    for img_name in args.images:
        img_path = os.path.join(script_dir, img_name)
        if not os.path.exists(img_path):
            img_path = img_name
        if not os.path.exists(img_path):
            logger.info(f"  [{img_name}]  NOT FOUND — skipping")
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.info(f"  [{img_name}]  Could not open: {e} — skipping")
            continue

        pv = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            with autocast("cuda", dtype=torch.bfloat16):
                out = model(pv)

            logits   = out["cls_logits"].float()
            probs    = F.softmax(logits, dim=-1)
            cls_pred = logits.argmax(-1).item()
            cls_conf = probs[0, cls_pred].item()

            result = model.predict(pv, tokenizer,
                                   max_new_tokens=args.max_new_tokens)

        raw_text = trim_after_answer(result["reasoning"][0])
        tags     = parse_reasoning(raw_text)
        verdict  = "FAKE" if cls_pred == 1 else "REAL"

        logger.info(format_result(img_path, cls_pred, cls_conf, raw_text, tags))

        summary.append({
            "image":   os.path.basename(img_path),
            "verdict": verdict,
            "conf":    cls_conf,
        })

    # ── Summary table ─────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  {'Image':<15} {'Verdict':<8} {'Confidence'}")
    logger.info(f"  {'-'*15} {'-'*8} {'-'*10}")
    for r in summary:
        logger.info(f"  {r['image']:<15} {r['verdict']:<8} {r['conf']:.1%}")
    logger.info("=" * 60)
    logger.info(f"\nLog saved to: {log_path}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
