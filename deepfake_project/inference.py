"""
inference.py
============
Run classification + structured reasoning generation on custom images
using DeepfakeReasoningModel v2 (InternVL3-2B).

Architecture v2 Alignment:
  - Fuses ViT features + 2D FFT frequency representation
  - Utilizes native multi-token visual representation for LLM reasoning
  - Deterministic greedy generation with structured forensic tag parsing

Usage:
  python inference.py --model_path "./models/InternVL3-2B" --checkpoint "./runs/intern_v2/checkpoints/best.pth" --images test_images/1.jpg test_images/2.jpg
"""

import argparse
import logging
import os
import re
import sys
from datetime import datetime
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.amp import autocast
from PIL import Image
from transformers import AutoTokenizer
import torchvision.transforms as T

from dataset import compute_fft_features
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


def _load_config(path="config.yaml"):
    cfg = {}
    if path and os.path.isfile(path):
        try:
            import yaml
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                for k, v in data.items():
                    if not isinstance(v, dict) and v is not None and not str(k).startswith("_"):
                        cfg[k] = v
                if "inference_images" in data and data["inference_images"]:
                    cfg["inference_images"] = data["inference_images"]
        except ImportError:
            pass
        except Exception:
            pass
    return cfg


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Inference for DeepfakeReasoningModel v2")
    p.add_argument("--config",         default="config.yaml")
    p.add_argument("--model_path",     default="./models/InternVL3-2B")
    p.add_argument("--checkpoint",     required=False, default=None)
    p.add_argument("--images",         nargs="+", default=None)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--image_size",     type=int, default=448)
    p.add_argument("--arch_version",   default="v2", choices=["v1", "v2"])
    p.add_argument("--attn_implementation", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    p.add_argument("--log_dir",        default=None)
    return p.parse_args()


def build_transform(image_size: int):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


# ── Reasoning Parser ──────────────────────────────────────────────────────────

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
    text = text.replace("<|im_end|>", "").replace("<|endoftext|>", "").replace("<|im_start|>", "")
    end = text.rfind("</answer>")
    if end != -1:
        return text[:end + len("</answer>")].strip()
    return text.strip()


def format_result(img_path: str, cls_label: int, cls_conf: float,
                  raw_text: str, tags: dict) -> str:
    verdict = "FAKE" if cls_label == 1 else "REAL"
    sep = "=" * 64
    lines = [
        f"\n{sep}",
        f"  Image             : {os.path.basename(img_path)}",
        f"  Classifier verdict: {verdict}  (conf {cls_conf:.1%})",
        sep,
    ]

    order = ["fast", "planning", "reasoning", "reflection", "conclusion"]
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

    gen_answer = tags.get("answer", "unknown")
    lines.append(f"\n  Generated answer : {gen_answer.upper()}")
    if gen_answer in ("real", "fake"):
        gen_verdict = "FAKE" if gen_answer == "fake" else "REAL"
        if gen_verdict != verdict:
            lines.append(f"  [WARNING] Reasoning answer ({gen_verdict}) disagrees with classifier ({verdict})")

    lines.append(sep)
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    _cfg = _load_config(args.config)

    if not args.checkpoint and "checkpoint" in _cfg:
        args.checkpoint = _cfg["checkpoint"]
    if not args.checkpoint:
        sys.exit("ERROR: --checkpoint required (via CLI or config.yaml)")

    if not args.images:
        if "inference_images" in _cfg and _cfg["inference_images"]:
            args.images = _cfg["inference_images"]
        else:
            args.images = ["test_images/1.jpg", "test_images/2.jpg",
                           "test_images/3.jpg", "test_images/4.jpg"]

    for _k, _v in _cfg.items():
        if not hasattr(args, _k):
            continue
        if f"--{_k}" not in sys.argv and f"--{_k.replace('_','-')}" not in sys.argv:
            setattr(args, _k, _v)

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    transform  = build_transform(args.image_size)

    ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.log_dir is None:
        args.log_dir = os.path.join(script_dir, "runs", "inference_logs")
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, f"inference_{ckpt_name}_{ts}.log")
    logger = setup_logger(log_path)

    logger.info(f"Inference log : {log_path}")
    logger.info(f"Checkpoint    : {args.checkpoint}")
    logger.info(f"Images        : {args.images}")
    logger.info(f"Device        : {device}")
    if device.type == "cuda":
        logger.info(f"GPU           : {torch.cuda.get_device_name(0)}")
    logger.info("")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model
    logger.info("Loading model ...")
    model = DeepfakeReasoningModel(
        model_path=args.model_path,
        arch_version=args.arch_version,
        attn_implementation=getattr(args, "attn_implementation", "sdpa"),
        use_gradient_checkpointing=False,
    ).to(device)

    logger.info(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    epoch = ckpt.get("epoch", "?")
    stage = ckpt.get("stage", "?")
    logger.info(f"  epoch={epoch}  stage={stage}  missing={len(missing)}  unexpected={len(unexpected)}")
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
        freq = compute_fft_features(img, target_size=224).unsqueeze(0).to(device)

        with torch.no_grad():
            autocast_ctx = autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.nullcontext()
            with autocast_ctx:
                out = model(pv, freq_features=freq)

            logits   = out["cls_logits"].float()
            probs    = F.softmax(logits, dim=-1)
            cls_pred = logits.argmax(-1).item()
            cls_conf = probs[0, cls_pred].item()

            result = model.predict(
                pv, tokenizer, freq_features=freq,
                max_new_tokens=args.max_new_tokens
            )

        raw_text = trim_after_answer(result["reasoning"][0])
        tags     = parse_reasoning(raw_text)
        verdict  = "FAKE" if cls_pred == 1 else "REAL"

        logger.info(format_result(img_path, cls_pred, cls_conf, raw_text, tags))

        summary.append({
            "image":   os.path.basename(img_path),
            "verdict": verdict,
            "conf":    cls_conf,
            "gen_ans": tags.get("answer", "unknown").upper(),
        })

    logger.info("\n" + "=" * 64)
    logger.info("SUMMARY")
    logger.info("=" * 64)
    logger.info(f"  {'Image':<18} {'Classifier':<10} {'Conf':<8} {'Gen Answer'}")
    logger.info(f"  {'-'*18} {'-'*10} {'-'*8} {'-'*12}")
    for r in summary:
        logger.info(f"  {r['image']:<18} {r['verdict']:<10} {r['conf']:<8.1%} {r['gen_ans']}")
    logger.info("=" * 64)
    logger.info(f"\nLog saved to: {log_path}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
