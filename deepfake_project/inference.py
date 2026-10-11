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
from model import (
    DeepfakeReasoningModel,
    build_grounding_prompt,
    describe_frequency,
    regions_from_topk,
)


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
    p.add_argument("--grounded_reasoning", action="store_true", default=True,
                   help="Condition LLM on classifier telemetry (verdict, attention hotspots, FFT)")
    p.add_argument("--no_grounded_reasoning", action="store_false", dest="grounded_reasoning",
                   help="Disable telemetry conditioning (pure visual reasoning)")
    p.add_argument("--temperature",    type=float, default=1.0)
    p.add_argument("--top_p",          type=float, default=1.0)
    p.add_argument("--repetition_penalty", type=float, default=1.05)
    p.add_argument("--do_sample",      action="store_true", default=False,
                   help="Enable stochastic sampling (default: deterministic greedy, "
                        "which is what the model was validated with)")
    p.add_argument("--raw",            action="store_true",
                   help="Run the base pretrained model without any fine-tuned checkpoint "
                        "(randomly-initialized v2 heads). Skips --checkpoint.")
    p.add_argument("--force_verdict",    default=None, choices=["REAL", "FAKE"],
                   help="Override the classifier verdict word in the telemetry prefix "
                        "(sycophancy audit: feed a deliberately wrong verdict and check "
                        "whether the reasoning parrots it or weighs the evidence). "
                        "Classifier logits/confidence are still reported unmodified.")
    p.add_argument("--verdict_source",   default="classifier", choices=["classifier", "both"],
                   help="classifier (default): classifier verdict is the only decision shown; "
                        "LLM <answer> is hidden and mismatches route to a review queue. "
                        "both: legacy display with Generated answer + WARNING (debugging).")
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
# Fallback: model sometimes emits a boolean instead of real/fake.
# Interpreted as "is the image real?" -> True=real, False=fake.
_ANSWER_BOOL_RE = re.compile(
    r"<answer>\s*(true|false)\s*</answer>", re.IGNORECASE
)

def parse_reasoning(text: str) -> dict:
    tags = {m.group(1).lower(): m.group(2).strip()
            for m in _TAG_RE.finditer(text)}
    m = _ANSWER_RE.search(text)
    if m:
        tags["answer"] = m.group(1).lower()
        tags["answer_heuristic"] = False
    else:
        mb = _ANSWER_BOOL_RE.search(text)
        if mb:
            tags["answer"] = "real" if mb.group(1).lower() == "true" else "fake"
            tags["answer_heuristic"] = True
        else:
            tags["answer"] = "unknown"
            tags["answer_heuristic"] = False
    return tags


def trim_after_answer(text: str) -> str:
    text = text.replace("<|im_end|>", "").replace("<|endoftext|>", "").replace("<|im_start|>", "")
    end = text.rfind("</answer>")
    if end != -1:
        return text[:end + len("</answer>")].strip()
    return text.strip()


def extract_forensic_telemetry(
    cls_pred: int,
    cls_conf: float,
    freq_features: Optional[torch.Tensor] = None,
    attn_weights: Optional[torch.Tensor] = None,
    grid_size: int = 32,
) -> dict:
    """
    Extracts structured forensic telemetry from classifier confidence,
    FFT frequency spectrum, and spatial attention pooling weights.
    """
    verdict = "FAKE" if cls_pred == 1 else "REAL"

    # 1. Spatial Anomaly Localization from Attention Weights.
    # Shared bucketing with model.regions_from_topk (also used to build
    # training-time telemetry) so train/inference region names can never drift.
    regions = "Uniform facial distribution"
    if attn_weights is not None:
        flat_weights = attn_weights.squeeze().float()
        topk = torch.topk(flat_weights, k=min(8, flat_weights.numel())).indices.cpu().tolist()
        regions = regions_from_topk(topk)

    # 2. Spectral (FFT) Anomaly Analysis — shared neutral measurement
    # (model.describe_frequency). Deliberately verdict-neutral: the verdict
    # lives on its own line; the spectrum is independent evidence.
    spectral_desc = "Natural spectral distribution"
    if freq_features is not None:
        ff = freq_features.squeeze()
        spectral_desc = describe_frequency(ff.unsqueeze(0) if ff.dim() == 2 else ff)

    regions_str = regions
    return {
        "verdict": verdict,
        "conf": cls_conf,
        "regions": regions_str,
        "spectral_desc": spectral_desc,
    }


def build_telemetry_prompt(telemetry: dict) -> str:
    """Builds structured forensic context passed to the LLM.

    Thin wrapper over model.build_grounding_prompt — the single canonical
    template also used for training-time telemetry. See model.py.
    """
    c = telemetry["conf"]
    return build_grounding_prompt(
        verdict=telemetry["verdict"],
        conf_str=f"confidence {c:.1%}",
        spectral_desc=telemetry["spectral_desc"],
        regions_str=telemetry["regions"],
    )


def format_result(img_path: str, cls_label: int, cls_conf: float,
                   raw_text: str, tags: dict, telemetry: Optional[dict] = None,
                   verdict_source: str = "both") -> str:
    """Render one image block.

    verdict_source="classifier" (production rule): the classifier verdict is
    the only decision shown — the LLM's <answer> is never displayed. A
    mismatch (or unparsable answer) is routed to the human review queue
    instead of a WARNING. "both" keeps the legacy display (gen answer +
    WARNING) for debugging.
    """
    verdict = "FAKE" if cls_label == 1 else "REAL"
    gen_answer = tags.get("answer", "unknown")
    mismatch = gen_answer in ("real", "fake") and (
        ("FAKE" if gen_answer == "fake" else "REAL") != verdict)
    unparsed = gen_answer == "unknown"
    needs_review = mismatch or unparsed

    sep = "=" * 64
    lines = [
        f"\n{sep}",
        f"  Image             : {os.path.basename(img_path)}",
        f"  Classifier verdict: {verdict}  (conf {cls_conf:.1%})",
    ]
    if telemetry:
        lines.append(f"  Anomaly Hotspots  : {telemetry.get('regions', 'N/A')}")
        lines.append(f"  Frequency Domain  : {telemetry.get('spectral_desc', 'N/A')}")
    lines.append(sep)

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
    if verdict_source == "classifier":
        # Production rule: LLM classification is never shown and never decides.
        if mismatch:
            lines.append(f"\n  [REVIEW] Reasoning disagrees with classifier verdict — queued for human review")
        elif unparsed:
            lines.append(f"\n  [REVIEW] Reasoning answer unparsable — queued for human review")
    else:
        lines.append(f"\n  Generated answer : {gen_answer.upper()}")
        if tags.get("answer_heuristic"):
            lines.append(f"  [NOTE] Answer tag was boolean (true/false); mapped heuristically")
        if mismatch:
            gen_verdict = "FAKE" if gen_answer == "fake" else "REAL"
            lines.append(f"  [WARNING] Reasoning answer ({gen_verdict}) disagrees with classifier ({verdict})")

    lines.append(sep)
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    _cfg = _load_config(args.config)

    if not args.checkpoint and "checkpoint" in _cfg and not args.raw:
        args.checkpoint = _cfg["checkpoint"]
    if not args.checkpoint and not args.raw:
        sys.exit("ERROR: --checkpoint required (via CLI or config.yaml), or pass --raw for the base model")

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

    ckpt_name = "raw" if args.raw else os.path.splitext(os.path.basename(args.checkpoint))[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.log_dir is None:
        args.log_dir = os.path.join(script_dir, "runs", "inference_logs")
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, f"inference_{ckpt_name}_{ts}.log")
    logger = setup_logger(log_path)

    logger.info(f"Inference log : {log_path}")
    logger.info(f"Checkpoint    : {'(none — raw base model)' if args.raw else args.checkpoint}")
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

    if args.raw:
        logger.info("Raw base model: no checkpoint loaded "
                    "(pretrained backbone + randomly-initialized v2 heads; "
                    "classifier verdict is untrained, reasoning comes from the base LLM)")
    else:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state_dict = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        epoch = ckpt.get("epoch", "?")
        stage = ckpt.get("stage", "?")
        logger.info(f"  epoch={epoch}  stage={stage}  missing={len(missing)}  unexpected={len(unexpected)}")
    model.eval()

    summary = []
    review_queue = []
    verdict_source = getattr(args, "verdict_source", "both")
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

            telemetry = None
            prompt_text = None
            if getattr(args, "grounded_reasoning", True):
                # Extract attention pooling spatial weights
                _, _, _, attn_weights = model._get_vision_features(pv, return_attn_weights=True)
                telemetry = extract_forensic_telemetry(
                    cls_pred=cls_pred,
                    cls_conf=cls_conf,
                    freq_features=freq,
                    attn_weights=attn_weights,
                )
                if getattr(args, "force_verdict", None):
                    telemetry = dict(telemetry, verdict=args.force_verdict)
                    logger.info(f"  [AUDIT] telemetry verdict overridden to {args.force_verdict} "
                                f"(classifier still says {'FAKE' if cls_pred == 1 else 'REAL'})")
                prompt_text = build_telemetry_prompt(telemetry)

            result = model.predict(
                pv, tokenizer, freq_features=freq,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                prompt_text=prompt_text,
            )

        raw_text = trim_after_answer(result["reasoning"][0])
        tags     = parse_reasoning(raw_text)
        verdict  = "FAKE" if cls_pred == 1 else "REAL"

        logger.info(format_result(img_path, cls_pred, cls_conf, raw_text, tags,
                                  telemetry=telemetry, verdict_source=verdict_source))

        gen_ans = tags.get("answer", "unknown").upper()
        summary.append({
            "image":   os.path.basename(img_path),
            "verdict": verdict,
            "conf":    cls_conf,
            "gen_ans": gen_ans,
        })
        if verdict_source == "classifier":
            if gen_ans == "UNKNOWN":
                review_queue.append((os.path.basename(img_path), verdict, "unparsable answer"))
            elif gen_ans != verdict:
                review_queue.append((os.path.basename(img_path), verdict,
                                     f"reasoning says {gen_ans}"))

    logger.info("\n" + "=" * 64)
    logger.info("SUMMARY")
    logger.info("=" * 64)
    if verdict_source == "classifier":
        logger.info(f"  {'Image':<18} {'Verdict':<10} {'Conf'}")
        logger.info(f"  {'-'*18} {'-'*10} {'-'*8}")
        for r in summary:
            logger.info(f"  {r['image']:<18} {r['verdict']:<10} {r['conf']:<8.1%}")
    else:
        logger.info(f"  {'Image':<18} {'Classifier':<10} {'Conf':<8} {'Gen Answer'}")
        logger.info(f"  {'-'*18} {'-'*10} {'-'*8} {'-'*12}")
        for r in summary:
            logger.info(f"  {r['image']:<18} {r['verdict']:<10} {r['conf']:<8.1%} {r['gen_ans']}")
    logger.info("=" * 64)
    if verdict_source == "classifier":
        logger.info(f"\nREVIEW QUEUE ({len(review_queue)}): mismatches + unparsable answers for human review")
        for img, ver, reason in review_queue:
            logger.info(f"  {img:<18} classifier={ver:<5} reason: {reason}")
        if not review_queue:
            logger.info("  (empty — all reasoning agreed and parsed)")
        logger.info("=" * 64)
    logger.info(f"\nLog saved to: {log_path}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
