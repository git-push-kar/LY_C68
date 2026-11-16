"""
inference_faces.py
==================
Multi-face inference: detect -> crop -> batched per-face classification.

Routing (plan section 5.1). The model, its weights and its preprocessing are
unchanged -- detection only decides *which image* the model sees:

  exactly 1 face, and it fills >= --passthrough_min_face_frac of the frame
      -> PASSTHROUGH: the whole image goes through the same preprocessing and
         forward call inference.py uses, so verdicts match it exactly. Checked
         before --min_face_px, because passthrough crops nothing.
  >=2 faces, or 1 face that is small in the frame
      -> PER_FACE: one context crop per face (face_detect.crop_with_context),
         all crops classified in a single batched forward
  0 faces, or every face below --min_face_px
      -> NO_FACE: the whole image is evaluated and flagged faces_detected=0

Reasoning uses the same grounded prompt and tag parser as inference.py. It runs
for the highest-risk face by default (--reason_faces top1), optionally for every
face (all) or not at all (none). The classifier verdict remains the only
decision; the LLM answer is reported for audit and any disagreement sets
flags.model_disagreement and forces human review.

Examples:
    python inference_faces.py --adapter_dir ./runs/intern_fused_ground/adapters \
        --images test_images/multiple-faces.jpg test_images/real/161022.jpg

    # reference behaviour, no detection at all (identical to inference.py)
    python inference_faces.py --checkpoint ./runs/intern_fused_ground/checkpoints/best.pth \
        --no_detect --images test_images/real/161022.jpg
"""

import argparse
import json
import os
import sys
from contextlib import nullcontext
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from PIL import Image

from adapter_loader import AdapterLoadError, load_model
from dataset import compute_fft_features
from face_detect import (
    FaceDetector,
    crop_with_context,
    draw_boxes,
    filter_faces,
)
from inference import (
    build_telemetry_prompt,
    build_transform,
    extract_forensic_telemetry,
    parse_reasoning,
    setup_logger,
    trim_after_answer,
)


# ── Args ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Detect, crop and classify every face in an image")
    p.add_argument("--base_model", default="./models/internvl3_2b_lr_fused")
    p.add_argument("--adapter_dir", default=None,
                   help="adapter package (llm_adapter + vision_adapter + custom_heads)")
    p.add_argument("--checkpoint", default=None,
                   help="whole-model .pth (mutually exclusive with --adapter_dir)")
    p.add_argument("--images", nargs="+", default=None)
    p.add_argument("--arch_version", default="v2", choices=["v1", "v2"])
    p.add_argument("--image_size", type=int, default=448)
    p.add_argument("--attn_implementation", default="sdpa")
    p.add_argument("--device", default=None)
    p.add_argument("--log_dir", default=None)
    p.add_argument("--json_out", default=None)

    g = p.add_argument_group("routing")
    g.add_argument("--no_detect", action="store_true",
                   help="skip detection and evaluate the whole image (inference.py behaviour)")
    g.add_argument("--passthrough_min_face_frac", type=float, default=0.05,
                   help="a lone face must cover this fraction of the frame to keep the "
                        "whole-image path; below it, the crop path is used. Measured on "
                        "test_images: portraits ~0.088 (omi.jpg), faces in group or wide "
                        "scenes 0.021-0.029 (3-people.jpg, chick.jpg)")
    g.add_argument("--aggregate", default="max",
                   choices=["max", "mean", "any_fake", "per_face"],
                   help="document verdict from per-face probabilities (default: max)")

    g2 = p.add_argument_group("detector")
    g2.add_argument("--detector_weights", default="./models/yolov8n-face.pt")
    g2.add_argument("--detector_imgsz", type=int, default=640)
    g2.add_argument("--detector_conf", type=float, default=0.25)
    g2.add_argument("--detector_iou", type=float, default=0.50)
    g2.add_argument("--min_face_px", type=int, default=96)
    g2.add_argument("--max_faces", type=int, default=8)

    g3 = p.add_argument_group("decisions & artifacts")
    g3.add_argument("--margin", type=float, default=0.30,
                    help="background ring added around each detected face")
    g3.add_argument("--face_threshold", type=float, default=0.50,
                    help="p_fake decision boundary. 0.50 keeps per-face verdicts consistent "
                         "with inference.py's argmax until the plan's margin sweep calibrates it")
    g3.add_argument("--save_crops", default=None, help="directory for per-face crops")
    g3.add_argument("--save_annotated", default=None, help="directory for box overlays")

    g4 = p.add_argument_group("reasoning")
    g4.add_argument("--reason_faces", default="top1", choices=["none", "top1", "all"],
                    help="which evaluated faces get LLM reasoning: the highest p_fake only "
                         "(default), every face, or none")
    g4.add_argument("--grounded_reasoning", action="store_true", default=True,
                    help="condition the LLM on classifier telemetry (verdict, hotspots, FFT)")
    g4.add_argument("--no_grounded_reasoning", action="store_false", dest="grounded_reasoning",
                    help="pure visual reasoning, no telemetry prefix")
    g4.add_argument("--max_new_tokens", type=int, default=512)
    g4.add_argument("--do_sample", action="store_true", default=False,
                    help="stochastic sampling (default: deterministic greedy, which is how "
                         "the model was validated)")
    g4.add_argument("--temperature", type=float, default=1.0)
    g4.add_argument("--top_p", type=float, default=1.0)
    g4.add_argument("--repetition_penalty", type=float, default=1.05)
    return p.parse_args()


# ── Routing ──────────────────────────────────────────────────────────────────

def route_image(image: Image.Image, detector: Optional[FaceDetector], args) -> Dict[str, Any]:
    """Decide what the model is shown, without touching the model.

    Returns {"path", "boxes", "kept", "skipped", "items"} where each item is
    {"kind", "image", "meta", "box", "id"}.
    """
    img_w, img_h = image.size
    # Whole-image rows still carry an integer id: faces[].id and
    # flags.reasoned_faces are keyed by id, so a null here leaks into the JSON
    # (reasoned_faces: [null]) for passthrough, no_face and no_detect alike.
    whole = {"kind": "whole", "image": image, "meta": {}, "box": None, "id": 0}

    if detector is None:
        return {"path": "no_detect", "boxes": [], "kept": [], "skipped": [], "items": [whole]}

    boxes = detector.detect(image)

    # Passthrough is decided BEFORE the minimum-crop-size gate. A lone face that
    # fills the frame is handed over as the whole image (exactly what
    # inference.py does), so nothing is cropped and min_face_px never applies.
    # Applying the gate first would reject the 200x200 training-style crops,
    # whose face boxes are ~80 px wide, and mislabel them "no_face".
    if len(boxes) == 1:
        lone = boxes[0].clamped(img_w, img_h)
        if lone.area / float(img_w * img_h) >= args.passthrough_min_face_frac:
            return {"path": "passthrough", "boxes": boxes, "kept": [lone],
                    "skipped": [], "items": [whole]}

    kept, skipped = filter_faces(boxes, (img_w, img_h),
                                 min_face_px=args.min_face_px, max_faces=args.max_faces)
    if not kept:
        return {"path": "no_face", "boxes": boxes, "kept": [], "skipped": skipped,
                "items": [whole]}

    items = []
    for i, box in enumerate(kept):
        crop, meta = crop_with_context(image, box, margin=args.margin,
                                       target_size=args.image_size)
        items.append({"kind": "face", "image": crop, "meta": meta, "box": box, "id": i})
    return {"path": "per_face", "boxes": boxes, "kept": kept, "skipped": skipped,
            "items": items}


# ── Model calls ──────────────────────────────────────────────────────────────

def _autocast_ctx(device: torch.device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()          # torch.nullcontext does not exist


def _to_batch(tensors: List[torch.Tensor]) -> torch.Tensor:
    # unsqueeze for a single item: exactly the tensor inference.py builds
    return tensors[0].unsqueeze(0) if len(tensors) == 1 else torch.stack(tensors)


@torch.no_grad()
def classify(model, images: List[Image.Image], transform, device: torch.device):
    """Batched classifier pass. Returns (probs[N,2], freq[N,3,224,224])."""
    pv = _to_batch([transform(im) for im in images]).to(device)
    freq = _to_batch([compute_fft_features(im, target_size=224)
                      for im in images]).to(device)
    with _autocast_ctx(device):
        out = model(pv, freq_features=freq)
        logits = out["cls_logits"].float()
    probs = F.softmax(logits, dim=-1).cpu()
    return probs, freq.cpu(), pv


@torch.no_grad()
def attention_weights(model, pv: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Spatial attention weights for telemetry (same extra pass inference.py makes)."""
    with _autocast_ctx(device):
        _, _, _, attn = model._get_vision_features(pv.to(device), return_attn_weights=True)
    return attn.cpu()


# ── Aggregation ──────────────────────────────────────────────────────────────

def aggregate_verdict(rows: List[Dict[str, Any]], mode: str,
                      threshold: float) -> Optional[str]:
    if not rows or mode == "per_face":
        return None
    p = [r["p_fake"] for r in rows]
    if mode == "mean":
        return "FAKE" if (sum(p) / len(p)) >= threshold else "REAL"
    return "FAKE" if max(p) >= threshold else "REAL"     # "max" and "any_fake"


def select_reason_targets(rows: List[Dict[str, Any]], mode: str) -> List[int]:
    """Indices of rows to reason about.

    top1 = the highest p_fake (ties broken by position, so the choice is
    deterministic). Generation is one LLM call per item, so a 6-face photo is 6x
    the latency for reasoning the classifier already decided.
    """
    if mode == "none" or not rows:
        return []
    ranked = sorted(range(len(rows)), key=lambda i: (-rows[i]["p_fake"], i))
    return list(ranked) if mode == "all" else ranked[:1]


def find_disagreements(rows: List[Dict[str, Any]], reasoned: List[int]) -> List[int]:
    """Indices where the LLM's answer contradicts the classifier verdict.

    Only reasoned rows can disagree, and only a parseable real/fake answer
    counts: 'unknown' means the generation did not commit to anything, which is
    a parse failure rather than a contradiction.
    """
    out = []
    for i in reasoned:
        answer = (rows[i].get("reasoning") or {}).get("generated_answer", "unknown")
        if answer in ("real", "fake") and answer.upper() != rows[i]["verdict"]:
            out.append(i)
    return out


REASON_FIELDS = (("quick_take", "Quick take"), ("planning", "Planning"),
                 ("analysis", "Reasoning"), ("reflection", "Reflection"),
                 ("conclusion", "Conclusion"))


def _log_reasoning(logger, label: str, reason: Dict[str, Any], verdict: str) -> None:
    logger.info(f"              -- reasoning for {label} --")
    for key, title in REASON_FIELDS:
        text = (reason.get(key) or "").strip()
        if not text:
            continue
        logger.info(f"              [{title}]")
        for line in text.splitlines():
            if line.strip():
                logger.info(f"                {line.strip()}")
    answer = reason.get("generated_answer", "unknown")
    mark = "" if answer.upper() == verdict else f"  (classifier says {verdict})"
    logger.info(f"              LLM answer: {answer}{mark}")


@torch.no_grad()
def reason_about_item(model, tokenizer, item_pv: torch.Tensor, item_freq: torch.Tensor,
                      telemetry: Dict[str, Any], args, device) -> Dict[str, Any]:
    """One grounded generation for one already-classified item.

    Mirrors inference.py: optional telemetry prefix via build_grounding_prompt,
    deterministic greedy by default, output trimmed at </answer> and parsed by
    the shared tag parser.
    """
    prompt_text = build_telemetry_prompt(telemetry) if args.grounded_reasoning else None
    result = model.predict(
        item_pv, tokenizer, freq_features=item_freq,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        prompt_text=prompt_text,
    )
    raw = trim_after_answer(result["reasoning"][0])
    tags = parse_reasoning(raw)
    return {
        "quick_take": tags.get("fast", ""),
        "planning": tags.get("planning", ""),
        "analysis": tags.get("reasoning", ""),
        "reflection": tags.get("reflection", ""),
        "conclusion": tags.get("conclusion", ""),
        "generated_answer": tags.get("answer", "unknown"),
        "answer_heuristic": bool(tags.get("answer_heuristic", False)),
        "raw": raw,
    }


# ── Per-image evaluation ─────────────────────────────────────────────────────

def evaluate_image(image: Image.Image, route: Dict[str, Any], model, tokenizer,
                   transform, device, args, logger) -> Dict[str, Any]:
    """Classify the route's items, reason where asked, then aggregate."""
    items = route["items"]

    probs, freqs, pv = classify(model, [it["image"] for it in items], transform, device)
    attn = attention_weights(model, pv, device)

    rows, telemetries = [], []
    for i, item in enumerate(items):
        p_fake = float(probs[i, 1])
        conf = float(probs[i].max())
        pred = 0 if p_fake < args.face_threshold else 1
        telemetry = extract_forensic_telemetry(
            cls_pred=pred,
            cls_conf=conf,
            freq_features=freqs[i:i + 1],
            attn_weights=attn[i:i + 1],
        )
        telemetries.append(telemetry)
        rows.append({
            "id": item["id"],
            "kind": "whole_image" if item["kind"] == "whole" else "face",
            "verdict": "FAKE" if pred else "REAL",
            "p_fake": round(p_fake, 6),
            "confidence": round(conf, 6),
            "box": item["box"].as_list() if item["box"] is not None else None,
            "crop": item["meta"],
            "telemetry": {
                "hotspots": telemetry["regions"],
                "spectral": telemetry["spectral_desc"],
            },
            "reasoning": None,
        })

    # Reasoning for the selected items only. Classification above is untouched:
    # predict() receives the same crops but never feeds back into the verdict.
    reasoned = []
    for i in select_reason_targets(rows, args.reason_faces):
        rows[i]["reasoning"] = reason_about_item(
            model, tokenizer, pv[i:i + 1], freqs[i:i + 1].to(device),
            telemetries[i], args, device)
        reasoned.append(i)

    doc = aggregate_verdict(rows, args.aggregate, args.face_threshold)
    face_verdicts = [r["verdict"] for r in rows if r["kind"] == "face"] or [r["verdict"] for r in rows]
    # The classifier is the decision authority (INTEGRATION_GUIDE section 9);
    # the LLM answer is recorded and a disagreement escalates to review.
    disagreements = find_disagreements(rows, reasoned)
    flags = {
        "faces_disagree": len(set(face_verdicts)) > 1,
        "model_disagreement": bool(disagreements),
        "requires_human_review": (len(set(face_verdicts)) > 1
                                  or bool(disagreements)
                                  or any(r["confidence"] < 0.70 for r in rows)
                                  or bool(route["skipped"])),
        "faces_skipped": len(route["skipped"]),
        "reasoned_faces": [rows[i]["id"] for i in reasoned],
    }

    logger.info(f"\n{'=' * 64}")
    logger.info(f"  Image             : {getattr(image, 'filename', '')}")
    logger.info(f"  Routing           : {route['path']}  "
                f"(faces detected {len(route['boxes'])}, evaluated {len(rows)})")
    if route["skipped"]:
        logger.info(f"  Skipped           : {route['skipped']}")
    logger.info(f"  Document verdict  : {doc if doc else 'per-face only'} "
                f"(aggregate={args.aggregate}, threshold={args.face_threshold})")
    for r in rows:
        label = "whole image" if r["kind"] == "whole_image" else f"face {r['id']}"
        logger.info(f"    [{label}] {r['verdict']:<4} p_fake={r['p_fake']:.4f} "
                    f"conf={r['confidence']:.1%}  hotspots={r['telemetry']['hotspots']}")
        logger.info(f"              spectral={r['telemetry']['spectral']}")
        if r["crop"]:
            logger.info(f"              crop={r['crop']['crop_size']} "
                        f"pad={r['crop']['pad_mode']} upscale={r['crop']['upscale']}")
        if r["reasoning"]:
            _log_reasoning(logger, label, r["reasoning"], r["verdict"])
    logger.info(f"  Review needed     : {flags['requires_human_review']}"
                + ("   (LLM and classifier disagree)" if disagreements else ""))
    logger.info("=" * 64)

    return {
        "image": getattr(image, "filename", ""),
        "path": route["path"],
        "faces_detected": len(route["boxes"]),
        "faces_evaluated": sum(1 for r in rows if r["kind"] == "face") or len(rows),
        "faces_skipped": route["skipped"],
        "document_verdict": doc,
        "aggregate": args.aggregate,
        "threshold": args.face_threshold,
        "reason_mode": args.reason_faces,
        "faces": rows,
        "flags": flags,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if not (args.adapter_dir or args.checkpoint):
        sys.exit("ERROR: pass --adapter_dir (adapter package) or --checkpoint (whole model)")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    script_dir = os.path.dirname(os.path.abspath(__file__))
    transform = build_transform(args.image_size)

    images = args.images or ["test_images/3-people.jpg", "test_images/multiple-faces.jpg"]
    log_dir = args.log_dir or os.path.join(script_dir, "runs", "inference_logs")
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger(os.path.join(log_dir, f"inference_faces_{ts}.log"))

    logger.info(f"Device        : {device}")
    logger.info(f"Base model    : {args.base_model}")
    logger.info(f"Weights       : {args.adapter_dir or args.checkpoint}")
    logger.info(f"Images        : {images}")
    logger.info(f"Routing       : aggregate={args.aggregate} "
                f"passthrough_min_face_frac={args.passthrough_min_face_frac} "
                f"detect={not args.no_detect}")
    logger.info(f"Reasoning     : {args.reason_faces} "
                f"(grounded={args.grounded_reasoning}, max_new_tokens={args.max_new_tokens}, "
                f"sample={args.do_sample})")

    try:
        model, tokenizer, info = load_model(
            base_model=args.base_model, adapter_dir=args.adapter_dir,
            checkpoint=args.checkpoint, arch_version=args.arch_version,
            attn_implementation=args.attn_implementation, device=str(device))
    except AdapterLoadError as exc:
        sys.exit(f"ERROR: {exc}")
    logger.info(f"Weight source : {info['source']}")

    detector = None
    if not args.no_detect:
        detector = FaceDetector(args.detector_weights, imgsz=args.detector_imgsz,
                               conf=args.detector_conf, iou=args.detector_iou)

    if args.save_crops:
        os.makedirs(args.save_crops, exist_ok=True)
    if args.save_annotated:
        os.makedirs(args.save_annotated, exist_ok=True)

    results = []
    for name in images:
        path = name if os.path.isabs(name) else os.path.join(script_dir, name)
        if not os.path.exists(path):
            path = name
        if not os.path.exists(path):
            logger.info(f"  [{name}] NOT FOUND -- skipping")
            continue
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            logger.info(f"  [{name}] could not open: {exc} -- skipping")
            continue

        image.filename = os.path.basename(path)
        route = route_image(image, detector, args)          # detection happens once

        if args.save_crops:
            for item in route["items"]:
                if item["kind"] != "face":
                    continue
                item["image"].save(os.path.join(
                    args.save_crops, f"{os.path.splitext(image.filename)[0]}_face{item['id']}.jpg"),
                    quality=92)
        if args.save_annotated and route["boxes"]:
            draw_boxes(image, route["boxes"]).save(os.path.join(
                args.save_annotated, f"{os.path.splitext(image.filename)[0]}_annotated.jpg"),
                quality=88)

        results.append(evaluate_image(image, route, model, tokenizer, transform,
                                      device, args, logger))

    logger.info("\nSUMMARY")
    logger.info(f"  {'Image':<24} {'Path':<12} {'Faces':<6} {'Doc verdict':<12}")
    for r in results:
        logger.info(f"  {r['image']:<24} {r['path']:<12} {r['faces_detected']:<6} "
                    f"{r['document_verdict'] or 'per-face':<12}")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nJSON written to {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
