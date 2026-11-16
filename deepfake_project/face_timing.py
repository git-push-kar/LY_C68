"""
face_timing.py
==============
Per-stage wall time for the face pipeline (plan section 8.5): detect, classify
and reason, on a 1-face and a 4-face image. Exists so `--reason_faces top1` is a
decision backed by numbers rather than an assumption.

What is timed, and how:

  detect    FaceDetector.detect() -- ultralytics forward + NMS on the full frame
  classify  classify() + attention_weights(): exactly the work evaluate_image
            does before reasoning (batched crops, bf16 autocast, plus the extra
            attention pass telemetry needs)
  reason    reason_about_item() once per face, measured per face; `top1` is the
            single generation for the highest-p_fake face, `all` is the sum of
            those per-face measurements

Every timed call is bracketed by torch.cuda.synchronize(), so the numbers are
wall time and not kernel-launch time. The first call of each stage is a warmup
and is discarded (CUDA context, cuDNN autotuning, allocator growth).

The 4-face image: FFIW's only rated 4-face frame has just two faces above the
96 px gate (sides 133/183/79/64), so it would not exercise four crops. A real
4-photo montage is built instead -- four distinct photographs, each scaled to a
tile, tiled 2x2 -- which detection then finds as four independent faces at
default settings. The FFIW frame is measured too, as `--extra`, to show what a
frame with sub-gate faces actually costs.

Example:
    python face_timing.py --adapter_dir ./runs/intern_fused_ground/adapters \
        --out runs/timing/face_timing.json
"""

import argparse
import json
import os
import statistics
import time
from types import SimpleNamespace
from typing import Any, Dict, List

import torch
from PIL import Image

from adapter_loader import AdapterLoadError, load_model
from face_detect import FaceDetector
from inference import build_transform, extract_forensic_telemetry
from inference_faces import (attention_weights, classify, reason_about_item,
                             route_image)

# Defaults mirror inference_faces: greedy, grounded, 512 tokens.
REASON_ARGS = SimpleNamespace(grounded_reasoning=True, max_new_tokens=512,
                              do_sample=False, temperature=1.0, top_p=1.0,
                              repetition_penalty=1.05)
ROUTING = SimpleNamespace(margin=0.30, min_face_px=96, max_faces=8,
                          image_size=448, passthrough_min_face_frac=0.05)

MONTAGE_SOURCES = ("test_images/omi.jpg", "test_images/2.jpg",
                   "test_images/test1.png", "test_images/mar.jpg")
FFIW_FOUR_FACE = "datasets/hydrafake/test/cd/FFIW/1_fake/test_00000929/053.png"


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def timed(fn, repeats: int, device: torch.device, warmup: int = 1):
    """Run ``fn`` warmup+repeats times; return per-call milliseconds and the last result."""
    for _ in range(warmup):
        fn()
    _sync(device)
    samples: List[float] = []
    out = None
    for _ in range(repeats):
        _sync(device)
        t0 = time.perf_counter()
        out = fn()
        _sync(device)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return samples, out


def build_montage(sources, out_path: str, tile: int = 800) -> str:
    """Tile four photos into one 2x2 image so detection sees four real faces."""
    canvas = Image.new("RGB", (tile * 2, tile * 2), (128, 128, 128))
    for i, path in enumerate(sources[:4]):
        img = Image.open(path).convert("RGB").resize((tile, tile), Image.LANCZOS)
        canvas.paste(img, ((i % 2) * tile, (i // 2) * tile))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    canvas.save(out_path, quality=95)
    return out_path


def profile(name: str, image_path: str, model, tokenizer, detector, transform,
            device, repeats_fast: int, repeats_reason: int,
            skip_reason: bool = False) -> Dict[str, Any]:
    image = Image.open(image_path).convert("RGB")

    det_samples, _ = timed(lambda: detector.detect(image), repeats_fast, device)

    route = route_image(image, detector, ROUTING)
    items = route["items"]
    crops = [it["image"] for it in items]

    def classify_once():
        probs, freqs, pv = classify(model, crops, transform, device)
        return probs, freqs, pv, attention_weights(model, pv, device)

    cls_samples, (probs, freqs, pv, attn) = timed(classify_once, repeats_fast, device)

    telemetries = []
    for i in range(len(items)):
        p_fake = float(probs[i, 1])
        pred = 0 if p_fake < 0.50 else 1
        telemetries.append(extract_forensic_telemetry(
            cls_pred=pred, cls_conf=float(probs[i].max()),
            freq_features=freqs[i:i + 1], attn_weights=attn[i:i + 1]))

    per_face_ms: List[float] = []
    answers: List[str] = []
    for i in range(len(items) if not skip_reason else 0):
        samples, result = timed(
            lambda i=i: reason_about_item(
                model, tokenizer, pv[i:i + 1], freqs[i:i + 1].to(device),
                telemetries[i], REASON_ARGS, device),
            repeats_reason, device)
        per_face_ms.append(statistics.mean(samples))
        answers.append(result.get("generated_answer", "unknown") if result else "unknown")

    top1_index = max(range(len(items)), key=lambda i: (float(probs[i, 1]), -i)) if items else None
    detect_ms = statistics.mean(det_samples)
    classify_ms = statistics.mean(cls_samples)
    # Mean and median both: a single slow first call after warmup can double the
    # mean of ten samples, and the median is the honest central estimate.
    top1_ms = (per_face_ms[top1_index]
               if per_face_ms and top1_index is not None else 0.0)
    all_ms = sum(per_face_ms)

    return {
        "name": name,
        "image": image_path,
        "size": [image.width, image.height],
        "path": route["path"],
        "faces_detected": len(route["boxes"]),
        "faces_evaluated": len(items),
        "faces_skipped": len(route["skipped"]),
        "skip_reasons": [s.get("reason") for s in route["skipped"]],
        "detect_ms": detect_ms,
        "detect_median_ms": statistics.median(det_samples),
        "detect_min_ms": min(det_samples),
        "detect_max_ms": max(det_samples),
        "detect_sd": statistics.pstdev(det_samples) if len(det_samples) > 1 else 0.0,
        "classify_ms": classify_ms,
        "classify_median_ms": statistics.median(cls_samples),
        "classify_min_ms": min(cls_samples),
        "classify_max_ms": max(cls_samples),
        "classify_sd": statistics.pstdev(cls_samples) if len(cls_samples) > 1 else 0.0,
        "reason_per_face_ms": per_face_ms,
        "reason_top1_ms": top1_ms,
        "reason_all_ms": all_ms,
        "total_top1_ms": detect_ms + classify_ms + top1_ms,
        "total_all_ms": detect_ms + classify_ms + all_ms,
        "llm_answers": answers,
        "repeats_detect_classify": repeats_fast,
        "repeats_reason": repeats_reason,
    }


def print_table(rows: List[Dict[str, Any]]) -> None:
    print(f"\n{'image':<22} {'faces':>5} {'path':<12} {'detect':>8} {'classify':>9} "
          f"{'reason 1':>9} {'top1 tot':>9} {'all tot':>9}")
    for r in rows:
        print(f"{r['name']:<22} {r['faces_evaluated']:>5} {r['path']:<12} "
              f"{r['detect_ms']:>7.0f}ms {r['classify_ms']:>8.0f}ms "
              f"{r['reason_per_face_ms'][0] if r['reason_per_face_ms'] else 0:>8.0f}ms "
              f"{r['total_top1_ms']:>8.0f}ms {r['total_all_ms']:>8.0f}ms")
    print()


def main() -> int:
    p = argparse.ArgumentParser(description="Per-stage face pipeline timing")
    p.add_argument("--adapter_dir", default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--base_model", default="./models/internvl3_2b_lr_fused")
    p.add_argument("--arch_version", default="v2")
    p.add_argument("--attn_implementation", default="sdpa")
    p.add_argument("--device", default=None)
    p.add_argument("--one", default="test_images/omi.jpg", help="1-face image")
    p.add_argument("--four", default="runs/timing/quad.jpg", help="4-face image")
    p.add_argument("--montage_sources", nargs="+", default=list(MONTAGE_SOURCES))
    p.add_argument("--tile", type=int, default=800)
    p.add_argument("--extra", default=FFIW_FOUR_FACE,
                   help="additional image (the FFIW 4-face frame, 2 above the gate)")
    p.add_argument("--repeats_fast", type=int, default=10)
    p.add_argument("--repeats_reason", type=int, default=2)
    p.add_argument("--skip_reason", action="store_true",
                   help="time detect+classify only (cheap, many repeats); reason fields stay 0")
    p.add_argument("--detector_weights", default="./models/yolov8n-face.pt")
    p.add_argument("--out", default="runs/timing/face_timing.json")
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    try:
        model, tokenizer, info = load_model(
            base_model=args.base_model, adapter_dir=args.adapter_dir,
            checkpoint=args.checkpoint, arch_version=args.arch_version,
            attn_implementation=args.attn_implementation, device=str(device))
    except AdapterLoadError as exc:
        raise SystemExit(f"ERROR: {exc}")
    transform = build_transform(448)
    detector = FaceDetector(args.detector_weights,
                            device=str(device) if device.type == "cuda" else None)

    if not os.path.exists(args.four):
        build_montage(args.montage_sources, args.four, args.tile)
        print(f"built 4-face montage -> {args.four}")

    targets = [("1-face", args.one), ("4-face", args.four)]
    if args.extra and os.path.exists(args.extra):
        targets.append(("FFIW 4-face", args.extra))

    rows = []
    for name, path in targets:
        row = profile(name, path, model, tokenizer, detector, transform, device,
                      args.repeats_fast, args.repeats_reason, args.skip_reason)
        rows.append(row)
        print(f"{name}: {row['path']}, {row['faces_evaluated']} evaluated, "
              f"{row['faces_skipped']} skipped, detect {row['detect_ms']:.0f} ms, "
              f"classify {row['classify_ms']:.0f} ms, "
              f"reason/face {row['reason_per_face_ms']} ms", flush=True)

    print_table(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                   "rows": rows}, f, indent=2)
    print(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
