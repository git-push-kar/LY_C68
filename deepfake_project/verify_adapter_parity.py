"""
verify_adapter_parity.py
========================
M2 gate: prove the extracted adapter package produces the SAME classifier output
as the whole-model checkpoint it was extracted from.

Why this exists
---------------
`extract_adapters.py` verifies that tensors save/reload and that a clean base
accepts them via `PeftModel.from_pretrained`. Neither proves that loading those
adapters into `DeepfakeReasoningModel` reproduces `best.pth`: a partly-applied
adapter still loads and still returns confident verdicts. Nothing in the repo
measured that gap, so this script does.

Method
------
Both models are built the same way and fed byte-identical inputs, so any
difference is attributable to weight loading alone:

    inputs  : test_images/real/*  (single-face crops), preprocessed once
    model A : DeepfakeReasoningModel + load_state_dict(best.pth)   <- reference
    model B : DeepfakeReasoningModel + llm/vision LoRA + custom heads

Pass condition: identical verdicts on every image AND max |p_fake difference|
<= --tol. Failure prints the offending rows and exits 1.

Usage:
    python verify_adapter_parity.py
    python verify_adapter_parity.py --limit 8 --device cpu
"""

import argparse
import gc
import glob
import os
import sys

import torch
import torch.nn.functional as F
from PIL import Image

from adapter_loader import attach_adapters, load_backbone, load_checkpoint
from dataset import compute_fft_features
from inference import build_transform

DEFAULT_BASE = "./models/InternVL3-2B"
DEFAULT_ADAPTERS = "./runs/intern_v2_ground/adapters"
DEFAULT_CKPT = "./runs/intern_v2_ground/checkpoints/best.pth"
DEFAULT_GLOB = "test_images/real/*"


def parse_args():
    p = argparse.ArgumentParser(description="Adapter vs whole-model checkpoint parity")
    p.add_argument("--base_model", default=DEFAULT_BASE)
    p.add_argument("--adapter_dir", default=DEFAULT_ADAPTERS)
    p.add_argument("--checkpoint", default=DEFAULT_CKPT)
    p.add_argument("--glob", default=DEFAULT_GLOB, help="single-face image glob")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    p.add_argument("--tol", type=float, default=1e-4,
                   help="max allowed |p_fake(adapter) - p_fake(checkpoint)|")
    p.add_argument("--image_size", type=int, default=448)
    return p.parse_args()


def collect_inputs(paths, image_size):
    tf = build_transform(image_size)
    inputs = []
    for path in paths:
        try:
            img = Image.open(path).convert("RGB")
        except Exception as exc:
            print(f"  [skip] {path}: {exc}")
            continue
        # Preprocess ONCE so both models see identical tensors.
        inputs.append({
            "path": path,
            "pv": tf(img).unsqueeze(0),
            "freq": compute_fft_features(img, target_size=224).unsqueeze(0),
        })
    return inputs


@torch.no_grad()
def predict_all(model, inputs, device):
    rows = []
    for item in inputs:
        pv = item["pv"].to(device)
        freq = item["freq"].to(device)
        autocast_ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
                        if device.type == "cuda" else torch.nullcontext())
        with autocast_ctx:
            logits = model(pv, freq_features=freq)["cls_logits"].float()
        probs = F.softmax(logits, dim=-1)[0].cpu()
        rows.append({
            "path": item["path"],
            "pred": int(probs.argmax()),
            "p_fake": float(probs[1]),
        })
    return rows


def free(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    paths = sorted(glob.glob(args.glob))
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        sys.exit(f"ERROR: no images matched {args.glob!r}")

    print(f"Device      : {device}")
    print(f"Images      : {len(paths)} from {args.glob}")
    print(f"Reference   : {args.checkpoint}")
    print(f"Under test  : {args.adapter_dir}")
    print()

    inputs = collect_inputs(paths, args.image_size)

    print("Loading reference model (whole-model checkpoint) ...")
    ref = load_backbone(args.base_model, attn_implementation="sdpa", device=str(device))
    ref_info = load_checkpoint(ref, args.checkpoint)
    print(f"  epoch={ref_info['epoch']} tensors={ref_info['tensors']} "
          f"missing={ref_info['missing']} unexpected={ref_info['unexpected']}")
    ref_rows = predict_all(ref, inputs, device)
    free(ref)

    print("\nLoading adapter model (llm_adapter + vision_adapter + custom_heads) ...")
    adapt = load_backbone(args.base_model, attn_implementation="sdpa", device=str(device))
    attach_adapters(adapt, args.adapter_dir, base_model=args.base_model)
    adapt_rows = predict_all(adapt, inputs, device)
    free(adapt)

    print(f"\n{'image':<24} {'ckpt':<6} {'adapter':<8} {'p_fake(ckpt)':>12} "
          f"{'p_fake(adp)':>12} {'|diff|':>9}")
    print("-" * 78)
    mismatches, worst = [], 0.0
    for a, b in zip(ref_rows, adapt_rows):
        diff = abs(a["p_fake"] - b["p_fake"])
        worst = max(worst, diff)
        same = a["pred"] == b["pred"]
        if not same:
            mismatches.append((a, b, diff))
        name = os.path.basename(a["path"])
        print(f"{name:<24} {'FAKE' if a['pred'] else 'REAL':<6} "
              f"{'FAKE' if b['pred'] else 'REAL':<8} {a['p_fake']:>12.6f} "
              f"{b['p_fake']:>12.6f} {diff:>9.2e}{'' if same else '   <-- MISMATCH'}")

    print("-" * 78)
    print(f"images={len(ref_rows)}  verdict mismatches={len(mismatches)}  "
          f"max |dp_fake|={worst:.3e}  tol={args.tol:.0e}")

    ok = not mismatches and worst <= args.tol
    if ok:
        print("\nPASS: the adapter package reproduces the checkpoint's verdicts.")
        return 0

    if mismatches:
        print(f"\nFAIL: {len(mismatches)} verdict mismatch(es) -- the adapter path is not "
              f"equivalent to {args.checkpoint}.")
    if worst > args.tol:
        print(f"\nFAIL: probability drift {worst:.3e} exceeds tol {args.tol:.0e} even where "
              f"verdicts agree.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
