"""Weight-diff probe for the stage history documented in PAPER_CLAIMS.md §1.5.

Shows, straight from the checkpoint files, whether a parameter group actually
moved between two checkpoints, and prints the recorded `stage` / `epoch`
metadata. It is read-only and uses torch.load(..., mmap=True), so it never
materialises a 4.7 GB state dict in RAM.

Usage
-----
    python verify_stage_history.py                 # the §1.5 pairs
    python verify_stage_history.py --pair A.pth B.pth
    python verify_stage_history.py --pair A B --pair C D --json

Interpretation of the §1.5 pairs
--------------------------------
  ep003 -> ep004 (intern_v2)   cls_only -> joint: cls_head and attn_pool move,
                               custom_projector / domain_router stay identical
                               (they are off the LM training path).
  ep001 -> ep003 (ground)      the grounded block: cls_head frozen (Δ = 0),
                               attn_pool + freq_branch keep training.
Batch-norm bookkeeping (`num_batches_tracked`, `running_mean`, `running_var`)
is excluded by default: those buffers change by counting/observing iterations,
not by learning, so they would mask a frozen weight. Pass --include-buffers to
see them anyway.
"""

import argparse
import collections
import json
import os
import sys

import torch

GROUPS = ("attn_pool", "freq_branch", "cls_head", "custom_projector", "domain_router")
BUFFER_PATTERNS = ("num_batches_tracked", "running_mean", "running_var")

DEFAULT_PAIRS = (
    ("runs/intern_v2/checkpoints/ep003.pth", "runs/intern_v2/checkpoints/ep004.pth",
     "intern_v2 ep003 -> ep004 (cls_only -> joint)"),
    ("runs/intern_v2_ground/checkpoints/ep001.pth", "runs/intern_v2_ground/checkpoints/ep003.pth",
     "intern_v2_ground ep001 -> ep003 (grounded block)"),
)


def load(path):
    ckpt = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    state = ckpt.get("model", ckpt)
    meta = {k: v for k, v in ckpt.items() if k != "model"}
    return state, meta


INCLUDE_BUFFERS = False


def is_buffer(name):
    if INCLUDE_BUFFERS:
        return False
    return any(p in name for p in BUFFER_PATTERNS)


def group_diffs(a, b, groups=GROUPS):
    """Max abs diff per parameter group, ignoring buffers and non-finite values."""
    agg = collections.defaultdict(lambda: {"tensors": 0, "max_abs_diff": 0.0, "argmax": ""})
    for name, tensor in a.items():
        if name not in b or is_buffer(name):
            continue
        if not torch.is_floating_point(tensor):
            continue
        for g in groups:
            if name.startswith(g):
                d = (tensor.float() - b[name].float()).abs().max().item()
                if d != d:  # NaN
                    d = float("inf")
                cell = agg[g]
                cell["tensors"] += 1
                if d >= cell["max_abs_diff"]:
                    cell["max_abs_diff"] = d
                    cell["argmax"] = name
                break
    lora = [n for n in a if "lora_" in n and n in b]
    if lora:
        d = max((a[n].float() - b[n].float()).abs().max().item() for n in lora)
        agg["lora"] = {"tensors": len(lora), "max_abs_diff": d, "argmax": "all lora tensors"}
    return agg


def compare(path_a, path_b, label):
    state_a, meta_a = load(path_a)
    state_b, meta_b = load(path_b)
    result = {
        "label": label,
        "a": path_a,
        "b": path_b,
        "stage": [meta_a.get("stage"), meta_b.get("stage")],
        "epoch": [meta_a.get("epoch"), meta_b.get("epoch")],
        "best_auc": [meta_a.get("best_auc"), meta_b.get("best_auc")],
        "groups": group_diffs(state_a, state_b),
    }
    return result


def report(result):
    print(f"\n=== {result['label']}")
    print(f"  A {result['a']}  stage={result['stage'][0]} epoch={result['epoch'][0]}")
    print(f"  B {result['b']}  stage={result['stage'][1]} epoch={result['epoch'][1]}")
    print(f"  {'group':<17}{'tensors':>8}{'max abs diff':>14}   argmax tensor")
    for g in list(GROUPS) + ["lora"]:
        cell = result["groups"].get(g)
        if not cell:
            continue
        flag = "   <-- identical (frozen / off the training path)" if cell["max_abs_diff"] == 0.0 else ""
        print(f"  {g:<17}{cell['tensors']:>8}{cell['max_abs_diff']:>14.6g}   {cell['argmax']}{flag}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pair", nargs=2, action="append", metavar=("A", "B"),
                    help="two checkpoints to diff; repeatable")
    ap.add_argument("--label", action="append", default=None,
                    help="label per --pair, in the same order")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    ap.add_argument("--include-buffers", action="store_true",
                    help="also diff batch-norm bookkeeping buffers (num_batches_tracked, running_mean/var)")
    args = ap.parse_args(argv)

    global INCLUDE_BUFFERS
    INCLUDE_BUFFERS = args.include_buffers

    if args.pair:
        pairs = [(a, b, None) for a, b in args.pair]
    else:
        pairs = DEFAULT_PAIRS

    labels = args.label or []
    results = []
    for i, (a, b, default_label) in enumerate(pairs):
        label = labels[i] if i < len(labels) else (default_label or f"{os.path.basename(a)} -> {os.path.basename(b)}")
        for p in (a, b):
            if not os.path.isfile(p):
                print(f"missing checkpoint: {p}", file=sys.stderr)
                return 2
        results.append(compare(a, b, label))

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for r in results:
            report(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
