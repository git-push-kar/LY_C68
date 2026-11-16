"""
eval_metrics.py
===============
Per-split F1, precision/recall and confidence intervals from an archived eval log.

`eval.py` prints N, GT R/F, Pred R/F, Acc, F1, AUC per split but never precision
or recall, and never an error bar. Those are recoverable: the four cells of the
confusion matrix are constrained by the printed counts and accuracy,

    d      = pred_fake - gt_fake        (== FP - FN)
    FP     = (pred_fake + gt_real - acc*N) / 2
    TP     = pred_fake - FP,  FN = gt_fake - TP,  TN = gt_real - FP

and the reconstruction is *checked* against the printed F1, so a parsing or
rounding error cannot slip through silently.

Intervals: Wilson score for proportions (accuracy, precision, recall). AUC is
printed without per-sample scores, so its interval is Hanley-McNeil's closed form
from AUC and the class counts -- an approximation, labelled as one. When scores
are available (the FFIW cache) `face_calibration.bootstrap_ci` is used instead.

Usage:
    python eval_metrics.py --log runs/intern_fused_ground/logs/eval_best.log
"""

import argparse
import math
import re
import sys
from typing import Dict, List, Optional

ROW_RE = re.compile(
    r"(?P<split>ID|CM|CF|CD)\s+\((?P<label>[^)]*)\)\s+"
    r"(?P<n>\d+)\s+(?P<gt_real>\d+)/(?P<gt_fake>\d+)\s+"
    r"(?P<pred_real>\d+)/(?P<pred_fake>\d+)\s+"
    r"(?P<acc>[\d.]+)\s+(?P<f1>[\d.]+)\s+(?P<auc>[\d.]+)")


def wilson(successes: float, n: float, z: float = 1.959963985) -> tuple:
    """Wilson score interval for a binomial proportion (works at n=0)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def hanley_mcneil(auc: float, n_pos: int, n_neg: int, z: float = 1.959963985) -> tuple:
    """Approximate AUC interval (Hanley & McNeil 1982). Labelled as approximate."""
    if n_pos <= 0 or n_neg <= 0 or not (0.0 < auc < 1.0):
        return (float("nan"), float("nan"))
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc * auc)
           + (n_neg - 1) * (q2 - auc * auc)) / (n_pos * n_neg)
    se = math.sqrt(max(var, 0.0))
    return (max(0.0, auc - z * se), min(1.0, auc + z * se))


def cells_from_counts(gt_real: int, gt_fake: int, pred_real: int, pred_fake: int,
                      acc: float, f1: float) -> Dict[str, int]:
    """Recover TP/FP/FN/TN from the printed counts, verified against printed F1."""
    n = gt_real + gt_fake
    estimate = (pred_fake + gt_real - acc * n) / 2.0
    best: Optional[Dict[str, int]] = None
    for fp in {int(round(estimate)) + k for k in range(-4, 5)}:
        tp = pred_fake - fp
        fn = gt_fake - tp
        tn = gt_real - fp
        if min(tp, fp, fn, tn) < 0:
            continue
        implied_f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        if best is None or abs(implied_f1 - f1) < best["_err"]:
            best = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "_err": abs(implied_f1 - f1)}
    if best is None or best["_err"] > 2e-3:
        raise ValueError(
            f"cannot reconstruct the confusion matrix (N={n}, printed F1={f1}, "
            f"best implied error={None if best is None else best['_err']:.4f})")
    best.pop("_err")
    return best


def parse_log(path: str) -> List[Dict[str, float]]:
    rows = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = ROW_RE.search(line)
            if not m:
                continue
            g = m.groupdict()
            gt_real, gt_fake = int(g["gt_real"]), int(g["gt_fake"])
            pred_real, pred_fake = int(g["pred_real"]), int(g["pred_fake"])
            acc, f1, auc = float(g["acc"]), float(g["f1"]), float(g["auc"])
            cells = cells_from_counts(gt_real, gt_fake, pred_real, pred_fake, acc, f1)
            precision = cells["tp"] / (cells["tp"] + cells["fp"]) if cells["tp"] + cells["fp"] else 0.0
            recall = cells["tp"] / (cells["tp"] + cells["fn"]) if cells["tp"] + cells["fn"] else 0.0
            rows.append({
                "split": g["split"], "label": g["label"], "n": gt_real + gt_fake,
                "gt_real": gt_real, "gt_fake": gt_fake,
                "pred_real": pred_real, "pred_fake": pred_fake,
                **cells, "acc": acc, "f1": f1, "auc": auc,
                "precision": precision, "recall": recall,
                "acc_ci": wilson(cells["tp"] + cells["tn"], gt_real + gt_fake),
                "precision_ci": wilson(cells["tp"], cells["tp"] + cells["fp"]),
                "recall_ci": wilson(cells["tp"], cells["tp"] + cells["fn"]),
                "auc_ci": hanley_mcneil(auc, gt_fake, gt_real),
            })
    return rows


def fmt(ci) -> str:
    return f"[{ci[0]:.4f}, {ci[1]:.4f}]"


def report(rows: List[Dict[str, float]], path: str) -> str:
    out = [f"source: {path}", "",
           "| Split | N | Acc [95% CI] | F1 | Precision [CI] | Recall [CI] | AUC [CI] |",
           "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        out.append(f"| {r['split']} ({r['label']}) | {r['n']:,} | "
                   f"{r['acc']:.4f} {fmt(r['acc_ci'])} | {r['f1']:.4f} | "
                   f"{r['precision']:.4f} {fmt(r['precision_ci'])} | "
                   f"{r['recall']:.4f} {fmt(r['recall_ci'])} | "
                   f"{r['auc']:.4f} {fmt(r['auc_ci'])} |")
    out.append("")
    out.append("Positive class = FAKE. TP/FP/FN/TN reconstructed from the logged counts and "
               "verified against the logged F1.")
    out.append("Acc/precision/recall: Wilson 95%. AUC: Hanley-McNeil approximation "
               "(no per-sample scores in the log).")
    return "\n".join(out)


def main() -> int:
    p = argparse.ArgumentParser(description="Per-split F1/PR with CIs from an eval log")
    p.add_argument("--log", default="runs/intern_fused_ground/logs/eval_best.log")
    p.add_argument("--json_out", default=None)
    args = p.parse_args()
    rows = parse_log(args.log)
    if not rows:
        print(f"no split rows found in {args.log}", file=sys.stderr)
        return 1
    print(report(rows, args.log))
    if args.json_out:
        import json
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        print(f"\njson -> {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
