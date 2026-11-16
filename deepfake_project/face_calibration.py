"""
face_calibration.py
===================
M5: margin sweep + FFIW document-level evaluation, and the two numbers they
calibrate (`face_threshold`, `passthrough_min_face_frac`).

What this measures, and why it is not just "run inference_faces on FFIW":

  FFIW as stored here is 256x256, and its frames are single-face (a probe of 40
  frames found 38 with exactly one face, face area 0.27-0.54 of the frame). Every
  such frame routes PASSTHROUGH, so running the shipped pipeline over FFIW would
  only ever exercise the whole-image path -- it could not say anything about the
  margin. So this script classifies, for every frame:

    * the whole image                       (what passthrough would use)
    * one context crop per detected face,   (what the per-face path would use)
      for every margin in --margins

  Then it reports margin-level AUC, the document-level (max-aggregated) AUC, the
  threshold sweep, and a passthrough-fraction sweep. Frames are classified
  regardless of which route they would take, which is what makes the margin
  comparison possible at all.

Label convention: FFIW `0_real`/`1_fake` directories give a frame-level label,
which is also the document-level label here (a document is fake if any face is
fake, and there is one face).

Caching: every frame is appended to --cache as JSON as soon as it is scored, and
finished frames are skipped on rerun, so an interrupted multi-minute run
resumes instead of starting over.

Examples:
    python face_calibration.py --adapter_dir ./runs/intern_fused_ground/adapters \
        --n_per_class 200 --out runs/calibration/ffiw_report.json

    # reuse the cache, only redo the analysis
    python face_calibration.py --adapter_dir ./runs/intern_fused_ground/adapters \
        --analyze_only --out runs/calibration/ffiw_report.json
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from adapter_loader import AdapterLoadError, load_model
from face_detect import FaceDetector, crop_with_context
from inference import build_transform
from inference_faces import classify, lone_face_fills_frame

MARGINS = (0.00, 0.15, 0.30, 0.45)
FRAC_CANDIDATES = (0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30)


# ── Manifest ─────────────────────────────────────────────────────────────────

def build_manifest(root: str, n_per_class: int, seed: int) -> List[Dict[str, Any]]:
    """One frame per video directory, sampled to ``n_per_class`` per label.

    FFIW stores ~2 frames per video, so taking one frame per video avoids
    near-duplicate pairs inflating the sample. The shuffle is seeded, so a
    rerun evaluates exactly the same frames.
    """
    root_p = Path(root)
    out: List[Dict[str, Any]] = []
    for label, sub in ((0, "0_real"), (1, "1_fake")):
        videos = sorted({p.parent for p in root_p.glob(f"{sub}/*/*")})
        rng = random.Random(f"{seed}:{label}")
        rng.shuffle(videos)
        for video in sorted(videos[:n_per_class]):
            frames = sorted(f for f in video.glob("*") if f.is_file())
            if not frames:
                continue
            out.append({"path": frames[0], "label": label,
                        "rel": frames[0].relative_to(root_p).as_posix()})
    return out


# ── Scoring ──────────────────────────────────────────────────────────────────

def _score_frame(image: Image.Image, boxes, model, transform, device,
                 margins) -> Dict[str, Any]:
    """Whole-image p plus one p per (margin, face). One forward per frame."""
    w, h = image.size
    images, plan = [image], [("whole", None, None)]
    metas: Dict[str, List[Any]] = {f"{m}": [] for m in margins}
    for m in margins:
        for i, box in enumerate(boxes):
            try:
                crop, meta = crop_with_context(image, box, margin=m)
            except Exception as exc:                      # degenerate box
                metas[f"{m}"].append({"p": None, "error": str(exc)})
                continue
            images.append(crop)
            plan.append(("crop", m, len(metas[f"{m}"])))
            metas[f"{m}"].append({"p": None, "pad_mode": meta["pad_mode"],
                                 "crop_size": meta["crop_size"],
                                 "upscale": meta["upscale"]})

    probs, _, _ = classify(model, images, transform, device)
    p = probs[:, 1].tolist()

    record: Dict[str, Any] = {
        "w": w, "h": h,
        "faces": [{"box": b.as_list(),
                   "side": round(min(b.width, b.height), 1),
                   "frac": round(b.area / float(w * h), 4)} for b in boxes],
        "whole_p": round(float(p[0]), 6),
        "margins": {},
    }
    # plan[i] lines up with probs row i: row 0 is the whole image, then the
    # crops in margin-then-face order. Rebuild the per-margin lists from it so no
    # index arithmetic can drift.
    per_margin: Dict[str, List[Dict[str, Any]]] = {f"{m}": [] for m in margins}
    for i, (key, m, idx) in enumerate(plan):
        if key != "crop":
            continue
        per_margin[f"{m}"].append({**metas[f"{m}"][idx], "p": round(float(p[i]), 6)})
    record["margins"] = per_margin
    return record


def score_paths(paths: List[Dict[str, Any]], model, transform, detector, device,
                margins, cache_path: str, label_key: str = "label") -> List[Dict[str, Any]]:
    """Score every path not already in the cache, appending to the cache as it goes."""
    done: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:            # truncated last line
                    continue
                done[rec["rel"]] = rec

    todo = [p for p in paths if p["rel"] not in done]
    print(f"cache: {len(done)} done, {len(todo)} to score", flush=True)
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    t0 = time.time()
    with open(cache_path, "a", encoding="utf-8") as fh:
        for n, item in enumerate(todo, 1):
            image = Image.open(item["path"]).convert("RGB")
            boxes = [b.clamped(image.width, image.height) for b in detector.detect(image)]
            record = {"rel": item["rel"], label_key: item.get(label_key),
                      "set": item.get("set", "ffiw"),
                      **_score_frame(image, boxes, model, transform, device, margins)}
            fh.write(json.dumps(record) + "\n")
            fh.flush()
            done[item["rel"]] = record
            if n % 20 == 0 or n == len(todo):
                rate = (time.time() - t0) / n
                print(f"  {n}/{len(todo)}  {rate:.2f}s/frame  "
                      f"eta {(len(todo) - n) * rate / 60:.1f} min", flush=True)
    return [done[p["rel"]] for p in paths if p["rel"] in done]


# ── Analysis (pure functions; no torch) ──────────────────────────────────────

def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    """Rank-based AUC, so this module has no sklearn dependency."""
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(sorted_s):                      # average ranks over ties
        j = i
        while j + 1 < len(sorted_s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def bootstrap_ci(y: np.ndarray, s: np.ndarray, n_boot: int = 1000, seed: int = 0,
                 alpha: float = 0.05) -> tuple:
    """Percentile bootstrap interval for AUC (scores are available here)."""
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    if len(y) < 2 or len(set(y.tolist())) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx].tolist())) < 2:
            continue
        stats.append(roc_auc(y[idx], s[idx]))
    if not stats:
        return (float("nan"), float("nan"))
    return (float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2)))


def face_rows(records: List[Dict[str, Any]], margin: float):
    """Yield (label, p, side, pad_mode) for every scored face at this margin."""
    for r in records:
        for face, entry in zip(r["faces"], r["margins"][f"{margin}"]):
            if entry.get("p") is None:
                continue
            yield (r["label"], entry["p"], face["side"], entry.get("pad_mode") or "unknown")


def auc_by(records: List[Dict[str, Any]], margin: float, key: str,
           min_face_px: int) -> List[Dict[str, Any]]:
    """Face AUC split by size gate ('size') or padding mode ('pad_mode').

    The size split answers whether min_face_px excludes faces that could have
    been judged; the pad split answers whether reflected/stretched padding is
    injecting the high-frequency evidence the FFT branch reads.
    """
    groups: Dict[str, List[tuple]] = {}
    for label, p, side, pad in face_rows(records, margin):
        if key == "size":
            name = f">= {min_face_px}px" if side >= min_face_px else f"< {min_face_px}px"
        else:
            name = pad
        groups.setdefault(name, []).append((label, p))
    out = []
    for name, pairs in sorted(groups.items()):
        y = np.asarray([a for a, _ in pairs], dtype=int)
        s = np.asarray([b for _, b in pairs], dtype=float)
        out.append({"group": name, "faces": len(pairs),
                    "positives": int(y.sum()), "auc": roc_auc(y, s),
                    "ci": bootstrap_ci(y, s)})
    return out


def face_probs(records: List[Dict[str, Any]], margin: float):
    """(labels, p) for every face that was scored at this margin."""
    y, s = [], []
    for r in records:
        for entry in r["margins"][f"{margin}"]:
            if entry.get("p") is not None:
                y.append(r["label"])
                s.append(entry["p"])
    return np.asarray(y, dtype=int), np.asarray(s, dtype=float)


def doc_probs(records: List[Dict[str, Any]], margin: float):
    """(labels, doc p) with the pipeline's max aggregation and no-face fallback."""
    y, s = [], []
    for r in records:
        ps = [e["p"] for e in r["margins"][f"{margin}"] if e.get("p") is not None]
        y.append(r["label"])
        s.append(max(ps) if ps else r["whole_p"])
    return np.asarray(y, dtype=int), np.asarray(s, dtype=float)


def sweep_threshold(y: np.ndarray, s: np.ndarray,
                    lo: float = 0.30, hi: float = 0.75,
                    step: float = 0.01) -> List[Dict[str, float]]:
    """Accuracy / TPR / TNR / F1 / Youden for each candidate threshold."""
    rows = []
    for t in np.round(np.arange(lo, hi + 1e-9, step), 4):
        pred = (s >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        tpr = tp / (tp + fn) if (tp + fn) else float("nan")
        tnr = tn / (tn + fp) if (tn + fp) else float("nan")
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2 * prec * tpr / (prec + tpr) if (prec + tpr) else 0.0
        rows.append({"threshold": float(t), "acc": (tp + tn) / len(y),
                     "tpr": tpr, "tnr": tnr, "f1": f1,
                     "youden": tpr + tnr - 1.0})
    return rows


def best_threshold(rows: List[Dict[str, float]], key: str = "youden") -> Dict[str, float]:
    """Highest ``key``; ties resolved to the threshold closest to 0.5."""
    return max(rows, key=lambda r: (round(r[key], 6), -abs(r["threshold"] - 0.5)))


def route_at_frac(record: Dict[str, Any], frac: float, margin: float,
                  min_face_px: int) -> Dict[str, Any]:
    """Mirror route_image() for one frame at a candidate passthrough floor.

    The passthrough predicate itself comes from inference_faces.lone_face_fills_frame,
    so this sweep measures the shipped rule rather than a copy of it.
    """
    faces = record["faces"]
    if lone_face_fills_frame([f["frac"] for f in faces], frac):
        return {"path": "passthrough", "p": record["whole_p"]}
    kept = [e["p"] for f, e in zip(faces, record["margins"][f"{margin}"])
            if f["side"] >= min_face_px and e.get("p") is not None]
    if not kept:
        return {"path": "no_face", "p": record["whole_p"]}
    return {"path": "per_face", "p": max(kept)}


def frac_sweep(records: List[Dict[str, Any]], margin: float, threshold: float,
               min_face_px: int, candidates=FRAC_CANDIDATES) -> List[Dict[str, Any]]:
    """Document AUC/Acc for each candidate passthrough floor."""
    y = np.asarray([r["label"] for r in records], dtype=int)
    out = []
    for frac in candidates:
        routed = [route_at_frac(r, frac, margin, min_face_px) for r in records]
        s = np.asarray([r["p"] for r in routed], dtype=float)
        paths = [r["path"] for r in routed]
        out.append({
            "frac": frac,
            "auc": roc_auc(y, s),
            "acc": float(((s >= threshold).astype(int) == y).mean()),
            "passthrough_share": paths.count("passthrough") / len(paths),
            "no_face_share": paths.count("no_face") / len(paths),
            "per_face_share": paths.count("per_face") / len(paths),
        })
    return out


def margin_table(records: List[Dict[str, Any]], margins, threshold: float) -> List[Dict[str, Any]]:
    rows = []
    for m in margins:
        yf, sf = face_probs(records, m)
        yd, sd = doc_probs(records, m)
        upscale = [e.get("upscale") for r in records
                   for e in r["margins"][f"{m}"] if e.get("upscale")]
        rows.append({
            "margin": m,
            "n_faces": len(sf),
            "face_auc": roc_auc(yf, sf),
            "doc_auc": roc_auc(yd, sd),
            "doc_acc": float(((sd >= threshold).astype(int) == yd).mean()),
            "mean_upscale": float(np.mean(upscale)) if upscale else float("nan"),
            "share_upscale_gt2": float(np.mean([u > 2.0 for u in upscale])) if upscale else float("nan"),
            "pad_modes": {k: sum(1 for r in records for e in r["margins"][f"{m}"]
                                 if e.get("pad_mode") == k)
                          for k in ("none", "reflect", "stretch")},
        })
    return rows


def summarize(records: List[Dict[str, Any]], margins, min_face_px: int,
              threshold: float, shipped_margin: float = 0.30) -> Dict[str, Any]:
    labeled = [r for r in records if r.get("label") is not None]
    y = np.asarray([r["label"] for r in labeled], dtype=int)
    ys, sw = np.asarray([r["label"] for r in labeled]), np.asarray([r["whole_p"] for r in labeled])
    faces_per_frame = [len(r["faces"]) for r in labeled]
    table = margin_table(labeled, margins, threshold)
    best = max(table, key=lambda r: r["doc_auc"])
    chosen = best["margin"]
    yd, sd = doc_probs(labeled, chosen)
    per_face_y, per_face_s = face_probs(labeled, chosen)

    return {
        "n_frames": len(labeled),
        "positives": int(y.sum()), "negatives": int((y == 0).sum()),
        "faces_per_frame": {str(k): faces_per_frame.count(k) for k in sorted(set(faces_per_frame))},
        "face_frac_range": [min(f["frac"] for r in labeled for f in r["faces"]),
                            max(f["frac"] for r in labeled for f in r["faces"])],
        "whole_image": {"auc": roc_auc(ys, sw),
                        "acc_at_0.50": float(((sw >= 0.5).astype(int) == ys).mean()),
                        "acc_at_0.42": float(((sw >= 0.42).astype(int) == ys).mean())},
        "margins": table,
        "chosen_margin": chosen,
        "face_auc_by_size": auc_by(labeled, chosen, "size", min_face_px),
        "face_auc_by_pad_mode": auc_by(labeled, chosen, "pad_mode", min_face_px),
        # The shipped default is 0.30; report the same splits there too, because
        # pad modes are wildly unbalanced at the sweep's data-chosen margin.
        "shipped_margin": shipped_margin,
        "face_auc_by_size_shipped": auc_by(labeled, shipped_margin, "size", min_face_px),
        "face_auc_by_pad_mode_shipped": auc_by(labeled, shipped_margin, "pad_mode", min_face_px),
        "doc_auc_ci_by_margin": {str(m["margin"]): bootstrap_ci(*doc_probs(labeled, m["margin"]))
                                 for m in table},
        "threshold_sweep_doc": sweep_threshold(yd, sd),
        "threshold_sweep_face": sweep_threshold(per_face_y, per_face_s),
        "frac_sweep": frac_sweep(labeled, chosen, threshold, min_face_px),
    }


# ── Reporting ────────────────────────────────────────────────────────────────

def print_report(rep: Dict[str, Any], margins) -> None:
    print("\n" + "=" * 78)
    print(f"FFIW subset: {rep['n_frames']} frames "
          f"({rep['positives']} fake / {rep['negatives']} real)")
    print(f"faces per frame: {rep['faces_per_frame']}")
    print(f"face area of frame: {rep['face_frac_range'][0]:.3f} - {rep['face_frac_range'][1]:.3f}")
    w = rep["whole_image"]
    print(f"whole image: AUC {w['auc']:.4f}  acc@0.50 {w['acc_at_0.50']:.4f}  "
          f"acc@0.42 {w['acc_at_0.42']:.4f}")
    print("\nMARGIN TABLE")
    print(f"{'margin':>7} {'faces':>6} {'face AUC':>9} {'doc AUC':>8} {'doc acc':>8} "
          f"{'upscale':>8} {'>2x':>6}")
    for r in rep["margins"]:
        print(f"{r['margin']:>7.2f} {r['n_faces']:>6} {r['face_auc']:>9.4f} "
              f"{r['doc_auc']:>8.4f} {r['doc_acc']:>8.4f} {r['mean_upscale']:>8.2f} "
              f"{r['share_upscale_gt2']:>6.0%}")
    print("pad modes:", {r["margin"]: r["pad_modes"] for r in rep["margins"]})
    for key, title, m in (("face_auc_by_size", "FACE AUC BY SIZE GATE", rep["chosen_margin"]),
                          ("face_auc_by_pad_mode", "FACE AUC BY PAD MODE", rep["chosen_margin"]),
                          ("face_auc_by_size_shipped", "FACE AUC BY SIZE GATE", rep["shipped_margin"]),
                          ("face_auc_by_pad_mode_shipped", "FACE AUC BY PAD MODE", rep["shipped_margin"])):
        print(f"\n{title} (margin {m})")
        for r in rep.get(key, []):
            ci = r["ci"]
            ci_txt = ("n too small for an interval" if r["faces"] < 20
                      else f"95% CI [{ci[0]:.4f}, {ci[1]:.4f}]")
            print(f"  {r['group']:<14} faces {r['faces']:>4} (pos {r['positives']:>4})  "
                  f"AUC {r['auc']:.4f}  {ci_txt}")

    for name in ("threshold_sweep_doc", "threshold_sweep_face"):
        rows = rep[name]
        b_y = best_threshold(rows, "youden")
        b_a = best_threshold(rows, "acc")
        print(f"\n{name}: best Youden t={b_y['threshold']:.2f} "
              f"(acc {b_y['acc']:.4f}, tpr {b_y['tpr']:.4f}, tnr {b_y['tnr']:.4f}) | "
              f"best acc t={b_a['threshold']:.2f} (acc {b_a['acc']:.4f}, F1 {b_a['f1']:.4f})")
        for t in (0.40, 0.42, 0.45, 0.50, 0.55, 0.60):
            near = min(rows, key=lambda r: abs(r["threshold"] - t))
            print(f"   t={near['threshold']:.2f}  acc {near['acc']:.4f}  "
                  f"F1 {near['f1']:.4f}  tpr {near['tpr']:.3f}  tnr {near['tnr']:.3f}")

    print("\nPASSTHROUGH FLOOR SWEEP (chosen margin)")
    print(f"{'frac':>6} {'doc AUC':>8} {'acc':>7} {'passthru':>9} {'no_face':>8} {'per_face':>9}")
    for r in rep["frac_sweep"]:
        print(f"{r['frac']:>6.2f} {r['auc']:>8.4f} {r['acc']:>7.4f} "
              f"{r['passthrough_share']:>9.0%} {r['no_face_share']:>8.0%} "
              f"{r['per_face_share']:>9.0%}")
    print("=" * 78)


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Margin sweep + FFIW evaluation (M5)")
    p.add_argument("--adapter_dir", default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--base_model", default="./models/InternVL3-2B")
    p.add_argument("--arch_version", default="v2")
    p.add_argument("--attn_implementation", default="sdpa")
    p.add_argument("--device", default=None)
    p.add_argument("--root", default="datasets/hydrafake/test/cd/FFIW")
    p.add_argument("--n_per_class", type=int, default=200)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--margins", type=float, nargs="+", default=list(MARGINS))
    p.add_argument("--ref_images", default="test_images",
                   help="local images measured alongside FFIW for the frac floor")
    p.add_argument("--min_face_px", type=int, default=96)
    p.add_argument("--threshold", type=float, default=0.50,
                   help="threshold used when reporting accuracy columns")
    p.add_argument("--detector_weights", default="./models/yolov8n-face.pt")
    p.add_argument("--cache", default="runs/calibration/ffiw_cache.jsonl")
    p.add_argument("--out", default="runs/calibration/ffiw_report.json")
    p.add_argument("--shipped_margin", type=float, default=0.30,
                   help="margin reported alongside the data-chosen one (default 0.30)")
    p.add_argument("--analyze_only", action="store_true")
    return p.parse_args()


def reference_images(ref_dir: str) -> List[Dict[str, Any]]:
    """Local images for the passthrough-floor evidence (no labels).

    test_images/real/ repeats several top-level images byte-for-byte, so this
    de-duplicates by content: counting the same photo twice would overstate how
    much evidence the floor actually has.
    """
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for path in sorted(Path(ref_dir).rglob("*")):
        if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        out.append({"path": path, "label": None, "set": "local",
                    "rel": path.as_posix()})
    return out


def main() -> int:
    args = parse_args()
    if not args.analyze_only and not (args.adapter_dir or args.checkpoint):
        sys.exit("ERROR: pass --adapter_dir or --checkpoint (or --analyze_only)")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    margins = [float(m) for m in args.margins]

    manifest = build_manifest(args.root, args.n_per_class, args.seed)
    refs = reference_images(args.ref_images) if args.ref_images else []
    print(f"manifest: {len(manifest)} FFIW frames, {len(refs)} local reference images")

    records: List[Dict[str, Any]] = []
    if args.analyze_only:
        with open(args.cache, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
    else:
        try:
            model, _, info = load_model(
                base_model=args.base_model, adapter_dir=args.adapter_dir,
                checkpoint=args.checkpoint, arch_version=args.arch_version,
                attn_implementation=args.attn_implementation, device=str(device))
        except AdapterLoadError as exc:
            sys.exit(f"ERROR: {exc}")
        print(f"weights: {info['source']}  device: {device}")
        transform = build_transform(448)
        detector = FaceDetector(args.detector_weights,
                                device=str(device) if device.type == "cuda" else None)
        everything = manifest + refs
        records = score_paths(everything, model, transform, detector, device,
                              margins, args.cache)

    labeled = [r for r in records if r.get("label") is not None]
    if not labeled:
        sys.exit("ERROR: no labeled records scored yet")

    rep = summarize(labeled, margins, args.min_face_px, args.threshold,
                    shipped_margin=args.shipped_margin)
    rep["local_reference"] = [
        {"image": r["rel"], "faces": len(r["faces"]),
         "fracs": [f["frac"] for f in r["faces"]],
         "whole_p": r["whole_p"],
         "crop_p": [e.get("p") for e in r["margins"][f"{rep['chosen_margin']}"]]}
        for r in records if r.get("set") == "local"]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    print_report(rep, margins)
    print(f"\nreport -> {args.out}")
    print("local reference images:")
    for r in rep["local_reference"]:
        print(f"  {r['image']:<34} faces={r['faces']} fracs={r['fracs']} "
              f"whole_p={r['whole_p']:.4f} crop_p={r['crop_p']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
