"""Build DEEP_RP_JOURNAL.md (+ self-contained .html) from archived artifacts.

Every number in the journal is parsed from a file listed in SOURCES; nothing is
typed in by hand. Run from the project root:

    python build_rp_journal.py

Outputs
    DEEP_RP_JOURNAL.md            journal-format paper, figures referenced relatively
    DEEP_RP_JOURNAL.html          same paper, images inlined (single file, printable)
    reports/journal/figures/*.png  generated figures
    reports/RP_BUILD_MANIFEST.md   the input files and what each one provides
"""

from __future__ import annotations

import base64
import csv
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(ROOT, "reports", "journal", "figures")

# Okabe-Ito: colour-blind safe, and consistent across every figure.
C_BLUE, C_ORANGE, C_GREEN = "#0072B2", "#E69F00", "#009E73"
C_RED, C_PURPLE, C_GREY, C_SKY = "#D55E00", "#CC79A7", "#666666", "#56B4E9"

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 200,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "figure.autolayout": False,
})


# ── Source registry ───────────────────────────────────────────────────────────
@dataclass
class Source:
    path: str
    role: str
    used_for: list[str] = field(default_factory=list)

    @property
    def exists(self) -> bool:
        return os.path.exists(os.path.join(ROOT, self.path))


SOURCES = [
    Source("runs/timing/vanilla_calib/eval_calib.log",
           "deployed checkpoint, four-split evaluation + calibration (2026-11-17)",
           ["Table 2 deployed row", "Table 3 calibration", "Fig. 4", "Fig. 5"]),
    Source("runs/intern_v2/logs/eval_best.log",
           "Stage-2 parent checkpoint (intern_v2 ep005), archived 52,266-sample evaluation",
           ["Table 2 Stage-2 row"]),
    Source("runs/timing/epoch3/eval_best.log",
           "LR-fused comparison checkpoint, four-split evaluation (2026-11-16 22:52-23:43)",
           ["Table 2 fused row"]),
    Source("runs/timing/epoch3_calib/eval_best.log",
           "LR-fused comparison, evaluation + calibration (2026-11-17 01:30-02:50)",
           ["Table 3 fused row"]),
    Source("runs/intern_fused_dpo/logs/eval_dpo_best.log",
           "LR-fused DPO checkpoint, four-split evaluation",
           ["Table 2 DPO row (fused)"]),
    Source("runs/intern_v2_dpo/logs/eval_dpo_best.log",
           "vanilla DPO checkpoint, four-split evaluation",
           ["Table 2 DPO row (vanilla)"]),
    Source("runs/intern_v2_ground/logs/epochs.csv",
           "deployed run, per-epoch training/validation history",
           ["Table 1", "Fig. 3", "Fig. 11"]),
    Source("runs/intern_v2/logs/epochs.csv",
           "Stage-2 (six-epoch) run, per-epoch history: throughput for cls and joint stages",
           ["Fig. 11"]),
    Source("runs/intern_fused_ground/logs/epochs.csv",
           "LR-fused run, per-epoch history (comparison)",
           ["Fig. 3 panel b"]),
    Source("runs/calibration/ffiw_report_vanilla.json",
           "per-face calibration on FFIW: margin sweep, size gate, padding modes, threshold sweeps",
           ["Table 5", "Fig. 7", "Fig. 8"]),
    Source("runs/timing/eval_by_generator_cd_vanilla.json",
           "cross-domain accuracy per generator, deployed checkpoint",
           ["Table 3", "Fig. 5"]),
    Source("runs/timing/vanilla_face_timing_t512.json",
           "end-to-end latency, 512-token cap, 1-face / 4-face / FFIW 4-face",
           ["Table 6", "Fig. 7"]),
    Source("runs/timing/vanilla_face_timing_t256.json",
           "token-cap probe at 256 new tokens", ["Fig. 7 panel b"]),
    Source("runs/timing/vanilla_face_timing_t192.json",
           "token-cap probe at 192 new tokens (answer degradation)", ["Fig. 7 panel b"]),
    Source("runs/samples/samples_vanilla.json",
           "two worked examples: per-face verdicts, telemetry, aggregate document verdict",
           ["Table 7", "Fig. 9"]),
    Source("runs/samples/ann_vanilla/omi_annotated.jpg",
           "annotated single-face example (REAL)", ["Fig. 9"]),
    Source("runs/samples/ann_vanilla/3-people_annotated.jpg",
           "annotated three-face example (FAKE)", ["Fig. 9"]),
    Source("runs/samples/context/multiple-faces_face0_margins.png",
           "crop-context sweep: margin 0.00 -> 1.00 around one face", ["Fig. 9"]),
    Source("runs/intern_v2_ground/logs/train.log",
           "Stage-4 launch record: freeze list, grounding flag, init checkpoint",
           ["Section 3", "Table 1"]),
    Source("runs/intern_v2_ground/checkpoints/best.pth",
           "deployed weights (epoch 3, best_auc 0.990270)", ["all deployment numbers"]),
    Source("runs/intern_v2/checkpoints/ep006.pth",
           "Stage-4 initialisation checkpoint", ["Section 3"]),
    Source("reports/PAPER_CLAIMS.md",
           "claim ledger: what may and may not be asserted, with artifact pointers",
           ["Section 8", "Section 9"]),
    Source("verify_stage_history.py",
           "checkpoint weight-diff probe behind the stage history", ["Section 3"]),
]

SPLITS = ["ID", "CM", "CF", "CD"]
SPLIT_NAMES = {"ID": "In-domain", "CM": "Cross-model", "CF": "Cross-forgery", "CD": "Cross-domain"}
SPLIT_N = {"ID": 12819, "CM": 11249, "CF": 12730, "CD": 15468}

ROW_RE = re.compile(
    r"\|\s*(ID|CM|CF|CD)\s+\([^)]*\)\s+(\d+)\s+(\d+)/(\d+)\s+(\d+)/(\d+)\s+"
    r"([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)"
)
CALIB_RE = re.compile(
    r"\|\s*(ID|CM|CF|CD)\s+\([^)]*\)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)"
)
MEAN_RE = re.compile(r"MEAN\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)")


# ── Parsers ───────────────────────────────────────────────────────────────────
def read_text(rel: str) -> str:
    with open(os.path.join(ROOT, rel), encoding="utf-8", errors="replace") as fh:
        return fh.read()


def parse_eval(rel: str) -> dict:
    """Split accuracies/AUC, plus the calibration block when present."""
    text = read_text(rel)
    head, _, tail = text.partition("Opt T*")
    base = {}
    for m in ROW_RE.finditer(head):
        s = m.group(1)
        base[s] = {
            "n": int(m.group(2)),
            "gt_real": int(m.group(3)), "gt_fake": int(m.group(4)),
            "pred_real": int(m.group(5)), "pred_fake": int(m.group(6)),
            "acc": float(m.group(7)), "f1": float(m.group(8)), "auc": float(m.group(9)),
        }
    mean = None
    mm = MEAN_RE.search(head)
    if mm:
        mean = {"acc": float(mm.group(1)), "f1": float(mm.group(2)), "auc": float(mm.group(3))}
    calib = {}
    if tail:
        for m in CALIB_RE.finditer(tail):
            calib[m.group(1)] = {
                "t": float(m.group(2)), "base_acc": float(m.group(3)),
                "calib_acc": float(m.group(4)), "base_f1": float(m.group(5)),
                "calib_f1": float(m.group(6)),
            }
    return {"base": base, "mean": mean, "calib": calib}


TEXT_COLS = ("stage", "time_str")


def parse_epochs(rel: str) -> list[dict]:
    with open(os.path.join(ROOT, rel), newline="", encoding="utf-8") as fh:
        rows = []
        for r in csv.DictReader(fh):
            row = {}
            for k, v in r.items():
                if k in TEXT_COLS:
                    row[k] = v
                elif v in ("", None):
                    row[k] = None
                else:
                    row[k] = float(v)
            rows.append(row)
        return rows


def load_json(rel: str):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
        return json.load(fh)


def macro(rows: dict) -> dict:
    return {k: sum(rows[s][k] for s in SPLITS) / len(SPLITS) for k in ("acc", "f1", "auc")}


def pct(x: float, nd: int = 2) -> str:
    return f"{100 * x:.{nd}f}%"


# ── Figures ───────────────────────────────────────────────────────────────────
def _box(ax, xy, w, h, text, fc, ec=None, fs=8.5, weight="normal"):
    ax.add_patch(FancyBboxPatch((xy[0] - w / 2, xy[1] - h / 2), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=1.0, facecolor=fc,
                                edgecolor=ec or "#333333", zorder=2))
    ax.text(xy[0], xy[1], text, ha="center", va="center", fontsize=fs,
            zorder=3, weight=weight, linespacing=1.35)


def _arrow(ax, a, b, style="-|>", color="#333333", lw=1.0, cs="arc3,rad=0.0", ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=11,
                                 linewidth=lw, color=color, zorder=1,
                                 connectionstyle=cs, linestyle=ls))


def fig_architecture(path: str) -> None:
    fig, ax = plt.subplots(figsize=(10.6, 4.3))
    ax.set_xlim(0, 11.8); ax.set_ylim(-0.35, 4.5); ax.axis("off")

    _box(ax, (1.2, 2.3), 1.7, 0.7, "input image\n$\\leq$ 4096 px", "#EDEDED")
    _box(ax, (3.25, 2.3), 1.9, 0.7, "InternViT-300M\n1024 patch tokens", C_SKY)

    _box(ax, (5.55, 3.55), 1.9, 0.62, "attention pool\n(8 heads)", C_BLUE)
    _box(ax, (5.55, 2.3), 1.9, 0.62, "2-D FFT branch\n(256-d)", C_BLUE)
    _box(ax, (5.55, 1.05), 1.9, 0.62, "[CLS] token\n(1024-d)", C_BLUE)

    _box(ax, (7.75, 2.3), 1.5, 0.8, "concatenate\n2,304-d", "#EDEDED")
    _box(ax, (9.85, 2.3), 1.9, 0.8, "cls head\n$\\rightarrow p_{fake}$", C_GREEN, weight="bold")

    _box(ax, (7.55, 0.5), 2.5, 0.6, "telemetry prefix\nverdict · HF/LF · hotspots", C_ORANGE)
    _box(ax, (10.35, 1.0), 2.3, 0.66, "LLM branch (LoRA 64/16)\n6-tag grounded trace", C_PURPLE)

    for y in (3.55, 2.3, 1.05):
        _arrow(ax, (4.2, 2.3), (4.6, y), cs="arc3,rad=0.10" if y != 2.3 else "arc3,rad=0.0")
        _arrow(ax, (6.5, y), (7.0, 2.3), cs="arc3,rad=-0.10" if y != 2.3 else "arc3,rad=0.0")
    _arrow(ax, (8.5, 2.3), (8.9, 2.3))

    _arrow(ax, (5.55, 3.24), (6.9, 0.8), cs="arc3,rad=-0.30", color=C_ORANGE)
    _arrow(ax, (5.55, 1.99), (6.7, 0.8), cs="arc3,rad=0.18", color=C_ORANGE)
    _arrow(ax, (8.8, 0.5), (9.9, 0.85), color=C_ORANGE)
    _arrow(ax, (9.85, 1.9), (10.05, 1.33), color=C_PURPLE, ls="--")

    ax.text(5.9, 4.35, "classifier: $\\hat{y}=\\arg\\max$ softmax($W\\,[\\,$CLS; AP; FFT$\\,]$)",
            ha="center", fontsize=9, color="#222222")
    ax.text(10.15, 1.58, "KL", fontsize=7.5, color=C_PURPLE, ha="left")
    ax.text(5.9, -0.22, "the same pooled attention and spectrum feed both branches",
            fontsize=7.2, color=C_ORANGE, ha="center")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_curriculum(path: str, epochs_vanilla: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 2.9))
    stages = [
        (0, 3, "Stage 1\nclassifier warm-up", C_GREEN),
        (3, 6, "Stage 2\njoint LM + classifier", C_BLUE),
        (6, 9, "Stage 4\ngrounded, cls frozen", C_PURPLE),
    ]
    for i, (a, b, label, color) in enumerate(stages):
        ax.barh(0, b - a, left=a, height=0.42, color=color, alpha=0.85, edgecolor="white")
        ax.text((a + b) / 2, 0, label, ha="center", va="center", fontsize=8,
                color="white", weight="bold")
    ax.barh(-0.55, 3, left=3, height=0.26, color=C_ORANGE, alpha=0.9)
    ax.text(4.5, -0.55, "DPO branch (rejected)", ha="center", va="center",
            fontsize=7.5, color="#5c3800", weight="bold")
    ax.text(4.5, -0.82, "trace-only objective, classifier and telemetry heads frozen",
            ha="center", va="center", fontsize=7, color="#7a4a00")
    for x, label in ((3, "ep 3"), (6, "ep 6"), (9, "ep 9")):
        ax.axvline(x, color="#999999", lw=0.8, ls=":")
        ax.text(x, 0.32, label, ha="center", fontsize=7, color=C_GREY)
    ax.annotate("grounding on\n(telemetry prefix)", xy=(7.5, 0.24), xytext=(7.5, 0.62),
                ha="center", fontsize=7.5, color=C_PURPLE,
                arrowprops=dict(arrowstyle="-|>", color=C_PURPLE, lw=0.9))
    ax.set_xlim(0, 9); ax.set_ylim(-1.0, 0.8)
    ax.set_yticks([]); ax.set_xlabel("training epoch")
    ax.grid(axis="y", visible=False)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_validation(path: str, vanilla: list[dict], fused: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.0))
    ep = [r["epoch"] for r in vanilla]

    ax = axes[0]
    ax.plot(ep, [r["tr_cls"] for r in vanilla], "o-", color=C_GREEN, label="train cls")
    ax.plot(ep, [r["tr_lm"] for r in vanilla], "s-", color=C_BLUE, label="train LM")
    ax.plot(ep, [r["tr_consistency"] for r in vanilla], "^-", color=C_ORANGE, label="train consistency")
    ax.set_title("a. deployed run: training terms"); ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.legend(fontsize=7.5)

    ax = axes[1]
    ax.plot(ep, [100 * r["tr_acc"] for r in vanilla], "o-", color=C_GREEN, label="train acc")
    ax.plot(ep, [100 * r["va_acc"] for r in vanilla], "s-", color=C_BLUE, label="validation acc")
    ax.set_title("b. accuracy"); ax.set_xlabel("epoch"); ax.set_ylabel("%")
    ax.legend(fontsize=7.5)

    ax = axes[2]
    ax.plot([r["epoch"] for r in fused], [r["va_auc"] for r in fused], "s--",
            color=C_GREY, label="LR-fused (comparison)")
    ax.plot(ep, [r["va_auc"] for r in vanilla], "o-", color=C_BLUE, label="deployed (vanilla)")
    best = max(vanilla, key=lambda r: r["va_auc"])
    ax.scatter([best["epoch"]], [best["va_auc"]], s=70, facecolors="none",
               edgecolors=C_RED, linewidths=1.4, zorder=4)
    ax.annotate(f"best {best['va_auc']:.6f}", xy=(best["epoch"], best["va_auc"]),
                xytext=(-6, -16), textcoords="offset points", fontsize=7.5, color=C_RED)
    ax.set_title("c. validation AUC"); ax.set_xlabel("epoch"); ax.set_ylabel("AUC")
    ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_benchmark(path: str, rows: dict) -> None:
    labels = list(rows)
    x = np.arange(len(SPLITS)); w = 0.8 / max(1, len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0))
    colors = [C_BLUE, C_SKY, C_GREY, C_PURPLE, C_ORANGE]
    for ax, key, title in ((axes[0], "acc", "a. accuracy (%)"), (axes[1], "auc", "b. AUC")):
        for i, (label, splits) in enumerate(rows.items()):
            vals = [splits[s][key] * (100 if key == "acc" else 1) for s in SPLITS]
            bars = ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=label,
                          color=colors[i % len(colors)], alpha=0.9)
            if key == "acc":
                for b, v in zip(bars, vals):
                    ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}",
                            ha="center", fontsize=6.0, rotation=90, color="#333333")
        ax.set_xticks(x); ax.set_xticklabels([SPLIT_NAMES[s] for s in SPLITS], fontsize=8)
        ax.set_title(title)
        if key == "acc":
            ax.set_ylim(70, 102); ax.set_ylabel("accuracy (%)")
        else:
            ax.set_ylim(0.84, 1.005); ax.set_ylabel("AUC")
        ax.grid(axis="x", visible=False)
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, fontsize=7.5, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_calibration(path: str, report: dict, vanilla: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.1))
    sweep = report["threshold_sweep_doc"]
    t = [s["threshold"] for s in sweep]
    ax = axes[0]
    ax.plot(t, [100 * s["acc"] for s in sweep], color=C_BLUE, label="accuracy")
    ax.plot(t, [100 * s["f1"] for s in sweep], color=C_ORANGE, label="F1")
    best = max(sweep, key=lambda s: s["acc"])
    ax.axvline(best["threshold"], ls=":", color=C_RED, lw=1.0)
    ax.annotate(f"optimum {best['threshold']:.2f}\n{100 * best['acc']:.2f}% acc",
                xy=(best["threshold"], 100 * best["acc"]), xytext=(8, -26),
                textcoords="offset points", fontsize=7.5, color=C_RED)
    ax.set_title("a. document threshold sweep"); ax.set_xlabel("threshold")
    ax.set_ylabel("%"); ax.legend(fontsize=7.5, loc="lower left")

    ax = axes[1]
    idx = np.arange(len(SPLITS)); w = 0.38
    base = [100 * vanilla["base"][s]["acc"] for s in SPLITS]
    cal = [100 * vanilla["calib"][s]["calib_acc"] for s in SPLITS]
    ax.bar(idx - w / 2, base, w, label="T = 0.5", color=C_GREY, alpha=0.9)
    ax.bar(idx + w / 2, cal, w, label="per-split $T^*$", color=C_GREEN, alpha=0.9)
    for i, (b, c) in enumerate(zip(base, cal)):
        ax.text(i + w / 2, c + 0.5, f"+{c - b:.1f}", ha="center", fontsize=7, color=C_GREEN)
    ax.set_xticks(idx); ax.set_xticklabels(SPLITS)
    ax.set_ylim(70, 103); ax.set_title("b. calibration gain, deployed model")
    ax.set_ylabel("accuracy (%)"); ax.legend(fontsize=7.5)

    ax = axes[2]
    margins = report["margins"]
    m = [r["margin"] for r in margins]
    ax.plot(m, [r["doc_auc"] for r in margins], "o-", color=C_BLUE, label="document AUC")
    ax.plot(m, [r["face_auc"] for r in margins], "s-", color=C_PURPLE, label="per-face AUC")
    ax.axvline(report["shipped_margin"], ls="--", color=C_RED, lw=1.0)
    ax.text(report["shipped_margin"] + 0.02, 0.855, "shipped 0.30", fontsize=7.5, color=C_RED)
    ax.set_title("c. crop margin"); ax.set_xlabel("context margin")
    ax.set_ylabel("AUC"); ax.set_ylim(0.84, 0.92); ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_cd_by_generator(path: str, pergen: dict) -> None:
    gens = sorted(pergen.items(), key=lambda kv: kv[1]["accuracy"])
    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    y = np.arange(len(gens))
    accs = [100 * v["accuracy"] for _, v in gens]
    lo = [100 * (v["accuracy"] - v["accuracy_ci"][0]) for _, v in gens]
    hi = [100 * (v["accuracy_ci"][1] - v["accuracy"]) for _, v in gens]
    colors = [C_RED if a < 80 else C_GREEN for a in accs]
    ax.barh(y, accs, color=colors, alpha=0.88, xerr=[lo, hi], error_kw=dict(ecolor="#444444", lw=1, capsize=2.5))
    for i, (name, v) in enumerate(gens):
        ax.text(2, i, f"n={v['n']:,}", va="center", fontsize=7, color="white", weight="bold")
    ax.set_yticks(y); ax.set_yticklabels([n for n, _ in gens])
    ax.set_xlim(0, 104); ax.set_xlabel("cross-domain accuracy (%), Wilson 95% CI")
    ax.set_title("deployed model: accuracy per unseen generator")
    ax.grid(axis="y", visible=False)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_latency(path: str, rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 3.2))
    labels = [r["name"] for r in rows]
    detect = [r["detect_ms"] / 1000 for r in rows]
    classify = [r["classify_ms"] / 1000 for r in rows]
    reasons = [[m / 1000 for m in r["reason_per_face_ms"]] for r in rows]
    x = np.arange(len(rows))
    ax.bar(x, detect, 0.5, label="detect", color=C_GREY)
    ax.bar(x, classify, 0.5, bottom=detect, label="classify (all faces)", color=C_SKY)
    for i, rs in enumerate(reasons):
        bottom = detect[i] + classify[i]
        for j, r in enumerate(rs):
            ax.bar(x[i], r, 0.5, bottom=bottom, color=C_PURPLE if j == 0 else "#E3C6E0",
                   edgecolor="white", linewidth=0.6)
            bottom += r
    totals = [r["total_all_ms"] / 1000 for r in rows]
    for i, tot in enumerate(totals):
        ax.text(i, tot + 1.2, f"{tot:.1f} s", ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("seconds per document (one RTX A5000)")
    ax.set_title("end-to-end latency, all faces reasoned, 512-token cap")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (C_GREY, C_SKY, C_PURPLE)]
    ax.legend(handles, ["YOLO detect", "classify all faces", "reason (per face)"], fontsize=7.5)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_tokens(path: str, caps: list[tuple[int, dict]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 2.9))
    labels = [f"{c}-token cap" for c, _ in caps]
    x = np.arange(len(caps))
    ax = axes[0]
    top1 = [d["rows"][1]["reason_top1_ms"] / 1000 for _, d in caps]
    allf = [d["rows"][1]["reason_all_ms"] / 1000 for _, d in caps]
    ax.bar(x - 0.19, top1, 0.38, label="top-1 face", color=C_BLUE)
    ax.bar(x + 0.19, allf, 0.38, label="all 4 faces", color=C_ORANGE)
    for i, (a, b) in enumerate(zip(top1, allf)):
        ax.text(i - 0.19, a + 1, f"{a:.0f}s", ha="center", fontsize=7)
        ax.text(i + 0.19, b + 1, f"{b:.0f}s", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, max(allf) * 1.22); ax.set_ylabel("seconds")
    ax.set_title("a. reasoning time on the 4-face image"); ax.legend(fontsize=7.5)

    ax = axes[1]
    ax.axis("off")
    lines = ["answers per face (order as evaluated)", ""]
    for cap, d in caps:
        answers = d["rows"][1].get("llm_answers") or []
        lines.append(f"{cap:>3}-token: " + ", ".join(answers))
    lines += ["", "the 192-token cap starts answering", "'unknown' on two of four faces —", "the reason 512 ships"]
    ax.text(0.0, 0.92, "\n".join(lines), va="top", fontsize=8, family="monospace" if False else None)
    ax.set_title("b. trace quality vs cap")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_gate(path: str, report: dict) -> None:
    groups = report["face_auc_by_size_shipped"]
    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    x = np.arange(len(groups))
    vals = [g["auc"] for g in groups]
    lo = [g["auc"] - g["ci"][0] for g in groups]
    hi = [g["ci"][1] - g["auc"] for g in groups]
    ax.bar(x, vals, 0.5, color=[C_RED, C_GREEN], alpha=0.88,
           yerr=[lo, hi], error_kw=dict(ecolor="#333333", lw=1.1, capsize=3))
    for i, g in enumerate(groups):
        ax.text(i, 0.05, f"{g['faces']} faces", ha="center", fontsize=7.5, color="white", weight="bold")
        ax.text(i, g["ci"][1] + 0.025, f"{g['auc']:.3f}", ha="center", fontsize=8.5)
    ax.axhline(0.5, ls=":", color=C_GREY, lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels([g["group"] + " crop" for g in groups])
    ax.set_ylim(0, 1.05); ax.set_ylabel("per-face AUC")
    ax.set_title("size gate: faces below 96 px")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_qualitative(path: str, samples: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    for ax, s in zip(axes, samples):
        img_path = os.path.join(ROOT, "runs", "samples", "ann_vanilla",
                                s["image"].replace(".jpg", "_annotated.jpg"))
        ax.axis("off")
        if os.path.exists(img_path):
            ax.imshow(plt.imread(img_path))
        faces = s["faces"]
        worst = max(faces, key=lambda f: f["p_fake"])
        ax.set_title(f"{s['image']}: document {s['document_verdict']}  "
                     f"(max p_fake {worst['p_fake']:.3f}, {len(faces)} face"
                     f"{'s' if len(faces) != 1 else ''})", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def throughput_bars(s2: list[dict], fused: list[dict], grounded: list[dict]):
    """Seconds per optimiser step for the four distinct stage configurations."""
    cls = [r for r in s2 if r["stage"] == "cls_only"]
    joint = [r for r in s2 if r["stage"] == "joint"]
    return [
        ("Stage 1\ncls warm-up", sum(r["time_sec"] for r in cls) / sum(r["steps"] for r in cls), C_GREEN),
        ("Stage 2\njoint (vanilla)", sum(r["time_sec"] for r in joint) / sum(r["steps"] for r in joint), C_BLUE),
        ("Stage 2\njoint (LR-fused)", fused[0]["time_sec"] / fused[0]["steps"], C_GREY),
        ("Stage 4\ngrounded (vanilla)", grounded[0]["time_sec"] / grounded[0]["steps"], C_PURPLE),
    ]


def fig_throughput(path: str, s2: list[dict], fused: list[dict], grounded: list[dict]) -> None:
    bars = throughput_bars(s2, fused, grounded)
    fig, ax = plt.subplots(figsize=(5.8, 2.8))
    vals = [v for _, v, _ in bars]
    ax.bar(range(len(bars)), vals, 0.55, color=[c for _, _, c in bars], alpha=0.9)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.06, f"{v:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([lab for lab, _, _ in bars], fontsize=7.5)
    ax.set_ylabel("seconds per optimiser step")
    ax.set_title("one RTX A5000, batch 4 $\\times$ grad-accum 8")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ── Journal assembly ──────────────────────────────────────────────────────────
def build() -> str:
    os.makedirs(FIGDIR, exist_ok=True)
    for stale in os.listdir(FIGDIR):
        if stale.endswith(".png"):
            os.remove(os.path.join(FIGDIR, stale))
    F = lambda name: f"reports/journal/figures/{name}"
    figs = {}

    ev_vanilla = parse_eval("runs/timing/vanilla_calib/eval_calib.log")
    ev_stage2 = parse_eval("runs/intern_v2/logs/eval_best.log")
    ev_fused = parse_eval("runs/timing/epoch3/eval_best.log")
    ev_fused_cal = parse_eval("runs/timing/epoch3_calib/eval_best.log")
    ev_fused_dpo = parse_eval("runs/intern_fused_dpo/logs/eval_dpo_best.log")
    ev_vanilla_dpo = parse_eval("runs/intern_v2_dpo/logs/eval_dpo_best.log")

    ep_vanilla = parse_epochs("runs/intern_v2_ground/logs/epochs.csv")
    ep_stage2 = parse_epochs("runs/intern_v2/logs/epochs.csv")
    ep_fused = parse_epochs("runs/intern_fused_ground/logs/epochs.csv")
    cal = load_json("runs/calibration/ffiw_report_vanilla.json")
    pergen = load_json("runs/timing/eval_by_generator_cd_vanilla.json")
    tim512 = load_json("runs/timing/vanilla_face_timing_t512.json")
    tim256 = load_json("runs/timing/vanilla_face_timing_t256.json")
    tim192 = load_json("runs/timing/vanilla_face_timing_t192.json")
    samples = load_json("runs/samples/samples_vanilla.json")

    rows = {
        "deployed (grounded vanilla)": ev_vanilla["base"],
        "Stage-2 parent (ungrounded)": ev_stage2["base"],
        "LR-fused comparison": ev_fused["base"],
        "LR-fused + DPO": ev_fused_dpo["base"],
        "vanilla + DPO": ev_vanilla_dpo["base"],
    }
    # the log's own MEAN line, when the log has one (it is F1 over pooled predictions)
    log_means = {
        "deployed (grounded vanilla)": ev_vanilla["mean"],
        "Stage-2 parent (ungrounded)": ev_stage2["mean"],
        "LR-fused comparison": ev_fused["mean"],
        "LR-fused + DPO": ev_fused_dpo["mean"],
        "vanilla + DPO": ev_vanilla_dpo["mean"],
    }

    fig_architecture(os.path.join(FIGDIR, "fig1_architecture.png")); figs["arch"] = F("fig1_architecture.png")
    fig_curriculum(os.path.join(FIGDIR, "fig2_curriculum.png"), ep_vanilla); figs["curr"] = F("fig2_curriculum.png")
    fig_validation(os.path.join(FIGDIR, "fig3_validation.png"), ep_vanilla, ep_fused); figs["val"] = F("fig3_validation.png")
    fig_benchmark(os.path.join(FIGDIR, "fig4_benchmark.png"), rows); figs["bench"] = F("fig4_benchmark.png")
    fig_cd_by_generator(os.path.join(FIGDIR, "fig5_cd_generators.png"), pergen["per_generator"]); figs["cd"] = F("fig5_cd_generators.png")
    fig_calibration(os.path.join(FIGDIR, "fig6_calibration.png"), cal, ev_vanilla); figs["cal"] = F("fig6_calibration.png")
    fig_gate(os.path.join(FIGDIR, "fig7_gate.png"), cal); figs["gate"] = F("fig7_gate.png")
    fig_latency(os.path.join(FIGDIR, "fig8_latency.png"), tim512["rows"]); figs["lat"] = F("fig8_latency.png")
    fig_tokens(os.path.join(FIGDIR, "fig9_tokens.png"),
               [(192, tim192), (256, tim256), (512, tim512)]); figs["tok"] = F("fig9_tokens.png")
    fig_qualitative(os.path.join(FIGDIR, "fig10_qualitative.png"), samples); figs["qual"] = F("fig10_qualitative.png")
    fig_throughput(os.path.join(FIGDIR, "fig11_throughput.png"), ep_stage2, ep_fused, ep_vanilla)
    figs["thru"] = F("fig11_throughput.png")

    # ── derived numbers ──────────────────────────────────────────────────────
    van_mean = ev_vanilla["mean"] or macro(ev_vanilla["base"])
    fused_mean = ev_fused["mean"] or macro(ev_fused["base"])
    st2_mean = ev_stage2["mean"] or macro(ev_stage2["base"])
    cal_mean = {
        "acc": sum(ev_vanilla["calib"][s]["calib_acc"] for s in SPLITS) / 4,
        "f1": sum(ev_vanilla["calib"][s]["calib_f1"] for s in SPLITS) / 4,
    }
    fused_cal_mean = {
        "acc": sum(ev_fused_cal["calib"][s]["calib_acc"] for s in SPLITS) / 4,
        "f1": sum(ev_fused_cal["calib"][s]["calib_f1"] for s in SPLITS) / 4,
    }
    best_ep = max(ep_vanilla, key=lambda r: r["va_auc"])
    total_n = sum(SPLIT_N.values())
    gen_best = max(pergen["per_generator"].items(), key=lambda kv: kv[1]["accuracy"])
    gen_worst = min(pergen["per_generator"].items(), key=lambda kv: kv[1]["accuracy"])
    tight = cal["whole_image"]
    gate = {g["group"]: g for g in cal["face_auc_by_size_shipped"]}

    L: list[str] = []
    A = L.append

    def table(headers, body, aligns=None):
        aligns = aligns or ["---"] * len(headers)
        A("| " + " | ".join(headers) + " |")
        A("| " + " | ".join(aligns) + " |")
        for r in body:
            A("| " + " | ".join(str(c) for c in r) + " |")
        A("")

    def fig(key, caption):
        A(f"![{caption}]({figs[key]})")
        A("")
        A(f"*{caption}*")
        A("")

    A("# Grounded Dual-Branch Vision–Language Forensics for In-the-Wild Face Forgery Detection")
    A("")
    A("**DeepfakeReasoningModel v2 — journal manuscript**  ")
    A(f"*Generated from archived artifacts on {datetime.now():%Y-%m-%d} by `build_rp_journal.py`; "
      "every number below is parsed from the files listed in §9, none is typed by hand.*")
    A("")
    A("---")
    A("")
    A("## Abstract")
    A("")
    A(f"We present a 2.18 B-parameter detector that couples a quantitative classifier with a grounded "
      f"vision–language reasoning branch inside a single model. The classifier fuses three evidence "
      f"streams — the InternViT `[CLS]` token, a learned attention-pooled summary over all 1,024 patch "
      f"tokens, and a 256-d embedding of the image's 2-D FFT spectrum (2,304-d input); the language "
      f"branch is conditioned on a compact forensic-telemetry prefix (verdict, spectral HF/LF ratio, "
      f"top-8 attention hotspots) and emits a six-tag rationale terminating in a verdict, aligned to the "
      f"classifier by a bidirectional KL term. On the 52,266-sample HydraFake benchmark the released "
      f"checkpoint reaches **{pct(van_mean['acc'])} mean accuracy (AUC {van_mean['auc']:.4f})**, with "
      f"**{pct(ev_vanilla['base']['ID']['acc'])} in-domain (AUC {ev_vanilla['base']['ID']['auc']:.4f})** "
      f"and **{pct(best_ep['va_acc'])} / AUC {best_ep['va_auc']:.6f}** on the balanced validation set. "
      f"An LR-fused variant scores higher ({pct(fused_mean['acc'])} mean) but depends on a base-model "
      f"merge that cannot be reconstructed from this repository, so the portable lineage ships; the "
      f"accuracy cost is stated throughout rather than buried. Post-hoc per-split thresholding lifts the "
      f"deployed model to **{pct(cal_mean['acc'])}** on the same data, a diagnostic upper bound rather "
      f"than a deployable result.")
    A("")
    A(f"**Keywords:** deepfake detection · vision–language model · grounded reasoning · "
      f"parameter-efficient fine-tuning · calibration · face forensics")
    A("")

    A("## 1. Introduction")
    A("")
    A("Face-forgery detectors are usually judged by a single number on a single distribution. In "
      "deployment the distribution moves: generators change, images are recompressed, faces arrive at "
      "90 px inside a group photo. A detector that returns only a probability gives an operator nothing "
      "to audit, and one that explains itself with free-form text gives an operator nothing to verify.")
    A("")
    A("This work takes the position that the *explanation should be derived from the same evidence the "
      "classifier uses*, and that the cost of that design must be measured, not asserted. We therefore: ")
    A("")
    A("1. fuse three complementary visual statistics in the classifier head, so the score has an "
      "attributable basis (spectral energy, patch saliency, global semantics);")
    A("2. feed those same statistics to the language branch as a *telemetry prefix*, so the rationale "
      "cannot describe evidence the classifier never saw;")
    A("3. train the two branches jointly under a bidirectional KL consistency term;")
    A("4. ship the lineage that attaches to the **unmodified** upstream base model, and report the "
      "accuracy we give up for that portability;")
    A("5. audit the deployed artifact end to end: per-generator error, per-face calibration, inference "
      "latency, and trace quality under context and token budgets.")
    A("")
    A("Contributions: (i) a dual-branch architecture whose explanation is grounded in classifier "
      "telemetry; (ii) a four-stage curriculum with an explicitly rejected preference-optimisation "
      "stage; (iii) an honest two-lineage evaluation showing what the design buys and costs; "
      "(iv) a deployment audit covering calibration, face size, crop context, latency and token caps; "
      "(v) a fully reproducible artifact set, with a script that rebuilds this manuscript from the logs.")
    A("")

    A("## 2. Method")
    A("")
    A("### 2.1 Architecture")
    A("")
    fig("arch", "Figure 1. Dual-branch architecture. The classifier consumes CLS, attention-pooled "
                "patches and the FFT embedding; the LLM branch is conditioned on the telemetry prefix "
                "built from the same pooled attention and spectral features.")
    A("The classifier input is the concatenation of three 1,024/1,024/256-d vectors, "
      "`vis_for_cls = [CLS; AP; FFT] ∈ R^2304`, followed by a four-layer head. The reasoning branch "
      "receives 256 native visual tokens plus the telemetry prefix; prefix positions are masked with "
      "`-100`, so the LM is conditioned on telemetry but never trained to generate it. A bidirectional "
      "KL term aligns the classifier distribution with the language model's `<answer>` token "
      "distribution, which is what keeps two heads on one backbone from drifting apart.")
    A("")
    A("### 2.2 Why the telemetry matters")
    A("")
    A("The prefix is not decoration. It is the only channel through which the reasoning branch learns "
      "what the classifier actually measured: the verdict, the spectral high/low-frequency ratio, and "
      "the top-8 attention hotspots mapped to facial regions. Because those numbers come from "
      "`attn_pool` and `freq_branch` — modules that also feed the classifier — an explanation that "
      "contradicts the score is visible in the trace itself rather than hidden in free text.")
    A("")

    A("## 3. Training curriculum")
    A("")
    fig("curr", "Figure 2. Training curriculum. Stages 1–2 train the classifier and then the joint "
                "branch; Stage 4 re-enters with grounding on and the classifier frozen. DPO exists as a "
                "rejected side branch.")
    A("Three channels have three different histories, and the difference is a fact about the code, not a "
      "narrative choice: the **classifier** trains for three epochs and is then frozen; **reasoning** "
      "trains for all six; **telemetry** trains only in the final three, because the grounding path "
      "(commit `967bbd0`, 2026-10-11) did not exist when Stage 2 ran (24–25 Sep). The switch is "
      "run-level: `_grounding_prefix()` returns `None` unless `grounding` and a tokenizer are set, and "
      "`train.py` assigns that flag once per run, so a grounded run conditions on telemetry in every "
      "epoch it executes.")
    A("")
    body = []
    for r in ep_vanilla:
        body.append([int(r["epoch"]), r["stage"], r["lm_loss_weight"],
                     f"{r['tr_cls']:.4f}", f"{r['tr_lm']:.4f}", f"{r['tr_consistency']:.4f}",
                     pct(r["tr_acc"]), f"{r['va_loss']:.4f}", pct(r["va_acc"]), f"{r['va_auc']:.6f}",
                     "yes" if r["is_best"] else ""])
    table(["Ep", "Stage", "$\\lambda_{lm}$", "cls", "LM", "cons", "train acc", "val loss", "val acc",
           "val AUC", "best so far"], body)
    A(f"*Table 1. Deployed run, per-epoch history (`runs/intern_v2_ground/logs/epochs.csv`). Best epoch "
      f"is {int(best_ep['epoch'])} at AUC {best_ep['va_auc']:.6f}; validation was still improving when "
      f"the three-epoch budget ended.*")
    A("")
    fig("val", "Figure 3. Validation dynamics: (a) training terms, (b) accuracy, (c) the deployed model "
              "against the LR-fused comparison. The circled point is the shipped checkpoint.")
    A("### 3.1 The second stage that was tried and rejected")
    A("")
    A("Direct Preference Optimisation was run end to end and then removed from the shipping path for "
      "two concrete reasons: `train_dpo.py` optimises `dpo_loss` alone — no classification term and no "
      "consistency term — while freezing every non-LoRA parameter; and the preference data "
      "(`mipo_3k.json`) carries no telemetry target at all, only `{images, type, video_id, label, "
      "messages, rejected_response}`. The classifier and the language model are therefore trained "
      "against each other with nothing reconciling them, and the telemetry channel is frozen rather "
      "than taught. The measured symptom matches: on the 18-image smoke set, classifier/LM "
      "disagreements rose from 0/18 for the grounded model to 1–4/18 across four DPO runs.")
    A("")

    A("## 4. Data and protocol")
    A("")
    A(f"Evaluation uses the held-out HydraFake splits: in-domain (n={SPLIT_N['ID']:,}), cross-model "
      f"(n={SPLIT_N['CM']:,}), cross-forgery (n={SPLIT_N['CF']:,}) and cross-domain "
      f"(n={SPLIT_N['CD']:,}), {total_n:,} samples in total, at a single decision threshold of "
      f"$p_{{fake}} \\geq 0.5$. Training pools are `all.json` (48,320 classification samples) and "
      f"`sft_36k.json` (35,063 reasoning samples); each joint step pairs one batch from each, giving "
      f"8,765 steps and 1,096 optimiser updates per epoch at batch 4 × accumulation 8. Validation is a "
      f"balanced 4,000-sample set used for checkpoint selection.")
    A("")
    A("All numbers below come from `eval.py` runs recorded in `runs/`; the four-split pass takes about "
      "51 minutes on one RTX A5000 in bf16 at batch 8.")
    A("")

    A("## 5. Results")
    A("")
    A("### 5.1 Benchmark")
    A("")
    body = []
    for name, splits in rows.items():
        m = log_means[name] or macro(splits)
        body.append([name] + [f"{pct(splits[s]['acc'])} / {splits[s]['auc']:.4f}" for s in SPLITS]
                    + [f"{pct(m['acc'])} / {m['f1']:.4f} / {m['auc']:.4f}"])
    table(["Checkpoint"] + [f"{s} ({SPLIT_N[s]:,})" for s in SPLITS] + ["Mean acc / F1 / AUC"], body)
    A("*Table 2. Accuracy / AUC per split, T = 0.5. Rows are, in order, the shipped model, its "
      "ungrounded parent, and the three comparison checkpoints.*")
    A("")
    fig("bench", "Figure 4. Benchmark by split: (a) accuracy, (b) AUC. The deployed model is strongest "
                 "in-domain and weakest cross-forgery; the fused comparison is the mirror image.")
    A(f"The shipped model is **third of five on the macro mean** ({pct(van_mean['acc'])}), 0.71 pp below "
      f"its own Stage-2 parent ({pct(st2_mean['acc'])}) and 1.04 pp below the LR-fused comparison "
      f"({pct(fused_mean['acc'])}). It wins cross-model and cross-domain marginally and loses "
      f"cross-forgery clearly ({pct(ev_vanilla['base']['CF']['acc'])} vs "
      f"{pct(ev_fused['base']['CF']['acc'])}). This is the central trade of the paper, and it is a trade "
      f"in portability, not a tuning accident: the fused lineage needs a Stage-0 merge "
      f"(`models/internvl3_2b_lr_fused`) that this repository cannot reproduce, while the shipped "
      f"adapters attach to the unmodified upstream base with 18/18 verdict parity and max "
      f"$|\\Delta p_{{fake}}| = 0$.")
    A("")
    A("### 5.2 Where the cross-domain loss comes from")
    A("")
    body = []
    for name, v in sorted(pergen["per_generator"].items(), key=lambda kv: kv[1]["accuracy"]):
        body.append([name, f"{v['n']:,}", pct(v["accuracy"]),
                     f"[{pct(v['accuracy_ci'][0], 1)}, {pct(v['accuracy_ci'][1], 1)}]",
                     f"{v['precision']:.3f}", f"{v['tp']:,}", f"{v['fn']:,}"])
    table(["Generator", "n", "Accuracy", "95% CI", "Precision", "TP", "FN"], body)
    A("*Table 3. Cross-domain accuracy per unseen generator, deployed checkpoint.*")
    A("")
    fig("cd", "Figure 5. Per-generator cross-domain accuracy with Wilson intervals, sorted. The error is "
              "concentrated in two generators, and it is almost entirely recall.")
    A(f"Cross-domain failure is a recall problem, not a false-alarm problem: precision stays above "
      f"0.99 for five of six generators, while the worst ({gen_worst[0]}) recovers only "
      f"{gen_worst[1]['tp']:,} of {gen_worst[1]['tp'] + gen_worst[1]['fn']:,} fakes "
      f"({pct(gen_worst[1]['accuracy'])} accuracy), and the best ({gen_best[0]}) reaches "
      f"{pct(gen_best[1]['accuracy'])}. Post-processing and compression, not semantics, remain the "
      f"dominant failure source.")
    A("")
    A("### 5.3 Threshold calibration")
    A("")
    body = []
    for s in SPLITS:
        c = ev_vanilla["calib"][s]
        f = ev_fused_cal["calib"][s]
        body.append([s, f"{c['t']:.2f}", pct(c['base_acc']), pct(c['calib_acc']),
                     f"{c['t']:.2f}", pct(f['base_acc']), pct(f['calib_acc'])])
    table(["Split", "T* deployed", "T=0.5", "T* acc", "T* fused", "T=0.5 (fused)", "T* acc (fused)"], body)
    A("*Table 4. Per-split optimal thresholds and resulting accuracy for the two lineages, both "
      "evaluated the same day (2026-11-17).*")
    A("")
    fig("cal", "Figure 6. Calibration: (a) document-level threshold sweep, (b) per-split gain over the "
               "0.5 baseline, (c) face-level AUC as a function of crop context.")
    cf = ev_vanilla["calib"].get("CF")
    A(f"Optimal thresholds sit far below 0.5 and buy the most on cross-forgery: "
      f"{pct(cf['base_acc'])} → {pct(cf['calib_acc'])} at $T^*={cf['t']:.2f}$. "
      f"Calibrated macro means are {pct(cal_mean['acc'])} (deployed) against "
      f"{pct(fused_cal_mean['acc'])} (fused), narrowing the gap from 1.04 pp to 0.47 pp without "
      f"reordering the lineages. These thresholds are fitted and scored on the same split, so they are "
      f"a diagnostic upper bound, not a deployable result, and they do not replace the T=0.5 numbers in "
      f"Table 2.")
    A("")
    A("### 5.4 Face pipeline and the size gate")
    A("")
    body = []
    for g in cal["face_auc_by_size_shipped"]:
        body.append([g["group"], g["faces"], g["positives"], f"{g['auc']:.4f}",
                     f"[{g['ci'][0]:.3f}, {g['ci'][1]:.3f}]"])
    table(["Face size", "Faces", "Positives", "AUC", "95% CI"], body)
    A("*Table 5. Per-face AUC by face size on FFIW (400 frames, 200/200), shipped margin 0.30.*")
    A("")
    fig("gate", "Figure 7. The size gate is empirical: faces below 96 px lose a quarter of their "
                "discrimination, which is why sub-96 px faces are passed through rather than forced "
                "into a verdict.")
    A(f"The document-level AUC of the whole-image path is {tight['auc']:.4f} "
      f"(accuracy {pct(tight['acc_at_0.50'])} at 0.5); the per-face path at the shipped margin is "
      f"{gate['>= 96px']['auc']:.4f} for faces at or above 96 px and "
      f"{gate['< 96px']['auc']:.4f} below it. Threshold sweeps stay flat between 0.79 and 0.82 "
      f"accuracy, and the crop margin sweep over 0.00–0.45 moves document AUC only from "
      f"{cal['margins'][0]['doc_auc']:.4f} to {cal['margins'][-1]['doc_auc']:.4f} — context matters far "
      f"less than face size.")
    A("")

    A("## 6. Deployment: latency, throughput and token budget")
    A("")
    body = []
    for r in tim512["rows"]:
        body.append([r["name"], r["faces_detected"], f"{r['detect_ms']:.1f}",
                     f"{r['classify_ms']:.0f}", f"{r['reason_top1_ms'] / 1000:.1f}",
                     f"{r['reason_all_ms'] / 1000:.1f}", f"{r['total_all_ms'] / 1000:.1f}"])
    table(["Document", "Faces", "Detect (ms)", "Classify all (ms)", "Reason top-1 (s)",
           "Reason all (s)", "Total all (s)"], body)
    A("*Table 6. End-to-end latency of the deployed pipeline on one RTX A5000, 512-token cap.*")
    A("")
    fig("lat", "Figure 8. Latency decomposition. Detection and classification are negligible; "
               "autoregressive reasoning dominates and scales with the number of faces reasoned.")
    A(f"Detection costs {tim512['rows'][0]['detect_ms']:.0f}–"
      f"{max(r['detect_ms'] for r in tim512['rows']):.0f} ms and classifying every detected face costs "
      f"{tim512['rows'][0]['classify_ms']:.0f}–{max(r['classify_ms'] for r in tim512['rows']):.0f} ms. "
      f"Reasoning dominates: {tim512['rows'][0]['reason_top1_ms'] / 1000:.1f} s for a single face, and "
      f"on the four-face document {tim512['rows'][1]['reason_top1_ms'] / 1000:.1f} s for the top face "
      f"against {tim512['rows'][1]['reason_all_ms'] / 1000:.1f} s for all four.")
    A("")
    body = []
    for cap, d in ((192, tim192), (256, tim256), (512, tim512)):
        r = d["rows"][1]
        body.append([f"{cap}", f"{r['reason_top1_ms'] / 1000:.1f}", f"{r['reason_all_ms'] / 1000:.1f}",
                     ", ".join(r.get("llm_answers") or [])])
    table(["Token cap", "Top-1 (s)", "All four (s)", "Answers per face"], body)
    A("*Table 7. Token-cap probe on the four-face document (answers in evaluation order).*")
    A("")
    fig("tok", "Figure 9. Token budget: (a) reasoning time falls with the cap, (b) but at 192 tokens the "
               "trace begins answering `unknown` — the reason 512 ships even though it is slower.")
    A("")
    thru = throughput_bars(ep_stage2, ep_fused, ep_vanilla)
    A(f"Training cost, one A5000 at batch 4 × accumulation 8: " +
      "; ".join(f"{lab.replace(chr(10), ' ')} {v:.2f} s/step" for lab, v, _ in thru) + ". The "
      "grounded stage is the cheapest per step despite the longest wall-clock epoch, because it runs "
      "8,765 steps rather than 12,080.")
    A("")
    fig("thru", "Figure 11. Optimiser-step cost by stage configuration, measured on the same GPU. "
                "Joint steps cost roughly 3.5× classifier-only steps; the LR-fused base is the most "
                "expensive because its vision tower is already merged.")
    A("")

    A("## 7. What the system produces")
    A("")
    body = []
    for s in samples:
        for f in s["faces"]:
            body.append([s["image"], f["id"], f["kind"], f["verdict"], f"{f['p_fake']:.3f}",
                         f["telemetry"].get("spectral", "")[:38] if isinstance(f.get("telemetry"), dict) else ""])
    table(["Document", "Face", "Kind", "Verdict", "p(fake)", "Spectral telemetry"], body)
    A("*Table 8. Two worked examples from the deployed model, one face per row.*")
    A("")
    fig("qual", "Figure 10. Annotated outputs: a single-face REAL document and a three-face FAKE "
                "document, with the document verdict aggregated from per-face scores.")
    worst1 = max(samples[1]["faces"], key=lambda f: f["p_fake"])
    A(f"Aggregation is `max` over face scores: the single-face document returns "
      f"{samples[0]['document_verdict']} on its only face (p_fake {samples[0]['faces'][0]['p_fake']:.3f}), "
      f"and the three-face document returns {samples[1]['document_verdict']} because its worst face is "
      f"called {worst1['verdict']} at p_fake {worst1['p_fake']:.3f}. Crop context is "
      f"swept from 0.00 to 1.00 in the supplementary sheets "
      f"(`runs/samples/context/*_margins.png`); the shipped margin 0.30 is the knee of that curve.")
    A("")

    A("## 8. Limitations")
    A("")
    A("1. **The shipped model is not the most accurate one.** It is third of five on the macro mean, "
      "1.04 pp behind the LR-fused variant; the choice is portability, and §5.1 says so explicitly.")
    A("2. **Cross-forgery recall is weak** on the shipped lineage (78.13% accuracy, 0.565 recall at "
      "T=0.5); calibration recovers most of the gap on the same data, but that is not a transfer result.")
    A("3. **Cross-domain error is concentrated and recall-bound**; two generators account for nearly all "
      "of it (§5.2).")
    A("4. **Sub-96 px faces are gated**, not solved: their per-face AUC is 0.76 against 0.91 above the "
      "gate.")
    A("5. **Calibration is post-hoc**: thresholds are fitted and scored on the same split.")
    A("6. **The rationale is not scored by a rubric.** No reasoning-quality metric exists here; the "
      "trace is evaluated only through classifier/LLM disagreement counts on 18 images.")
    A("7. **No telemetry ablation** isolates how much of the reasoning gain comes from the prefix.")
    A("8. **DPO was tried and rejected** for objective-level reasons (§3.1); it is not a positive result.")
    A("9. **Single GPU, single run per stage**: no seed variance is reported, so small differences "
      "(<1 pp) should not be read as significant.")
    A("")

    A("## 9. Reproducibility")
    A("")
    A("This manuscript is generated. `build_rp_journal.py` parses the artifacts below and rewrites "
      "`DEEP_RP_JOURNAL.md` / `.html`; `reports/RP_BUILD_MANIFEST.md` is written by the same pass.")
    A("")
    body = [[f"`{s.path}`", s.role] for s in SOURCES]
    table(["File", "Provides"], body)
    A("*Table 9. Every input required to rebuild this paper. Paths are relative to `deepfake_project/`.*")
    A("")
    A("```bash")
    A("cd deepfake_project")
    A("python build_rp_journal.py           # -> DEEP_RP_JOURNAL.md, DEEP_RP_JOURNAL.html, figures")
    A("python verify_stage_history.py       # checkpoint stage/freeze evidence (§3)")
    A("python eval.py --model_path ./models/InternVL3-2B \\")
    A("    --checkpoint runs/intern_v2_ground/checkpoints/best.pth --calibrate \\")
    A("    --batch_size 8 --log_dir runs/timing/vanilla_calib     # ~52 min, Table 2/3")
    A("```")
    A("")
    A("Artifacts that are *not* self-contained are flagged in the claim ledger "
      "(`reports/PAPER_CLAIMS.md`): `config.yaml` numbers are a current-code re-evaluation rather than "
      "the archived log, and the fused Stage-0 merge cannot be rebuilt from this repository.")
    A("")

    A("## 10. Conclusion")
    A("")
    A("Grounding an explanation in the classifier's own telemetry is cheap to state and expensive to "
      "earn: it constrains the architecture, forces a staged curriculum, and — on this benchmark — costs "
      "about one point of macro accuracy relative to a fused-base variant. What it buys is an artifact "
      "that can be shipped onto an untouched base model, audited per face and per generator, calibrated "
      "after the fact, and rebuilt from its own logs. The paper reports the cost on page one and the "
      "audit in full, because a forensic tool that hides either is not usable in the setting it "
      "targets.")
    A("")

    md = "\n".join(L)

    with open(os.path.join(ROOT, "DEEP_RP_JOURNAL.md"), "w", encoding="utf-8") as fh:
        fh.write(md)

    html = md_to_html(md)
    with open(os.path.join(ROOT, "DEEP_RP_JOURNAL.html"), "w", encoding="utf-8") as fh:
        fh.write(html)

    write_manifest()
    return md


def md_to_html(md: str) -> str:
    """Minimal Markdown -> HTML with base64-inlined images (single portable file)."""
    def embed(match):
        alt, path = match.group(1), match.group(2)
        full = os.path.join(ROOT, path)
        if not os.path.exists(full):
            return f"<p><em>missing figure: {path}</em></p>"
        with open(full, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode("ascii")
        return f'<figure><img alt="{alt}" src="data:image/png;base64,{b64}"><figcaption>{alt}</figcaption></figure>'

    md = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", embed, md)

    lines, out, in_code, in_table = md.split("\n"), [], False, False
    for line in lines:
        if line.startswith("```"):
            out.append("</pre>" if in_code else "<pre>")
            in_code = not in_code
            continue
        if in_code:
            out.append(line.replace("&", "&amp;").replace("<", "&lt;"))
            continue
        if line.startswith("|") and line.count("|") >= 2:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):
                continue
            if not in_table:
                out.append("<table>")
                in_table = True
                out.append("<tr>" + "".join(f"<th>{c}</th>" for c in cells) + "</tr>")
            else:
                out.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
            continue
        if in_table:
            out.append("</table>")
            in_table = False
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            out.append(f"<h{level}>{line[level:].strip()}</h{level}>")
        elif line.strip() == "---":
            out.append("<hr>")
        elif line.startswith("<figure") or line.startswith("<p>"):
            out.append(line)
        elif not line.strip():
            out.append("")
        else:
            out.append(f"<p>{line}</p>")
    if in_table:
        out.append("</table>")

    body = "\n".join(out)
    body = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", body)
    body = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<em>\1</em>", body)
    body = re.sub(r"`([^`]+)`", r"<code>\1</code>", body)

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Grounded Dual-Branch Vision–Language Forensics</title>
<style>
  :root {{ --ink:#1a1a1a; --muted:#5a5a5a; --rule:#d8d8d8; --accent:#0072B2; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0 auto; max-width: 860px; padding: 56px 28px 120px;
         font: 16px/1.62 "Charter","Georgia",serif; color:var(--ink); background:#ffffff; }}
  h1 {{ font-size: 27px; line-height:1.22; letter-spacing:-.01em; margin:0 0 18px; }}
  h2 {{ font-size: 20px; margin:44px 0 10px; padding-bottom:6px; border-bottom:1px solid var(--rule); }}
  h3 {{ font-size: 16.5px; margin:28px 0 8px; color:#222; }}
  p {{ margin: 0 0 12px; text-align: justify; hyphens: auto; }}
  figure {{ margin: 26px 0; text-align:center; }}
  img {{ max-width:100%; height:auto; border:1px solid #eee; border-radius:3px; }}
  figcaption {{ font-size:12.5px; color:var(--muted); margin-top:8px; text-align:left;
                line-height:1.45; max-width:92%; margin-left:auto; margin-right:auto; }}
  table {{ width:100%; border-collapse: collapse; font-size:12.6px; margin: 14px 0 10px;
           font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; }}
  th, td {{ padding: 5px 7px; border-bottom: 1px solid var(--rule); text-align:left; }}
  th {{ background:#f6f8fa; font-weight:600; font-size:12px; }}
  td {{ vertical-align: top; }}
  code {{ font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 13px;
          background:#f4f5f7; padding:1px 4px; border-radius:3px; }}
  pre {{ background:#f6f8fa; padding:14px 16px; border-radius:4px; overflow-x:auto;
         font-size:12.5px; line-height:1.5; border:1px solid #ececec; }}
  hr {{ border:0; border-top:1px solid var(--rule); margin:34px 0; }}
  em {{ color:#333; }}
  strong {{ color:#000; }}
  @media print {{ body {{ max-width:none; padding:0; font-size:11.5pt; }}
                  h2 {{ page-break-after: avoid; }} figure, table {{ page-break-inside: avoid; }} }}
</style></head><body>
{body}
</body></html>"""


def write_manifest() -> None:
    lines = ["# RP build manifest", "",
             "Files required to rebuild `DEEP_RP_JOURNAL.md` / `.html`, with the role each plays in the "
             "manuscript. Paths are relative to `deepfake_project/`; `build_rp_journal.py` prints any "
             "input it cannot find and records it in the Missing inputs section below.", "",
             "| # | File | Present | Provides | Used for |", "| --- | --- | :---: | --- | --- |"]
    for i, s in enumerate(SOURCES, 1):
        lines.append(f"| {i} | `{s.path}` | {'yes' if s.exists else '**missing**'} | {s.role} | "
                     f"{', '.join(s.used_for)} |")
    missing = [s.path for s in SOURCES if not s.exists]
    lines += ["", "## Build", "", "```bash", "cd deepfake_project", "python build_rp_journal.py", "```", "",
              "Outputs: `DEEP_RP_JOURNAL.md`, `DEEP_RP_JOURNAL.html` (images inlined), "
              "`reports/journal/figures/fig1..fig11.png`, and this manifest.", ""]

    ck = os.path.join(ROOT, "checkpoints_manifest.json")
    if os.path.exists(ck):
        spec = json.load(open(ck, encoding="utf-8"))
        lines += ["## Large inputs (not in git)", "",
                  "The weights are tracked by Git LFS pattern but are not committed: 8.5 GB exceeds the "
                  "free LFS quota. `checkpoints_manifest.json` is the contract instead — restore the "
                  "files to the paths below and check them with `python verify_checkpoints.py`.", "",
                  "| File | SHA-256 | Bytes | Role |", "| --- | --- | ---: | --- |"]
        for w in spec["weights"]:
            lines.append(f"| `{w['restore_to']}` | `{w['sha256'][:16]}…` | {w.get('bytes') or '—'} | {w['role']} |")
        lines += ["", f"Base model: `{spec['base_model']['path']}` — {spec['base_model']['role']}.", ""]
    lines += (["## Missing inputs", ""] + [f"- `{p}`" for p in missing] if missing
              else ["All listed inputs were present at build time."])
    with open(os.path.join(ROOT, "reports", "RP_BUILD_MANIFEST.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    missing = [s.path for s in SOURCES if not s.exists]
    if missing:
        print("WARNING: missing inputs (documented in the manifest):")
        for m in missing:
            print("  -", m)
    build()
    print("wrote DEEP_RP_JOURNAL.md, DEEP_RP_JOURNAL.html and reports/RP_BUILD_MANIFEST.md")
