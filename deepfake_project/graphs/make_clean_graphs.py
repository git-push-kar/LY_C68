#!/usr/bin/env python3
"""
Clean, slide-ready training + eval graphs for exp2 ep1-10 only.
White background, muted beautiful palette, minimal ink.
"""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# --- Style ---
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#D9D9D9",
    "axes.linewidth": 0.9,
    "grid.color": "#EEEEEE",
    "grid.linewidth": 0.7,
    "text.color": "#2B2B2B",
    "axes.labelcolor": "#3A3A3A",
    "xtick.color": "#4A4A4A",
    "ytick.color": "#4A4A4A",
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Segoe UI", "Helvetica", "Arial"],
})

# Beautiful muted palette (colorblind-friendly)
C_TR_LOSS = "#4C78A8"  # blue
C_VA_LOSS = "#72B66F"  # green
C_TR_ACC  = "#E45756"  # coral red
C_VA_ACC  = "#F58518"  # warm orange
C_VA_AUC  = "#54A8B5"  # teal
C_CORR    = "#B279A2"  # muted purple for corrective point
C_BEST    = "#EECA3B"  # gold star for best

# --- Load epochs.csv ---
import os
from collections import OrderedDict

def load_epochs(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    # deduplicate epoch 2 (keep last occurrence)
    dedup = OrderedDict()
    for r in rows:
        dedup[int(r["epoch"])] = r  # overwrites earlier ep2 with later
    return list(dedup.values())

exp2 = load_epochs("runs/intern_exp2/logs/epochs.csv")

# Only exp2 1-10
combined = [r for r in exp2 if int(r["epoch"]) <= 10]

# Extract series
epochs = [int(r["epoch"]) for r in combined]
x_labels = [str(e) for e in epochs]
tr_loss = [float(r["tr_loss"]) for r in combined]
va_loss = [float(r["va_loss"]) for r in combined]
tr_acc  = [float(r["tr_acc"])  for r in combined]
va_acc  = [float(r["va_acc"])  for r in combined]
va_auc  = [float(r["va_auc"])  for r in combined]
is_best = [int(r["is_best"]) for r in combined]

# Markers: best = gold star, corrective = purple diamond
# For training, is_best at epochs 1-10 = stars, epoch 11 = diamond

# ============ FIGURE 1: TRAINING ============
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), gridspec_kw={"wspace": 0.32})
fig.suptitle("Training", fontsize=13, fontweight=600, y=1.04, color="#1A1A1A")

# Panel 1: Loss
ax = axes[0]
ax.plot(epochs, tr_loss, color=C_TR_LOSS, linewidth=2.2, marker="o", markersize=4.5, markerfacecolor="white", markeredgewidth=1.3, label="Train loss", zorder=3)
ax.plot(epochs, va_loss, color=C_VA_LOSS, linewidth=2.2, marker="s", markersize=4.2, markerfacecolor="white", markeredgewidth=1.3, label="Val loss", zorder=3)
ax.set_title("Loss", fontsize=10.5, fontweight=600, pad=10)
ax.set_xlabel("Epoch", fontsize=9.5)
ax.set_ylabel("Loss", fontsize=9.5)
ax.set_xticks(epochs); ax.set_xticklabels(x_labels, fontsize=8.5)
ax.set_ylim(min(min(tr_loss), min(va_loss))*0.88, max(max(tr_loss), max(va_loss))*1.08)
ax.grid(True, axis="y", linestyle="--", alpha=0.9)
ax.grid(True, axis="x", linestyle="--", alpha=0.45)
for spine in ax.spines.values(): spine.set_visible(True)
ax.legend(frameon=True, facecolor="white", edgecolor="#E8E8E8", fontsize=7.8, loc="upper right", handlelength=1.6)

# Panel 2: Accuracy
ax = axes[1]
ax.plot(epochs, tr_acc, color=C_TR_ACC, linewidth=2.2, marker="o", markersize=4.5, markerfacecolor="white", markeredgewidth=1.3, label="Train acc", zorder=3)
ax.plot(epochs, va_acc, color=C_VA_ACC, linewidth=2.2, marker="s", markersize=4.2, markerfacecolor="white", markeredgewidth=1.3, label="Val acc", zorder=3)
# best epochs — gold star where is_best==1
for e, a, b in zip(epochs, va_acc, is_best):
    if b==1:
        ax.plot([e],[a], marker="*", color=C_BEST, markersize=9, markeredgecolor="white", markeredgewidth=0.6, linestyle="none", zorder=4, alpha=0.95)
ax.set_title("Accuracy", fontsize=10.5, fontweight=600, pad=10)
ax.set_xlabel("Epoch", fontsize=9.5)
ax.set_ylabel("Accuracy", fontsize=9.5)
ax.set_xticks(epochs); ax.set_xticklabels(x_labels, fontsize=8.5)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
ax.set_ylim(0.58, 0.90)
ax.grid(True, axis="y", linestyle="--", alpha=0.9)
ax.grid(True, axis="x", linestyle="--", alpha=0.45)
ax.legend(frameon=True, facecolor="white", edgecolor="#E8E8E8", fontsize=7.8, loc="lower right", handlelength=1.6)

# Panel 3: AUC (val only)
ax = axes[2]
ax.plot(epochs, va_auc, color=C_VA_AUC, linewidth=2.4, marker="o", markersize=4.8, markerfacecolor="white", markeredgewidth=1.4, label="Val AUC", zorder=3)
for e,a,b in zip(epochs, va_auc, is_best):
    if b==1:
        ax.plot([e],[a], marker="*", color=C_BEST, markersize=9, markeredgecolor="white", markeredgewidth=0.6, linestyle="none", zorder=4, alpha=0.95)
ax.set_title("Ranking (AUC) — Val", fontsize=10.5, fontweight=600, pad=10)
ax.set_xlabel("Epoch", fontsize=9.5)
ax.set_ylabel("AUC", fontsize=9.5)
ax.set_xticks(epochs); ax.set_xticklabels(x_labels, fontsize=8.5)
ax.set_ylim(0.84, 0.965)
ax.grid(True, axis="y", linestyle="--", alpha=0.9)
ax.grid(True, axis="x", linestyle="--", alpha=0.45)
# despine top/right
for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("graphs/training_clean.png", dpi=220, bbox_inches="tight", facecolor="white")
plt.savefig("runs/intern_exp3_corrective/logs/training_clean.png", dpi=220, bbox_inches="tight", facecolor="white")
print("saved training_clean.png")
plt.close()

# ============ FIGURE 2: EVAL (MEAN ONLY) ============

eval_epochs = [1, 2, 4, 6, 7, 8, 10]
eval_mean_acc = [0.7344, 0.7510, 0.7580, 0.7758, 0.8055, 0.7818, 0.8243]
eval_mean_auc = [0.8063, 0.8504, 0.8848, 0.8906, 0.9000, 0.8927, 0.9031]

fig, ax = plt.subplots(1, 1, figsize=(7.2, 3.6))
fig.suptitle("Evaluation", fontsize=13, fontweight=600, y=1.02, color="#1A1A1A")

ax.plot(eval_epochs, eval_mean_acc, color=C_TR_ACC, linewidth=2.4, marker="o", markersize=5, markerfacecolor="white", markeredgewidth=1.4, label="MEAN Acc", zorder=3)
ax.plot(eval_epochs, eval_mean_auc, color=C_VA_AUC, linewidth=2.4, marker="s", markersize=4.8, markerfacecolor="white", markeredgewidth=1.4, label="MEAN AUC", zorder=3)
ax.set_title("MEAN over ID / CM / CF / CD", fontsize=10.5, fontweight=600, pad=10)
ax.set_xlabel("Eval checkpoint (epoch)", fontsize=9.5)
ax.set_ylabel("Score", fontsize=9.5)
ax.set_xticks(eval_epochs); ax.set_xticklabels([str(e) for e in eval_epochs], fontsize=8.5)
ax.set_ylim(0.68, 0.97)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
ax.grid(True, axis="y", linestyle="--", alpha=0.9)
ax.grid(True, axis="x", linestyle="--", alpha=0.45)
ax.legend(frameon=True, facecolor="white", edgecolor="#E8E8E8", fontsize=8, loc="lower right", handlelength=1.6)
for spine in ax.spines.values(): spine.set_visible(True)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("graphs/eval_clean.png", dpi=220, bbox_inches="tight", facecolor="white")
plt.savefig("runs/intern_exp3_corrective/logs/eval_clean.png", dpi=220, bbox_inches="tight", facecolor="white")
print("saved eval_clean.png")
plt.close()
