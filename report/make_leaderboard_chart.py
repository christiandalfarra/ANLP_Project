#!/usr/bin/env python3
"""Horizontal bar chart of test ROUGE-L for all conditions, with bootstrap
95% CIs. Data: results/percondition_cis.csv (from compute_analysis.py) and
results/nullbaselines.csv (from compute_nullbaselines.py).
All rows are clean 9-match evaluations, finetuned_led included (run 5 is a
clean retrain). The dashed line is the strongest null baseline: another test
match's reference, i.e. the score attainable with zero correct match facts."""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))

LABELS = {
    "flan_chunk_cot": ("FLAN chunk CoT", "flan"),
    "flan_chunk_few": ("FLAN chunk few", "flan"),
    "flan_chunk_zero": ("FLAN chunk zero", "flan"),
    "bart_chunk_few": ("BART chunk few", "bart"),
    "bart_chunk_cot": ("BART chunk CoT", "bart"),
    "bart_chunk_zero": ("BART chunk zero", "bart"),
    "led_long_cot": ("LED long CoT", "led"),
    "led_long_zero": ("LED long zero", "led"),
    "led_long_few": ("LED long few", "led"),
    "finetuned_led": ("Fine-tuned LED", "ft"),
    "finetuned_bart": ("Fine-tuned BART", "ft"),
}
colors = {"flan": "#c8c8c8", "bart": "#9db8d9", "led": "#5d8ac4", "ft": "#0a3d7a"}

rows = []
with open(os.path.join(HERE, "results", "percondition_cis.csv")) as fh:
    for r in csv.DictReader(fh):
        label, fam = LABELS[r["condition"]]
        rows.append((label, fam, float(r["rougeL"]),
                     float(r["rougeL_lo"]), float(r["rougeL_hi"])))
rows.sort(key=lambda x: x[2])          # ascending so best ends up on top

with open(os.path.join(HERE, "results", "nullbaselines.csv")) as fh:
    nulls = {r["baseline"]: float(r["rougeL"]) for r in csv.DictReader(fh)}
NULL_FLOOR = nulls["cross-match reference (test donors)"]

fig, ax = plt.subplots(figsize=(8.4, 4.6))
labels = [r[0] for r in rows]
vals = [r[2] for r in rows]
cols = [colors[r[1]] for r in rows]
err_lo = [r[2] - r[3] for r in rows]
err_hi = [r[4] - r[2] for r in rows]

bars = ax.barh(labels, vals, color=cols, height=0.68,
               xerr=[err_lo, err_hi], error_kw=dict(lw=1.1, capsize=3,
                                                    ecolor="#444444"))
for b, (_, _, v, _, hi) in zip(bars, rows):
    ax.text(hi + 0.006, b.get_y() + b.get_height() / 2, f"{v:.3f}",
            va="center", fontsize=9)

ax.axvline(NULL_FLOOR, color="#b03030", ls="--", lw=1.3, zorder=0)
ax.text(NULL_FLOOR - 0.004, -0.62,
        f"null floor {NULL_FLOOR:.3f}: another match's report\n(correct format, every fact wrong)",
        ha="right", va="bottom", fontsize=8.5, color="#b03030", style="italic")

ax.set_xlabel("Test ROUGE-L (whiskers: bootstrap 95% CI, 10,000 resamples)")
ax.set_xlim(0, 0.31)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(handles=[
    Patch(color=colors["ft"], label="Fine-tuned"),
    Patch(color=colors["led"], label="Prompting — LED long-context"),
    Patch(color=colors["bart"], label="Prompting — BART chunk"),
    Patch(color=colors["flan"], label="Prompting — FLAN chunk"),
], loc="lower right", fontsize=8.5, frameon=False)

fig.tight_layout()
fig.savefig(os.path.join(HERE, "figures", "leaderboard.png"), dpi=170,
            bbox_inches="tight")
fig.savefig(os.path.join(HERE, "figures", "leaderboard.pdf"),
            bbox_inches="tight")
print("wrote figures/leaderboard.{png,pdf}")
