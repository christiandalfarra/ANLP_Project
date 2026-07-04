#!/usr/bin/env python3
"""Horizontal bar chart of test ROUGE-L for all conditions (run4 clean;
finetuned_led from run3 valid-only re-eval, shown hatched)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# (label, rougeL, family)  — run4/metrics.csv except finetuned_led (run3 valid-only)
rows = [
    ("FLAN chunk CoT",        0.0481, "flan"),
    ("FLAN chunk few",        0.0551, "flan"),
    ("FLAN chunk zero",       0.0700, "flan"),
    ("BART chunk few",        0.1030, "bart"),
    ("BART chunk CoT",        0.1230, "bart"),
    ("BART chunk zero",       0.1467, "bart"),
    ("LED long CoT",          0.1548, "led"),
    ("LED long zero",         0.1599, "led"),
    ("LED long few",          0.1599, "led"),
    ("Fine-tuned LED*",       0.2190, "ft"),
    ("Fine-tuned BART",       0.2476, "ft"),
]

colors = {"flan": "#c8c8c8", "bart": "#9db8d9", "led": "#5d8ac4", "ft": "#0a3d7a"}

fig, ax = plt.subplots(figsize=(8.4, 4.4))
labels = [r[0] for r in rows]
vals = [r[1] for r in rows]
cols = [colors[r[2]] for r in rows]
bars = ax.barh(labels, vals, color=cols, height=0.68)
bars[9].set_hatch("//")
bars[9].set_edgecolor("white")

for b, v in zip(bars, vals):
    ax.text(v + 0.003, b.get_y() + b.get_height() / 2, f"{v:.3f}",
            va="center", fontsize=9)

ax.set_xlabel("Test ROUGE-L")
ax.set_xlim(0, 0.29)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(handles=[
    Patch(color=colors["ft"], label="Fine-tuned"),
    Patch(color=colors["led"], label="Prompting — LED long-context"),
    Patch(color=colors["bart"], label="Prompting — BART chunk"),
    Patch(color=colors["flan"], label="Prompting — FLAN chunk"),
], loc="lower right", fontsize=8.5, frameon=False)

fig.tight_layout()
fig.savefig("figures/leaderboard.png", dpi=170, bbox_inches="tight")
fig.savefig("figures/leaderboard.pdf", bbox_inches="tight")
print("wrote figures/leaderboard.{png,pdf}")
