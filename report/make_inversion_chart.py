#!/usr/bin/env python3
"""Two-panel dot plot: BERTScore (vs reference) and NLI support (vs
transcript) for representative conditions — the reference-similarity vs
source-grounding inversion. Data: results/semantic.csv.
Palette validated (dataviz six checks, light surface): #2563a8 / #c2622d.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# (label, bertscore, nli, group) — ordered by BERTScore descending
rows = [
    ("Fine-tuned BART",      0.860, 0.00, "ft"),
    ("Fine-tuned LED",       0.848, 0.03, "ft"),
    ("BART chunk zero-shot", 0.819, 0.29, "pr"),
    ("FLAN chunk zero-shot", 0.807, 0.11, "pr"),
    ("LED long zero-shot",   0.792, 0.34, "pr"),
]
colors = {"ft": "#2563a8", "pr": "#c2622d"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.1), sharey=True,
                               gridspec_kw={"wspace": 0.06})
y = list(range(len(rows)))[::-1]

for ax, key, xlim, fmt in ((ax1, 1, (0.775, 0.875), "{:.3f}"),
                           (ax2, 2, (-0.015, 0.40), "{:.2f}")):
    for yi, r in zip(y, rows):
        ax.hlines(yi, xlim[0], r[key], color="#d9d9d9", lw=1, zorder=1)
        ax.plot(r[key], yi, "o", ms=9, color=colors[r[3]], zorder=3)
        ax.annotate(fmt.format(r[key]), (r[key], yi), textcoords="offset points",
                    xytext=(8, -3.5), fontsize=9, color="#333333")
    ax.set_xlim(*xlim)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(left=False)
    ax.grid(axis="x", color="#eeeeee", lw=0.8, zorder=0)
    ax.set_axisbelow(True)

ax1.set_yticks(y)
ax1.set_yticklabels([r[0] for r in rows], fontsize=10)
ax1.set_xlabel("BERTScore F1 — similarity to the reference", fontsize=10)
ax2.set_xlabel("NLI support — grounding in the transcript", fontsize=10)
ax1.set_title("Best model by every reference metric…", fontsize=10.5,
              color="#2563a8", loc="left")
ax2.set_title("…is the least grounded in the source", fontsize=10.5,
              color="#b00000", loc="left")

ax1.legend(handles=[
    Line2D([], [], marker="o", ls="", ms=8, color=colors["ft"], label="Fine-tuned"),
    Line2D([], [], marker="o", ls="", ms=8, color=colors["pr"], label="Prompting"),
], loc="lower right", fontsize=9, frameon=False, handletextpad=0.2)

fig.tight_layout()
fig.savefig("figures/inversion.png", dpi=170, bbox_inches="tight")
fig.savefig("figures/inversion.pdf", bbox_inches="tight")
print("wrote figures/inversion.{png,pdf}")
