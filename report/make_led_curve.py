#!/usr/bin/env python3
"""Generate the LED fine-tuning training-curve figure for the report.

By default the per-epoch validation metrics are those logged during run3
(runs/run3_led_finetuned_2026-05-03/kernel.log). After a clean LED retrain,
drop the run's trainer_state.json into runs/run5_led_clean_*/ (or pass its path
as argv[1]) and this script reads the per-epoch metrics from it automatically.
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# run3 fallback (hardcoded from kernel.log)
epochs = list(range(1, 12))
val_rougeL = [0.1066, 0.1105, 0.1135, 0.1113, 0.1251,
              0.1298, 0.1587, 0.1604, 0.1718, 0.1707, 0.1621]
eval_loss = [4.829, 4.333, 3.588, 2.707, 2.472,
             2.320, 2.157, 1.999, 1.851, 1.737, 1.631]


def _load_trainer_state(path):
    log = json.load(open(path))["log_history"]
    evals = [e for e in log if "eval_rougeL" in e]
    ep = [e["epoch"] for e in evals]
    rl = [e["eval_rougeL"] for e in evals]
    loss = [e.get("eval_loss") for e in evals]
    return ep, rl, loss


_state = sys.argv[1] if len(sys.argv) > 1 else None
if _state is None:
    hits = sorted(glob.glob(os.path.join(
        os.path.dirname(__file__), "..", "runs",
        "run5_led_clean_*", "trainer_state.json")))
    _state = hits[-1] if hits else None
if _state and os.path.exists(_state):
    epochs, val_rougeL, eval_loss = _load_trainer_state(_state)
    print(f"using clean retrain metrics from {_state}")
else:
    print("using hardcoded run3 metrics (no run5 trainer_state.json found)")

fig, ax1 = plt.subplots(figsize=(5.6, 3.2))

c1 = "#0a58ca"
ax1.plot(epochs, val_rougeL, "-o", color=c1, markersize=4, label="val ROUGE-L")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Validation ROUGE-L", color=c1)
ax1.tick_params(axis="y", labelcolor=c1)
ax1.set_xticks([int(e) for e in epochs])
_lo, _hi = min(val_rougeL), max(val_rougeL)
_pad = max(0.005, (_hi - _lo) * 0.15)
ax1.set_ylim(_lo - _pad, _hi + _pad)

# mark the best (early-stopping) epoch
best = val_rougeL.index(max(val_rougeL))
ax1.annotate(f"peak {val_rougeL[best]:.3f}\n(epoch {epochs[best]})",
             xy=(epochs[best], val_rougeL[best]),
             xytext=(epochs[best] - 3.4, val_rougeL[best] + 0.005),
             fontsize=8, color=c1,
             arrowprops=dict(arrowstyle="->", color=c1, lw=0.8))

c2 = "#b00000"
ax2 = ax1.twinx()
ax2.plot(epochs, eval_loss, "--s", color=c2, markersize=3, label="eval loss")
ax2.set_ylabel("Evaluation loss", color=c2)
ax2.tick_params(axis="y", labelcolor=c2)

fig.tight_layout()
fig.savefig("figures/led_training_curve.pdf", bbox_inches="tight")
fig.savefig("figures/led_training_curve.png", dpi=150, bbox_inches="tight")
print("wrote figures/led_training_curve.{pdf,png}")
