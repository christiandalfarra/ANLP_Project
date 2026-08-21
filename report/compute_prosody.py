#!/usr/bin/env python3
"""Prosody analysis: do crowd/commentator energy peaks coincide with goals?

This is a *dataset-level* analysis (independent of any model prediction), so it
runs over ALL matches that have a parseable "Key events" block with goal minutes
(90 of 99, 278 goal events) for maximum statistical power.

The hard part is that the audio clock (segment start-seconds) is not the match
clock (goal minutes): every broadcast has a variable pre-match build-up and a
~15-minute half-time gap, and neither is timestamped. We therefore:

  1. Estimate a per-match kick-off offset `o` GOAL-INDEPENDENTLY (first sustained
     energy rise), and assume a fixed 15-min half-time gap. This mapping is crude
     but is fixed *without looking at the goals*, so the permutation test below is
     not circular.
  2. Map each goal minute g to an audio second and read off the energy percentile
     of the surrounding minute-bin.
  3. Permutation-test the mean goal-time energy percentile against random minutes.
  4. Report, as a sensitivity upper bound, an "oracle" alignment whose offset is
     grid-searched to maximise goal-time energy (optimistically biased on purpose).

Outputs: results/prosody.csv  and  figures/prosody_alignment.pdf
"""
import os
import re
import glob

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SUMM = os.path.join(ROOT, "football_commentary_dataset", "data", "summaries")
SEGS = os.path.join(ROOT, "football_commentary_dataset", "data", "transcripts")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

BIN_SEC = 60.0          # one bin per audio-minute
HALFTIME_GAP_S = 15 * 60  # assumed break between 1st and 2nd half
SMOOTH_MIN = 2          # rolling-mean half-window (minutes) for offset heuristic
N_PERM = 10_000
SEED = 42

GOAL_PAT = re.compile(
    r"(\d+)(?:\+(\d+))?['′’]\s*[–—-]\s*"
    r"[A-Z].*?\b(goal|equaliser|equalizer|winner|penalty)\b",
    re.I,
)
SEG_PAT = re.compile(
    r"\[(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\]\s*P:[\d.]+\s*E:([\d.]+)\s*\|"
)


def parse_goal_minutes(match_id):
    """Minutes (int, added time folded into base) from the Key-events block."""
    path = os.path.join(SUMM, match_id + ".txt")
    text = open(path, encoding="utf-8").read()
    idx = text.lower().find("key events")
    block = text[idx:] if idx != -1 else text
    mins = []
    for base, added, _kind in GOAL_PAT.findall(block):
        mins.append(int(base) + (int(added) if added else 0))
    return sorted(set(mins))


def load_energy(match_id):
    """(start_seconds, energy) arrays from the _segments.txt file."""
    path = os.path.join(SEGS, match_id + "_segments.txt")
    starts, energies = [], []
    for line in open(path, encoding="utf-8"):
        m = SEG_PAT.match(line.strip())
        if m:
            starts.append(float(m.group(1)))
            energies.append(float(m.group(3)))
    return np.asarray(starts), np.asarray(energies)


def minute_bins(starts, energies):
    """Mean energy per audio-minute bin, and its percentile-rank transform."""
    if len(starts) == 0:
        return None, None
    n_bins = int(starts.max() // BIN_SEC) + 1
    sums = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    idx = (starts // BIN_SEC).astype(int)
    np.add.at(sums, idx, energies)
    np.add.at(counts, idx, 1)
    with np.errstate(invalid="ignore"):
        mean = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    # fill empty bins by interpolation so percentile ranks are well defined
    good = ~np.isnan(mean)
    if good.sum() < 2:
        return None, None
    mean = np.interp(np.arange(n_bins), np.flatnonzero(good), mean[good])
    # percentile rank of each bin (0..1)
    order = mean.argsort()
    ranks = np.empty(n_bins)
    ranks[order] = np.linspace(0, 1, n_bins)
    return mean, ranks


def estimate_offset_bins(mean):
    """Goal-INDEPENDENT kick-off estimate: first audio-minute in the first 20%
    of the broadcast whose smoothed energy first exceeds the match median.
    Returns the kickoff bin index (audio-minute of minute 0 of the match)."""
    n = len(mean)
    k = SMOOTH_MIN
    smooth = np.convolve(mean, np.ones(2 * k + 1) / (2 * k + 1), mode="same")
    med = np.median(mean)
    horizon = max(3, int(0.20 * n))
    for b in range(horizon):
        if smooth[b] >= med:
            return b
    return 0


def goal_bins(goal_minutes, kickoff_bin, n_bins):
    """Map match-minutes to audio-minute bins (offset + halftime gap)."""
    ht = int(HALFTIME_GAP_S // BIN_SEC)
    out = []
    for g in goal_minutes:
        b = kickoff_bin + g + (ht if g > 45 else 0)
        if 0 <= b < n_bins:
            out.append(b)
    return out


def oracle_offset_pctile(ranks, goal_minutes, n_bins):
    """Sensitivity upper bound: grid-search the offset to MAXIMISE goal energy."""
    ht = int(HALFTIME_GAP_S // BIN_SEC)
    best = -1.0
    for o in range(0, min(16, n_bins)):        # 0..15 audio-min of build-up
        bins = [o + g + (ht if g > 45 else 0) for g in goal_minutes]
        bins = [b for b in bins if 0 <= b < n_bins]
        if not bins:
            continue
        val = float(np.mean([ranks[b] for b in bins]))
        best = max(best, val)
    return best


def main():
    rng = np.random.default_rng(SEED)
    ids = sorted(f[:-4] for f in os.listdir(SUMM) if f.endswith(".txt"))

    rows = []                 # per-match records
    all_goal_pctiles = []     # pooled goal-time percentiles (primary)
    all_oracle = []
    # For the permutation null we store, per match, (bin_ranks, n_goals_used).
    perm_data = []

    for mid in ids:
        gmins = parse_goal_minutes(mid)
        if not gmins:
            continue
        starts, energies = load_energy(mid)
        mean, ranks = minute_bins(starts, energies)
        if ranks is None:
            continue
        n_bins = len(ranks)
        kb = estimate_offset_bins(mean)
        gbins = goal_bins(gmins, kb, n_bins)
        if not gbins:
            continue
        gp = [float(ranks[b]) for b in gbins]
        oracle = oracle_offset_pctile(ranks, gmins, n_bins)

        all_goal_pctiles.extend(gp)
        all_oracle.append(oracle)
        perm_data.append((ranks, len(gbins)))
        rows.append((mid, len(gmins), len(gbins), kb,
                     float(np.mean(gp)), oracle))

    goal_arr = np.asarray(all_goal_pctiles)
    obs_mean = goal_arr.mean()

    # ---- permutation test: replace each match's goals with random minute-bins
    perm_means = np.empty(N_PERM)
    for i in range(N_PERM):
        vals = []
        for ranks, k in perm_data:
            vals.append(rng.choice(ranks, size=k, replace=False))
        perm_means[i] = np.concatenate(vals).mean()
    p_value = (np.sum(perm_means >= obs_mean) + 1) / (N_PERM + 1)

    coincidence = float(np.mean(goal_arr >= 0.80))   # goals in top-quintile min
    oracle_mean = float(np.mean(all_oracle))

    # ---- write CSV -------------------------------------------------------
    with open(os.path.join(OUT, "prosody.csv"), "w") as fh:
        fh.write("match_id,n_goals,n_goals_mapped,kickoff_bin,"
                 "mean_goal_pctile,oracle_pctile\n")
        for r in rows:
            fh.write("{},{},{},{},{:.4f},{:.4f}\n".format(*r))

    # ---- console summary -------------------------------------------------
    print(f"matches used            : {len(rows)}")
    print(f"goal events (mapped)    : {len(goal_arr)}")
    print(f"mean goal-time pctile   : {obs_mean:.3f}  (null = 0.50)")
    print(f"permutation p-value     : {p_value:.4f}  ({N_PERM} draws)")
    print(f"null mean / 95%% band    : {perm_means.mean():.3f} "
          f"[{np.percentile(perm_means,2.5):.3f}, "
          f"{np.percentile(perm_means,97.5):.3f}]")
    print(f"top-quintile coincidence: {coincidence*100:.1f}%  (chance = 20%)")
    print(f"oracle-offset pctile     : {oracle_mean:.3f}  (biased upper bound)")

    # ---- figure: one clean modern 4-goal test match ----------------------
    demo = "2021_euro_england_ukraine"
    starts, energies = load_energy(demo)
    mean, ranks = minute_bins(starts, energies)
    kb = estimate_offset_bins(mean)
    gmins = parse_goal_minutes(demo)
    ht = int(HALFTIME_GAP_S // BIN_SEC)
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    t = np.arange(len(mean))
    ax.plot(t, mean, lw=0.8, color="#444", alpha=0.6, label="energy (per audio-min)")
    sm = np.convolve(mean, np.ones(5) / 5, mode="same")
    ax.plot(t, sm, lw=1.6, color="#c0392b", label="smoothed")
    ax.axvline(kb, ls=":", color="#2980b9", lw=1.2, label="est. kick-off")
    for g in gmins:
        b = kb + g + (ht if g > 45 else 0)
        if 0 <= b < len(mean):
            ax.axvline(b, ls="--", color="#27ae60", lw=1.1)
    ax.plot([], [], ls="--", color="#27ae60", lw=1.1, label="goal (mapped)")
    ax.set_xlabel("audio minute")
    ax.set_ylabel("RMS energy")
    ax.set_title(f"Energy vs. mapped goal times — {demo}")
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "prosody_alignment.pdf"))
    fig.savefig(os.path.join(FIG, "prosody_alignment.png"), dpi=150)
    print(f"\nwrote results/prosody.csv and figures/prosody_alignment.pdf")


if __name__ == "__main__":
    main()
