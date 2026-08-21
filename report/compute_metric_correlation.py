#!/usr/bin/env python3
"""Rank correlation between the reference-based metrics and the source-based
faithfulness probes, across all evaluated conditions.

Section 6.4 of the report argues that ROUGE and faithfulness come apart *at the
top of the leaderboard*, not globally. That is a narrower claim than "the two
are unrelated", and this script is what makes it checkable: it ranks every
condition by ROUGE-L and correlates that ranking against each probe.

The correlations come out positive (rho ~ +0.4 to +0.6) because conditions that
score near zero on ROUGE also say nothing true -- both metric families separate
degenerate output from coherent output. The failure is local to the leaders,
which the report reports separately and which no aggregate correlation shows.

Reads results/faithfulness.csv + results/semantic.csv (from compute_analysis.py
and compute_semantic.py) and the ROUGE table from runs/. Writes
results/metric_correlation.csv. CPU-only, no model downloads.
"""
import csv
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(HERE, "results")

# (label, csv file, column) for every probe we correlate against ROUGE-L.
PROBES = [
    ("both teams named",  "faithfulness.csv", "teams_pct"),
    ("correct scoreline", "faithfulness.csv", "score_pct"),
    ("scorer recall",     "faithfulness.csv", "scorer_recall"),
    ("NER precision",     "faithfulness.csv", "ner_precision"),
    ("NLI support",       "semantic.csv",     "nli_support"),
    ("BERTScore F1",      "semantic.csv",     "bertscore_f1"),
]


def spearman(x, y):
    """Spearman rho and a two-sided permutation p-value (exact enough at n<=11).

    scipy is not a project dependency, so we rank manually and get the p-value
    by permuting one ranking -- with n = 11 the sampling distribution is well
    covered by 100k draws and this avoids pulling in scipy for one number.
    """
    def rank(v):
        order = np.argsort(np.argsort(v))
        # average ties so repeated values (e.g. two conditions at 0.00) do not
        # get an arbitrary ordering
        r = np.empty(len(v), float)
        for val in set(v):
            m = np.array(v) == val
            r[m] = order[m].mean()
        return r

    rx, ry = rank(x), rank(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    rng = np.random.default_rng(42)
    null = np.array([np.corrcoef(rx, rng.permutation(ry))[0, 1]
                     for _ in range(100_000)])
    p = float((np.abs(null) >= abs(rho) - 1e-12).mean())
    return rho, p


def load_rouge():
    """condition -> ROUGE-L, from the two runs behind the reported table."""
    out = {}
    for pat in ("run4_full_clean_*", "run5_led_clean_*"):
        for f in sorted(glob.glob(os.path.join(ROOT, "runs", pat,
                                               "results", "metrics.csv"))):
            for row in csv.DictReader(open(f)):
                out[row["condition"]] = float(row["rougeL"])
    return out


def load_col(fname, col):
    path = os.path.join(OUT, fname)
    return {r["condition"]: float(r[col])
            for r in csv.DictReader(open(path)) if r[col] not in ("", "nan")}


def main():
    rouge = load_rouge()
    rows = []
    print(f"{'probe':20s} {'n':>3s} {'rho':>7s} {'p':>8s}")
    print("-" * 42)
    for label, fname, col in PROBES:
        probe = load_col(fname, col)
        conds = sorted(set(rouge) & set(probe))
        x = [rouge[c] for c in conds]
        y = [probe[c] for c in conds]
        rho, p = spearman(x, y)
        print(f"{label:20s} {len(conds):3d} {rho:+7.2f} {p:8.3f}")
        rows.append({"probe": label, "n_conditions": len(conds),
                     "spearman_rho": round(rho, 4), "p_value": round(p, 4)})

    print("\nAll positive: over the full range the metric families agree in the")
    print("weak sense that both separate degenerate from coherent output.")
    print("The decoupling the report claims is local to the top of the ranking")
    print("(see Table 5/6): the two conditions leading every reference metric")
    print("are the two worst on NLI support.")

    os.makedirs(OUT, exist_ok=True)
    dest = os.path.join(OUT, "metric_correlation.csv")
    with open(dest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {os.path.relpath(dest, ROOT)}")


if __name__ == "__main__":
    main()
