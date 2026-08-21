#!/usr/bin/env python3
"""Null baselines and input-recoverability ceilings, computed post-hoc.

Two controls that bound how much of the reported ROUGE is match-specific
content rather than the fixed reference template and match-report register:

1. NULL BASELINES. Five "predictions" that contain zero correct information
   about the match they are scored against (lead-N excepted -- it is verbatim
   source text, the standard extractive baseline). Scored with the same
   stemmed rouge_score configuration as src/evaluation/metrics.py, on the same
   9 test matches. A paired bootstrap compares finetuned_led against the
   strongest null.

2. SCORER RECOVERABILITY. Whether the reference's scorers are present in the
   transcript at all, using the same fuzzy surname matcher that
   compute_analysis.py credits predictions with. This is the ceiling on scorer
   recall for any model reading only the transcript, and it separates
   transcription loss from extraction failure.

Outputs: results/nullbaselines.csv
Needs only numpy + rouge_score (no GPU, no model downloads).
"""
import json
import os
import random
import re
import unicodedata
from difflib import SequenceMatcher

import numpy as np
from rouge_score import rouge_scorer as rs

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA = os.path.join(ROOT, "football_commentary_dataset", "data")
LED_PRED = os.path.join(ROOT, "runs", "run5_led_clean_2026-05-04",
                        "predictions", "finetuned_led.json")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

SEED = 42
N_TRAIN_DONORS = 20     # random train references sampled per test match
N_BOOT = 10_000
LEAD_WORDS = 190        # mean reference length, so the lead baseline is length-matched

# A generic match report: correct template, correct register, no match facts.
TEMPLATE = """Team A 1-1 Team B
Premier League, Stadium, City - 1 January 2020

The match ended level after both sides created chances in an evenly contested \
game. The first half saw both teams struggle to break through, with possession \
shared and few clear opportunities on goal. The opening goal arrived after \
sustained pressure, converted from close range following a cross into the \
penalty area.

The second half was more open. The equaliser came midway through the half, a \
well-taken finish after good work down the flank. Both goalkeepers made \
important saves as the game stretched, and the defence held firm under late \
pressure to secure a share of the points.

Tactically, the home side dominated possession without turning it into clear \
chances, while the visitors defended deep and looked to counter-attack. The \
result leaves both teams with work to do.

Key events:
- 27' - Goal
- 68' - Goal
- 74' - Yellow card"""


def reference(match_id):
    with open(os.path.join(DATA, "summaries", match_id + ".txt"),
              encoding="utf-8") as fh:
        return fh.read().strip()


def transcript(match_id):
    with open(os.path.join(DATA, "transcripts", match_id + "_transcript.txt"),
              encoding="utf-8") as fh:
        return fh.read()


def team_names(match_id):
    """Two team names as they appear in the match id (last two fields)."""
    parts = match_id.split("_")
    return parts[-2].capitalize(), parts[-1].capitalize()


def norm(text):
    text = unicodedata.normalize("NFD", text)
    return "".join(c for c in text if unicodedata.category(c) != "Mn").lower()


def gold_scorers(match_id):
    """Scorer names from the reference's key-events block (as compute_analysis)."""
    with open(os.path.join(DATA, "summaries", match_id + ".txt"),
              encoding="utf-8") as fh:
        text = fh.read()
    scorers = []
    for line in text.splitlines():
        if not re.search(r"\b(goal|equaliser|equalizer|winner)\b", line, re.I):
            continue
        m = re.search(
            r"[–—-]\s*([A-ZÀ-Ž][\w.'’-]+(?:\s+[A-ZÀ-Ž][\w.'’-]+)*)\s*\(", line)
        if m and m.group(1).strip() not in scorers:
            scorers.append(m.group(1).strip())
    return scorers


def main():
    splits = json.load(open(os.path.join(ROOT, "outputs", "splits.json")))
    test, train = splits["test"], splits["train"]
    scorer = rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rng = random.Random(SEED)

    def score_against_refs(preds):
        """Mean R-1/R-2/R-L plus the per-match R-L list."""
        r1, r2, rl = [], [], []
        for m in test:
            s = scorer.score(reference(m), preds[m].strip())
            r1.append(s["rouge1"].fmeasure)
            r2.append(s["rouge2"].fmeasure)
            rl.append(s["rougeL"].fmeasure)
        return float(np.mean(r1)), float(np.mean(r2)), float(np.mean(rl)), rl

    def score_against_donors(donors_of):
        """Average over several donor references per test match."""
        per = {m: [] for m in test}
        for m in test:
            for d in donors_of(m):
                s = scorer.score(reference(m), reference(d))
                per[m].append((s["rouge1"].fmeasure, s["rouge2"].fmeasure,
                               s["rougeL"].fmeasure))
        means = {m: tuple(float(np.mean([p[i] for p in per[m]])) for i in range(3))
                 for m in test}
        rl = [means[m][2] for m in test]
        return (float(np.mean([means[m][0] for m in test])),
                float(np.mean([means[m][1] for m in test])),
                float(np.mean(rl)), rl)

    results = {}

    # 1. Another test match's reference: right format, right register, all facts wrong.
    results["cross-match reference (test donors)"] = score_against_donors(
        lambda m: [d for d in test if d != m])

    # 2. A random training reference: the same control at training-set scale.
    results["cross-match reference (train donors)"] = score_against_donors(
        lambda m: rng.sample(train, N_TRAIN_DONORS))

    # 3-4. Generic template, with and without the real team names.
    results["generic template + real team names"] = score_against_refs(
        {m: TEMPLATE.replace("Team A", team_names(m)[0])
                    .replace("Team B", team_names(m)[1]) for m in test})
    results["generic template (no match facts)"] = score_against_refs(
        {m: TEMPLATE for m in test})

    # 5. Lead-N words of the transcript: standard extractive baseline.
    results[f"lead-{LEAD_WORDS} words of transcript"] = score_against_refs(
        {m: " ".join(transcript(m).split()[:LEAD_WORDS]) for m in test})

    print(f"{'null baseline':44s} {'R-1':>7} {'R-2':>7} {'R-L':>7}")
    print("-" * 68)
    for name, (r1, r2, rl, _) in results.items():
        print(f"{name:44s} {r1:7.4f} {r2:7.4f} {rl:7.4f}")

    # ---- paired bootstrap: finetuned_led vs the strongest null --------------
    led = json.load(open(LED_PRED, encoding="utf-8"))
    led_rl = [scorer.score(reference(m), led[m])["rougeL"].fmeasure for m in test]
    best_null = max(results, key=lambda k: results[k][2])
    deltas = np.array(led_rl) - np.array(results[best_null][3])
    boot = np.random.default_rng(SEED)
    means = [boot.choice(deltas, size=len(deltas), replace=True).mean()
             for _ in range(N_BOOT)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    print("-" * 68)
    print(f"finetuned_led R-L {np.mean(led_rl):.4f} vs strongest null "
          f"({best_null}) {results[best_null][2]:.4f}")
    print(f"  paired delta {deltas.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
          f"  -> {'significant' if lo > 0 else 'NOT significant'}")
    print(f"  strongest null is {results[best_null][2] / np.mean(led_rl):.1%} "
          f"of the fine-tuned score")

    # ---- scorer recoverability ceiling -------------------------------------
    print("-" * 68)
    total = hits = exact = 0
    unrecoverable = []
    for m in test:
        tr = norm(transcript(m))
        tokens = set(re.findall(r"[a-z'’-]+", tr))
        for name in gold_scorers(m):
            total += 1
            n, surname = norm(name), norm(name).split()[-1]
            if n in tr or surname in tokens:
                hits += 1
                exact += 1
            elif any(SequenceMatcher(None, surname, t).ratio() >= 0.85
                     for t in tokens):
                hits += 1
            else:
                unrecoverable.append((m, name))
    print(f"reference scorers present in the transcript: {hits}/{total} "
          f"= {hits / total:.1%}  ({exact} exact, {hits - exact} fuzzy)")
    if unrecoverable:
        print(f"  unrecoverable: {unrecoverable}")
    print("  -> ceiling on scorer recall for any model reading the transcript")

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "nullbaselines.csv")
    with open(path, "w") as fh:
        fh.write("baseline,rouge1,rouge2,rougeL,contains_match_facts\n")
        for name, (r1, r2, rl, _) in results.items():
            facts = "source text" if name.startswith("lead") else "none"
            fh.write(f'"{name}",{r1:.4f},{r2:.4f},{rl:.4f},{facts}\n')
        fh.write(f'"finetuned_led (for reference)",0.4502,0.1521,'
                 f'{np.mean(led_rl):.4f},model output\n')
        fh.write(f'"scorer recoverability ceiling",,,{hits / total:.4f},'
                 f'"{hits}/{total} scorers present in transcript"\n')
    print(f"wrote {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
