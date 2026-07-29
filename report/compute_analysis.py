#!/usr/bin/env python3
"""Post-hoc analysis for the report:

1. Per-match ROUGE for every condition (run4 predictions + run3 finetuned_led)
   against the clean references, so finetuned_led gets a true 9-match number.
2. Bootstrap 95% CIs on ROUGE-1/2/L per condition, plus paired bootstrap of
   the ROUGE-L difference vs finetuned_bart.
3. Faithfulness: per prediction, check (a) both teams mentioned, (b) correct
   oriented scoreline, (c) recall of gold scorers. Gold facts are parsed from
   the reference score line and "Key events" block.

Outputs: results/percondition_cis.csv, results/faithfulness.csv, and
LaTeX-ready rows on stdout.
"""
import json
import os
import re
import unicodedata
from difflib import SequenceMatcher

import numpy as np
from rouge_score import rouge_scorer

# spaCy is loaded lazily so the rest of the script still runs if it is absent.
_NLP = None


def _nlp():
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load("en_core_web_sm")
    return _NLP

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SUMM = os.path.join(ROOT, "football_commentary_dataset", "data", "summaries")
RUN4 = os.path.join(ROOT, "runs", "run4_full_clean_2026-05-04", "predictions")


def _resolve_led_pred():
    """Prefer a clean LED retrain (run5) if present, else fall back to run3.
    See report/RUNBOOK_led_retrain.md."""
    import glob
    hits = sorted(glob.glob(os.path.join(
        ROOT, "runs", "run5_led_clean_*", "predictions", "finetuned_led.json")))
    if hits:
        return hits[-1]
    return os.path.join(ROOT, "runs", "run3_led_finetuned_2026-05-03",
                        "predictions", "finetuned_led.json")


RUN3_LED = _resolve_led_pred()
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

N_BOOT = 10_000
SEED = 42

# Team aliases per test match, in reference score-line order (home, away).
ALIASES = {
    "2001_facup_arsenal_liverpool": (["arsenal"], ["liverpool"]),
    "2021_euro_england_ukraine": (["ukraine"], ["england"]),
    "2019_uclfinal_liverpool_tottenham": (["liverpool"], ["tottenham", "spurs"]),
    "2022_wc_poland_argentina": (["poland"], ["argentina"]),
    "1983_facup_final_brighton_manchesterunited":
        (["brighton"], ["manchester united", "united", "man utd"]),
    "2024_euro_england_slovakia": (["england"], ["slovakia"]),
    "2026_ucl_athleticclub_arsenal": (["athletic", "bilbao"], ["arsenal"]),
    "2022_wc_england_senegal": (["england"], ["senegal"]),
    "2021_euro_czechrepublic_england": (["czech"], ["england"]),
}


def norm(text):
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text.lower()


def parse_gold(match_id):
    """Return (scoreA, scoreB, scorer_names) from the reference."""
    path = os.path.join(SUMM, match_id + ".txt")
    text = open(path, encoding="utf-8").read()
    first = text.strip().splitlines()[0]
    m = re.search(r"(\d+)\s*[–—-]\s*(\d+)", first)
    assert m, f"no score line in {match_id}: {first!r}"
    score = (int(m.group(1)), int(m.group(2)))

    scorers = []
    for line in text.splitlines():
        if not re.search(r"\b(goal|equaliser|equalizer|winner)\b", line, re.I):
            continue
        em = re.search(
            r"[–—-]\s*([A-ZÀ-Ž][\w.'’-]+(?:\s+[A-ZÀ-Ž][\w.'’-]+)*)\s*\(", line)
        if em:
            name = em.group(1).strip()
            if name not in scorers:
                scorers.append(name)
    return score, scorers


def fuzzy_token_match(surname, pred_tokens, threshold=0.85):
    return any(SequenceMatcher(None, surname, t).ratio() >= threshold
               for t in pred_tokens)


def scorer_recall(gold_scorers, pred_norm):
    """Fraction of gold scorers mentioned (surname, fuzzy for Whisper noise)."""
    if not gold_scorers:
        return None
    pred_tokens = re.findall(r"[a-z'’-]+", pred_norm)
    hits = 0
    for name in gold_scorers:
        n = norm(name)
        surname = n.split()[-1]
        if n in pred_norm or surname in pred_tokens \
                or fuzzy_token_match(surname, pred_tokens):
            hits += 1
    return hits / len(gold_scorers)


def team_before(pred_norm, pos, aliases_a, aliases_b, window=30):
    ctx = pred_norm[max(0, pos - window):pos]
    best, side = -1, None
    for side_name, aliases in (("A", aliases_a), ("B", aliases_b)):
        for a in aliases:
            i = ctx.rfind(a)
            if i > best:
                best, side = i, side_name
    return side


def team_after(pred_norm, pos, aliases_a, aliases_b, window=30):
    ctx = pred_norm[pos:pos + window]
    best, side = len(ctx) + 1, None
    for side_name, aliases in (("A", aliases_a), ("B", aliases_b)):
        for a in aliases:
            i = ctx.find(a)
            if i != -1 and i < best:
                best, side = i, side_name
    return side


def score_correct(gold_score, aliases, pred, pred_norm):
    """True if the prediction states the gold scoreline with correct team
    orientation. Accepted forms: 'TeamX d1-d2 TeamY' (adjacent teams),
    'TeamX beat/defeated ... d1-d2' with TeamX the winner, or any multiset
    match when the gold result is a draw."""
    sa, sb = gold_score
    al_a, al_b = aliases
    counts = {"A": sa, "B": sb}
    for m in re.finditer(r"\b(\d{1,2})\s*[–—-]\s*(\d{1,2})\b", pred_norm):
        d1, d2 = int(m.group(1)), int(m.group(2))
        if sorted((d1, d2)) != sorted((sa, sb)):
            continue
        if sa == sb:                       # draw: orientation is symmetric
            return True
        before = team_before(pred_norm, m.start(), al_a, al_b)
        after = team_after(pred_norm, m.end(), al_a, al_b)
        if before and counts[before] == d1:
            return True
        if after and counts[after] == d2:
            return True
        # 'X beat/defeated Y d1-d2' → the verb's subject scored d1
        ctx = pred_norm[max(0, m.start() - 90):m.start()]
        vm = re.search(r"\b(beat|defeated|overcame|saw off|won)\b", ctx)
        if vm and d1 > d2:
            subj = team_before(ctx + " ", vm.start(), al_a, al_b, window=60)
            if subj and counts[subj] == d1:
                return True
    return False


def both_teams(aliases, pred_norm):
    return all(any(a in pred_norm for a in side) for side in aliases)


# ---- named-entity precision -----------------------------------------------
# Complements scorer *recall* (Table 5). Recall cannot see over-generation: a
# prediction that names ten wrong players plus one right one still scores well
# on recall. Precision = of the distinct PERSONs the model actually names, what
# fraction appear among the reference's people (its scorers and any person named
# in the report). Low precision = invented names, the hallucination signature.

def person_surnames(text):
    """Distinct normalised surnames of PERSON entities in `text` (spaCy)."""
    out = set()
    for ent in _nlp()(text).ents:
        if ent.label_ != "PERSON":
            continue
        toks = [t for t in norm(ent.text).split() if len(t) > 1]
        if toks:
            out.add(toks[-1])          # surname = last token
    return out


def ref_person_surnames(match_id, gold_scorers, ref_text):
    """Reference people: gold scorers plus every PERSON in the reference report."""
    surnames = set(person_surnames(ref_text))
    for name in gold_scorers:
        surnames.add(norm(name).split()[-1])
    return surnames


def ner_precision(pred_text, ref_surnames, threshold=0.85):
    """(precision, n_named). None when the prediction names nobody (undefined)."""
    pred = person_surnames(pred_text)
    if not pred:
        return None, 0
    hits = 0
    for s in pred:
        if s in ref_surnames or any(
                SequenceMatcher(None, s, r).ratio() >= threshold
                for r in ref_surnames):
            hits += 1
    return hits / len(pred), len(pred)


def main():
    splits = json.load(open(os.path.join(ROOT, "outputs", "splits.json")))
    test = splits["test"]
    refs = {m: open(os.path.join(SUMM, m + ".txt"), encoding="utf-8")
            .read().strip() for m in test}
    gold = {m: parse_gold(m) for m in test}

    preds = {}
    for f in sorted(os.listdir(RUN4)):
        preds[f[:-5]] = json.load(open(os.path.join(RUN4, f), encoding="utf-8"))
    preds["finetuned_led"] = json.load(open(RUN3_LED, encoding="utf-8"))

    rs = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"],
                                  use_stemmer=True)

    # ---- per-match ROUGE -------------------------------------------------
    per_match = {}      # cond -> np.array shape (9, 3)
    for cond, p in preds.items():
        rows = []
        for m in test:
            s = rs.score(refs[m], p[m])
            rows.append([s["rouge1"].fmeasure, s["rouge2"].fmeasure,
                         s["rougeL"].fmeasure])
        per_match[cond] = np.array(rows)

    # sanity: run4 aggregates must reproduce metrics.csv
    import csv
    with open(os.path.join(ROOT, "runs", "run4_full_clean_2026-05-04",
                           "results", "metrics.csv")) as fh:
        for row in csv.DictReader(fh):
            got = per_match[row["condition"]].mean(axis=0)
            exp = [float(row["rouge1"]), float(row["rouge2"]),
                   float(row["rougeL"])]
            assert np.allclose(got, exp, atol=1e-6), \
                f"mismatch {row['condition']}: {got} vs {exp}"
    print("sanity check vs run4 metrics.csv: OK\n")

    # ---- bootstrap -------------------------------------------------------
    rng = np.random.default_rng(SEED)
    n = len(test)
    idx = rng.integers(0, n, size=(N_BOOT, n))

    cis = {}
    for cond, arr in per_match.items():
        boot = arr[idx].mean(axis=1)               # (N_BOOT, 3)
        cis[cond] = {
            "mean": arr.mean(axis=0),
            "lo": np.percentile(boot, 2.5, axis=0),
            "hi": np.percentile(boot, 97.5, axis=0),
        }

    # paired bootstrap of ROUGE-L diff vs finetuned_bart (same resamples)
    base = per_match["finetuned_bart"][:, 2]
    paired = {}
    for cond, arr in per_match.items():
        if cond == "finetuned_bart":
            continue
        diff = base - arr[:, 2]
        boot = diff[idx].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        paired[cond] = (diff.mean(), lo, hi, lo > 0)

    # reference people per match (spaCy PERSONs + gold scorers), computed once
    ref_people = {m: ref_person_surnames(m, gold[m][1], refs[m]) for m in test}

    # ---- faithfulness ----------------------------------------------------
    faith = {}
    for cond, p in preds.items():
        teams_ok, score_ok, recalls, precisions, n_named = [], [], [], [], []
        for m in test:
            pn = norm(p[m])
            teams_ok.append(both_teams(ALIASES[m], pn))
            score_ok.append(score_correct(gold[m][0], ALIASES[m], p[m], pn))
            r = scorer_recall(gold[m][1], pn)
            if r is not None:
                recalls.append(r)
            prec, nn = ner_precision(p[m], ref_people[m])
            n_named.append(nn)
            if prec is not None:
                precisions.append(prec)
        faith[cond] = (np.mean(teams_ok), np.mean(score_ok), np.mean(recalls),
                       np.mean(precisions) if precisions else float("nan"),
                       np.mean(n_named))

    # ---- write outputs ---------------------------------------------------
    order = sorted(per_match, key=lambda c: -per_match[c][:, 2].mean())

    with open(os.path.join(OUT, "percondition_cis.csv"), "w") as fh:
        fh.write("condition,rouge1,rouge1_lo,rouge1_hi,rouge2,rouge2_lo,"
                 "rouge2_hi,rougeL,rougeL_lo,rougeL_hi,"
                 "dRL_vs_bart,dRL_lo,dRL_hi,significant\n")
        for c in order:
            d = cis[c]
            p = paired.get(c, (float("nan"),) * 3 + ("",))
            fh.write(f"{c},"
                     + ",".join(f"{d['mean'][i]:.4f},{d['lo'][i]:.4f},"
                                f"{d['hi'][i]:.4f}" for i in range(3))
                     + f",{p[0]:.4f},{p[1]:.4f},{p[2]:.4f},{p[3]}\n")

    with open(os.path.join(OUT, "faithfulness.csv"), "w") as fh:
        fh.write("condition,teams_pct,score_pct,scorer_recall,"
                 "ner_precision,avg_people_named\n")
        for c in order:
            t, s, r, pr, nn = faith[c]
            fh.write(f"{c},{t:.4f},{s:.4f},{r:.4f},{pr:.4f},{nn:.2f}\n")

    # ---- console tables --------------------------------------------------
    print(f"{'condition':22s} {'RL mean':>8s} {'95% CI':>17s} "
          f"{'ΔRL vs bart':>12s} {'sig':>4s}")
    for c in order:
        d = cis[c]
        line = (f"{c:22s} {d['mean'][2]:8.4f} "
                f"[{d['lo'][2]:.3f}, {d['hi'][2]:.3f}]")
        if c in paired:
            pm = paired[c]
            line += f" {pm[0]:+.3f} [{pm[1]:+.3f},{pm[2]:+.3f}]" \
                    + ("   *" if pm[3] else "")
        print(line)

    print(f"\n{'condition':22s} {'teams%':>7s} {'score%':>7s} "
          f"{'scorers%':>9s} {'NERprec%':>9s} {'#named':>7s}")
    for c in order:
        t, s, r, pr, nn = faith[c]
        print(f"{c:22s} {t*100:6.0f}% {s*100:6.0f}% {r*100:8.0f}% "
              f"{pr*100:8.0f}% {nn:7.1f}")

    # per-match detail for the two fine-tuned conditions (manual audit)
    print("\n--- audit: per-match extraction (finetuned_bart) ---")
    for m in test:
        pn = norm(preds["finetuned_bart"][m])
        sm = re.search(r"\b\d{1,2}\s*[–—-]\s*\d{1,2}\b",
                       preds["finetuned_bart"][m])
        print(f"{m[:44]:46s} gold={gold[m][0]} "
              f"pred_first_score={sm.group(0) if sm else None!r:12} "
              f"score_ok={score_correct(gold[m][0], ALIASES[m], preds['finetuned_bart'][m], pn)} "
              f"scorers={scorer_recall(gold[m][1], pn)}")


if __name__ == "__main__":
    main()
