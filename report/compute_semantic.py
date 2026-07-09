#!/usr/bin/env python3
"""Semantic quality metrics computed locally, post-hoc, on saved predictions:

1. BERTScore F1 (roberta-large, raw) of each prediction against its reference.
2. NLI support rate against the *transcript*: split each prediction into
   sentences, retrieve the top-3 most similar transcript windows by TF-IDF,
   score entailment with an MNLI cross-encoder, and call a sentence
   "supported" when max P(entailment) >= 0.5. Reported per condition as the
   macro-averaged fraction of supported sentences.

Outputs: results/semantic.csv
"""
import json
import os
import re

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA = os.path.join(ROOT, "football_commentary_dataset", "data")
RUN4 = os.path.join(ROOT, "runs", "run4_full_clean_2026-05-04", "predictions")
RUN3_LED = os.path.join(ROOT, "runs", "run3_led_finetuned_2026-05-03",
                        "predictions", "finetuned_led.json")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

NLI_MODEL = "microsoft/deberta-base-mnli"
WINDOW, STRIDE, TOP_K, THRESH = 60, 30, 3, 0.5
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def sentences(text, max_words=50):
    """Split into sentences; further chunk run-ons (LED emits comma chains)."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    out = []
    for p in parts:
        words = p.split()
        if len(words) < 4:
            continue
        for i in range(0, len(words), max_words):
            chunk = " ".join(words[i:i + max_words])
            if len(chunk.split()) >= 4:
                out.append(chunk)
    return out


def transcript_windows(match_id):
    path = os.path.join(DATA, "transcripts", match_id + "_transcript.txt")
    words = open(path, encoding="utf-8").read().split()
    return [" ".join(words[i:i + WINDOW])
            for i in range(0, max(1, len(words) - WINDOW + 1), STRIDE)]


def main():
    splits = json.load(open(os.path.join(ROOT, "outputs", "splits.json")))
    test = splits["test"]
    refs = {m: open(os.path.join(DATA, "summaries", m + ".txt"),
                    encoding="utf-8").read().strip() for m in test}

    preds = {}
    for f in sorted(os.listdir(RUN4)):
        preds[f[:-5]] = json.load(open(os.path.join(RUN4, f), encoding="utf-8"))
    preds["finetuned_led"] = json.load(open(RUN3_LED, encoding="utf-8"))
    conds = list(preds)

    # ---------- BERTScore vs reference -----------------------------------
    from bert_score import score as bs
    cands, golds, keys = [], [], []
    for c in conds:
        for m in test:
            cands.append(preds[c][m])
            golds.append(refs[m])
            keys.append((c, m))
    print(f"BERTScore on {len(cands)} pairs ({DEVICE}) ...")
    _, _, F1 = bs(cands, golds, lang="en", device=DEVICE, batch_size=16,
                  verbose=False)
    bert_f1 = {}
    for (c, m), f1 in zip(keys, F1.tolist()):
        bert_f1.setdefault(c, []).append(f1)

    # ---------- NLI support vs transcript --------------------------------
    tok = AutoTokenizer.from_pretrained(NLI_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
    model.to(DEVICE).eval()
    ent_idx = next(i for i, l in model.config.id2label.items()
                   if l.upper().startswith("ENTAIL"))
    print(f"NLI model {NLI_MODEL} on {DEVICE}; entailment index {ent_idx}")

    windows = {m: transcript_windows(m) for m in test}
    tfidf = {m: TfidfVectorizer(stop_words="english").fit(windows[m])
             for m in test}
    win_vecs = {m: tfidf[m].transform(windows[m]) for m in test}

    @torch.no_grad()
    def entail_probs(premises, hypothesis):
        enc = tok(premises, [hypothesis] * len(premises), truncation=True,
                  max_length=256, padding=True, return_tensors="pt").to(DEVICE)
        logits = model(**enc).logits
        return torch.softmax(logits, dim=-1)[:, ent_idx].tolist()

    support = {}
    for c in conds:
        rates = []
        for m in test:
            sents = sentences(preds[c][m])
            if not sents:
                rates.append(0.0)
                continue
            sv = tfidf[m].transform(sents)
            sims = sv @ win_vecs[m].T                      # (n_sent, n_win)
            supported = 0
            for i, s in enumerate(sents):
                row = sims[i].toarray().ravel()
                top = row.argsort()[-TOP_K:]
                probs = entail_probs([windows[m][j] for j in top], s)
                if max(probs) >= THRESH:
                    supported += 1
            rates.append(supported / len(sents))
        support[c] = rates
        print(f"  {c:22s} NLI support {np.mean(rates):.3f}  "
              f"BERTScore-F1 {np.mean(bert_f1[c]):.4f}")

    os.makedirs(OUT, exist_ok=True)
    order = sorted(conds, key=lambda c: -np.mean(bert_f1[c]))
    with open(os.path.join(OUT, "semantic.csv"), "w") as fh:
        fh.write("condition,bertscore_f1,nli_support\n")
        for c in order:
            fh.write(f"{c},{np.mean(bert_f1[c]):.4f},"
                     f"{np.mean(support[c]):.4f}\n")
    print("wrote results/semantic.csv")


if __name__ == "__main__":
    main()
