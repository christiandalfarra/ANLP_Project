# Results — All Runs

Five Kaggle runs on T4 GPU, comparing prompting vs fine-tuning across BART/LED/FLAN-T5 on 99 football match transcripts (80 train / 10 val / 9 test).

> **Status.** Runs 1–3 were executed against a dataset in which 4 of the 9 test references were 0-byte files (see the caveat below). **Runs 4 and 5, on the repaired dataset, supersede them and are the numbers reported in `report/main.pdf`.** Everything below runs 4–5 is kept as an engineering log of how we got there — its numbers are historical, not final.

## Headline numbers — final (runs 4–5, clean dataset)

Test ROUGE-L over all 9 test matches. Sources: `run4_full_clean_2026-05-04/results/metrics.csv` (all conditions except LED fine-tuning) and `run5_led_clean_2026-05-04/results/metrics.csv` (finetuned_led).

| Rank | Condition        | ROUGE-1 | ROUGE-2 | **ROUGE-L** | Run |
|------|------------------|--------:|--------:|------------:|-----|
| 🥇   | finetuned_led    | 0.4502  | 0.1521  | **0.2480**  | run5 |
| 2    | finetuned_bart   | 0.4205  | 0.1330  | **0.2476**  | run4 |
| 3    | led_long_few     | 0.2712  | 0.0531  | 0.1599      | run4 |
| 4    | led_long_zero    | 0.2712  | 0.0531  | 0.1599      | run4 |
| 5    | led_long_cot     | 0.2773  | 0.0559  | 0.1548      | run4 |
| 6    | bart_chunk_zero  | 0.2249  | 0.0516  | 0.1467      | run4 |
| 7    | bart_chunk_cot   | 0.1989  | 0.0408  | 0.1230      | run4 |
| 8    | bart_chunk_few   | 0.1765  | 0.0188  | 0.1030      | run4 |
| 9    | flan_chunk_zero  | 0.1164  | 0.0212  | 0.0700      | run4 |
| 10   | flan_chunk_few   | 0.0770  | 0.0086  | 0.0551      | run4 |
| 11   | flan_chunk_cot   | 0.0733  | 0.0148  | 0.0481      | run4 |

The two fine-tuned pipelines are separated by 0.0004 ROUGE-L; the paired bootstrap difference is `-0.0004`, CI `[-0.042, +0.035]`, so at n=9 they are statistically indistinguishable. Every fine-tuned-vs-prompting gap *is* significant. Per-condition bootstrap CIs: `report/results/percondition_cis.csv`. BERTScore and NLI grounding: `report/results/semantic.csv`. Faithfulness probes: `report/results/faithfulness.csv`.

## Headline numbers — superseded (runs 1–3, poisoned dataset)

Kept for the record. These averages include 4 test matches whose reference was a 0-byte file, so every condition scores ~1.8× lower than it should.

| Rank | Condition                       | ROUGE-1 | ROUGE-2 | **ROUGE-L** | Run |
|------|---------------------------------|--------:|--------:|------------:|-----|
| 🥇   | finetuned_bart (chunk-merger)   | 0.2484  | 0.0820  | **0.1361**  | run2 |
| 2    | finetuned_bart (truncated 1024) | 0.2295  | 0.0767  | 0.1274      | run1 |
| 3    | finetuned_led                   | 0.2317  | 0.0693  | 0.1216      | run3 |
| 4    | led_long_cot                    | 0.1427  | 0.0262  | 0.0846      | run1 |
| 5    | led_long_few                    | 0.1377  | 0.0236  | 0.0825      | run1 |
| 6    | led_long_zero                   | 0.1377  | 0.0236  | 0.0825      | run1 |
| 7    | flan_chunk_zero                 | 0.0646  | 0.0154  | 0.0413      | run1 |
| 8    | flan_chunk_cot                  | 0.0468  | 0.0093  | 0.0329      | run1 |
| 9    | flan_chunk_few                  | 0.0377  | 0.0020  | 0.0278      | run1 |

## Important methodological caveat (runs 1–3): 4 of 9 test references were empty — FIXED in run 4

`football_commentary_dataset/data/summaries/` has **0-byte reference files** for:

- `2001_facup_arsenal_liverpool`
- `1983_facup_final_brighton_manchesterunited`
- `2024_euro_england_slovakia`
- `2026_ucl_athleticclub_arsenal`

ROUGE against an empty reference is exactly 0 for any non-empty prediction. The reported averages above include these 4 zeros, dragging every condition down by a factor of ~1.8x. The **corrected averages over the 5 valid test matches** are:

| Condition                       | Reported RL | **Corrected RL** (×9/5) |
|---------------------------------|------------:|------------------------:|
| finetuned_bart (chunk-merger)   | 0.1361      | **0.245** |
| finetuned_bart (truncated)      | 0.1274      | **0.229** |
| finetuned_led                   | 0.1216      | **0.219** |
| led_long_cot                    | 0.0846      | **0.152** |
| led_long_few/zero               | 0.0825      | **0.149** |
| flan_chunk_zero                 | 0.0413      | **0.074** |
| flan_chunk_cot                  | 0.0329      | **0.059** |
| flan_chunk_few                  | 0.0278      | **0.050** |

**Once you remove the empty-reference penalty, fine-tuned BART hits ROUGE-L ≈ 0.25 on the valid test set** — that's competitive with established long-document summarization benchmarks. The "modest" appearance of the original 0.136 was largely a data quality artifact.

**Resolved.** The four references were re-extracted and the full experiment grid was re-run on the repaired dataset (run 4, plus run 5 for LED). `python scripts/check_dataset.py` now passes on all 99 matches. The ×9/5 extrapolation above predicted the outcome almost exactly: fine-tuned BART landed at 0.2476 measured (0.245 predicted). The final table at the top of this file is measured, not extrapolated.

## What works

### 1. Fine-tuning beats prompting by a large margin
Both architectures benefit substantially:

- **LED**: 0.160 prompt (`led_long_few`, best prompting condition) → 0.248 fine-tune (**+55%**)
- **BART**: 0.147 pre-trained under the identical chunk-aggregate pipeline → 0.248 fine-tune (**+69%**) — a same-model, same-pipeline comparison that isolates the contribution of fine-tuning

(Runs 1–3 figures were LED 0.085 → 0.122 (+44%) and BART 0.136; the ranking of the conclusion did not change, only its magnitude.)

**Read those percentages against the null floor.** The references follow a fixed template, so a large share of ROUGE is available without knowing anything about the match (`report/compute_nullbaselines.py`):

| "Prediction" — contains no correct match facts | R-1 | R-2 | R-L |
|---|---:|---:|---:|
| Another test match's reference (avg. over 8) | 0.3793 | 0.0952 | **0.2076** |
| A random train reference (avg. over 20) | 0.3533 | 0.0864 | 0.1963 |
| Generic template + real team names | 0.3556 | 0.0680 | 0.1863 |
| Generic template, no match-specific content | 0.3481 | 0.0656 | 0.1758 |
| Lead-190 words of the transcript | 0.2624 | 0.0311 | 0.1340 |

The strongest null is **84% of finetuned_led's 0.2480**, and every prompting condition scores *below* a content-free template (best prompting = 0.1599). So "+55% over the best prompting baseline" mostly measures template acquisition. The margin that survives a format-matched control is **+0.040 ROUGE-L, paired bootstrap CI [+0.022, +0.059]** — significant, but a fifth of the headline. ROUGE-2 discriminates better than ROUGE-L here (nulls 0.066–0.095 vs 0.152), which also means early stopping on validation ROUGE-L was the wrong selection criterion.

### 2. Chunk-merger > naive truncation for BART
Run 2's redesign — train BART on `(concatenated pretrained-BART chunk-summaries → gold reference)` instead of `(truncated transcript[:1024] → gold reference)` — improved ROUGE-L from 0.127 → 0.136 (+7%) on the poisoned dataset, and the same recipe reaches **0.2476** on the repaired dataset (run 4). This matches what BART actually does at inference time, so train and test distributions are aligned.

### 3. LED training works once you fix the bugs
The LED retrain went through three failed attempts before producing a usable model:

- **v3**: lr=5e-5 + label smoothing → loss=91, grad_norm=209, ROUGE collapsed to 0 by epoch 3.
- **v5**: lr=2e-5 + no label smoothing + beam=2 eval → still collapsed by epoch 3.
- **v7**: bug fix in `run_finetuning.py` (was pre-truncating to 8192 tokens before random-window sampler, defeating the random sampling) + `min_length=80` floor → trained successfully through epoch 5 (val 0.125), then crashed in epoch 6 eval with `OverflowError`.
- **v9**: same code with checkpoint-N fallback in inference → recovered the epoch-5 model and got test 0.098.
- **v11**: real fix to `compute_rouge_metrics` (clip -100 sentinels in `preds` before `batch_decode`) → completed all 11 epochs cleanly. Val ROUGE-L peaked at 0.172 (epoch 9), test 0.122.

LED progression once stable:

| Epoch | val ROUGE-L | eval_loss |
|------:|------------:|----------:|
| 1     | 0.107       | 4.83 |
| 3     | 0.114       | 3.59 |
| 5     | 0.125       | 2.47 |
| 7     | 0.159       | 2.16 |
| **9** | **0.172**   | 1.85 |
| 11    | 0.162       | 1.63 |

**Run 5 (clean dataset, retrained from scratch):** the same code trained 14 epochs, validation ROUGE-L peaking at **0.270** at epoch 12 (early stopping, patience 2), evaluation loss decreasing monotonically. Selected checkpoint → test ROUGE-L **0.2480**. The run-3 curve above and the run-5 curve look nearly identical in loss; only ROUGE distinguishes them, which is exactly how the empty-reference bug stayed hidden. Figure: `report/figures/led_training_curve.pdf`.

## What doesn't work

### 1. FLAN-T5 + chunk-aggregate is broken
ROUGE-L of 0.03–0.04 across all three FLAN conditions. Looking at the actual outputs, FLAN-T5-Large is failing to summarize at the chunk level and instead echoing the prompt scaffolding:

> *flan_chunk_cot:* `"[Segment 19] Goals (with approximate time and scorer if mentioned), yellow/red cards"`
> *flan_chunk_few:* `"[Segment 13] Morocco and Spain played a tightly contested Round of 16 match that remained goalless..."` (wrong match — that's leaking from a few-shot example)

This is a classic FLAN failure mode: the prompt is too structured and the model treats it as a template-completion task instead of a summarization task. A FLAN-tuned prompt or a different model (e.g. Llama-3-8B-Instruct) would likely do better, but FLAN as configured here is non-functional.

### 2. LED prompting is mediocre even with long context
LED's 16k context window should help, but zero/few/CoT all land around 0.083 on the poisoned data (0.155–0.160 after the fix) — only ~2× FLAN, and less than two thirds of either fine-tuned model. The outputs reveal why: LED-base regurgitates the prompt or produces commentary-text that mimics input style instead of summarizing:

> *led_long_few:* `"...the ball is going to the right side of the penalty area, it's going to be difficult to get the ball back to the left side, there's a little bit of time for the ball to go in the right hand side..."`

LED-base wasn't pretrained on this kind of summarization → fine-tuning is needed to teach it the task, not just feed it more context.

### 3. Long-context vs chunk-merger fine-tuning: no separable winner (revised in run 5)
On the poisoned dataset (run 3) LED ended at 0.122 vs BART's 0.136, and we concluded chunk-merger won. **The clean retrain (run 5) overturned that**: LED reaches 0.2480 vs BART's 0.2476 — nominally ahead, and ahead on BERTScore too (0.869 vs 0.860), but the paired bootstrap CI of the difference (`[-0.042, +0.035]`) includes zero. At n=9 the two pipelines cannot be separated, and the honest conclusion is that both work and neither dominates.

What the run-3 gap actually measured was the dataset bug, not the architecture: LED was disproportionately penalised because it had trained on poisoned targets, while BART's chunk-merger inputs were pre-distilled and more robust to them.

The residual uncertainty still most likely comes down to sample efficiency:

- **N=80 is small for LED.** With 80 (input, output) pairs and 162M parameters, LED must learn to attend across raw 8k-token windows from scratch.
- **Pretrained-BART chunk summaries are themselves a strong feature.** The merger operates on a ~960-token pre-distilled intermediate whose features are already informative.

LED's narrower CI (`[0.230, 0.266]` vs BART's `[0.208, 0.283]`) suggests more consistent per-match performance; BART's wider CI reflects higher variance with a higher ceiling.

## Sample outputs — runs 1–3 (test match: 2001 FA Cup Final, Arsenal vs Liverpool)

Reference: empty file at the time (the data quality issue below), so qualitative judgment only. These are the pre-fix models' outputs; for samples from the final run-4/run-5 models against a real reference, see Table 6 of `report/main.pdf`.

| Model | Output (first 250 chars) | Words |
|-------|--------------------------|------:|
| flan_chunk_zero | "Mark Warburton and Jamie Vardy contributed to BBC Radio 5 live coverage of Arsenal drew 1-1 at Anfield on Saturday." | 28 |
| led_long_zero   | "ope, and now to his right, the ball is going to the right side of the penalty area, it's going to be difficult to get the ball back to the left side..." | 215 |
| **finetuned_bart (run2)** | "Liverpool 1–1 Arsenal (Liverpool win 2–3 on penalties) 20 May 2014 – Emirates Stadium, London, London. The match was contested between Arsenal and Liverpool in the 2014 FA Cup Final..." | 112 |
| **finetuned_led** | "Liverpool 1–0 Arsenal, Premier League, 20 May 2001 – Anfield, Anfield, Emirates Stadium, Anfield – Anfield – 19 May 2001. Liverpool dominated the first half, dominating possession and controlling the game..." | 198 |

**What this shows:**

- **FLAN** generates short, stylistically-wrong (radio-broadcast snippets), and factually wrong content.
- **LED prompting** generates long but is essentially a slightly-summarized version of the input commentary stream. No structure, no events.
- **Fine-tuned BART** generates structured match-report prose with team names, score format, date format. **Hallucinates everything specific** — wrong year, wrong stadium, wrong scorer would all appear if the model knew them.
- **Fine-tuned LED** also generates match-report prose. **Gets the year right** for this match (2001) but loses control on specifics — repeats "Anfield" 4 times, wrong score (was 2-1 to Liverpool, not 1-0).

Both fine-tuned models have learned the **format** of a match summary but neither has learned **factual fidelity**. They produce confidently-presented hallucinations. ROUGE rewards them anyway because surface n-grams (team names, "in the Xth minute", dates, stadium types) overlap between predicted and actual summaries on the same matches.

## Limitations — status after runs 4–5

1. ~~**Fix the empty-reference data files.**~~ **Done (run 4).** All four 0-byte references were re-extracted; `scripts/check_dataset.py` verifies every one of the 99 matches has a reference ≥ 200 B. The whole grid was re-run on the repaired data, and ROUGE-L roughly doubled as the ×9/5 extrapolation predicted.
2. **Hallucination is unsolved — and now quantified.** Rule-based faithfulness probes plus NLI entailment against the transcript (`report/results/faithfulness.csv`, `report/results/semantic.csv`) show fine-tuned BART states the correct final score in **0 of 9** matches with **zero** NLI-supported sentences, and fine-tuned LED manages 2/9 scorelines with 2% support — despite leading on both ROUGE-L and BERTScore. The best-grounded conditions (34% NLI support) are the transcript-copying prompted baselines that ROUGE ranks near the bottom. Surface metrics and source faithfulness are fully decoupled on this task. Reducing the hallucination itself remains open. Note this is *not* a transcription-loss problem: all 22 reference scorers across the 9 test matches are present in the transcripts the models read (21 exact, 1 fuzzy), so the ceiling on scorer recall is 100% and the models realise 30% / 13% of it.
3. **N=80 is still the binding constraint.** Both models overfit; with 9 test matches, the two fine-tuned pipelines cannot be statistically separated. More data remains the highest-leverage improvement.
4. ~~**No BERTScore.**~~ **Done.** Computed locally with a pinned `bert_score` outside the Kaggle environment (`report/compute_semantic.py` → `report/results/semantic.csv`): fine-tuned LED 0.869, fine-tuned BART 0.860, all prompting conditions 0.79–0.82. BERTScore agrees with the ROUGE ranking.
5. ~~**No statistical comparison.**~~ **Done.** Paired bootstrap CIs per condition (`report/compute_analysis.py` → `report/results/percondition_cis.csv`). Every fine-tuning-vs-prompting gap excludes zero; the LED-vs-BART gap does not.
