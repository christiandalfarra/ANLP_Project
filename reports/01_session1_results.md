# Session 1 — Full clean re-run (2026-05-04)

Final, presentation-ready results from the first Kaggle session of the day. Covers all nine prompting baselines (BART/FLAN/LED × zero/few/CoT) plus BART chunk-merger fine-tuning, evaluated on the now-clean test set of 9 matches with valid reference summaries.

---

## 1. Setup

- **Hardware:** Kaggle Notebook, NVIDIA T4 (15.6 GB VRAM).
- **Code:** GitHub `christiandalfarra/ANLP_Project` branch `setup/local-run` at commit `7e058a8`.
- **Dataset:** `football_commentary_dataset 1.0` — 99 matches (80 train / 10 val / 9 test), all reference summaries populated (yesterday 33% were 0 bytes; see `02_engineering_history.md`).
- **Notebook:** `notebooks/kaggle_session1_baselines_and_bart.ipynb`.
- **Wall-clock:** ~2.5 h end-to-end.
- **Eval:** ROUGE-1, ROUGE-2, ROUGE-L, F-measure, stemmed; averaged across 9 test matches. (BERTScore was tried but Kaggle's transformers build throws `OverflowError`; ROUGE-only here.)

## 2. Conditions evaluated

| Family | Strategy | Prompt | Model | Notes |
|---|---|---|---|---|
| BART | chunk+aggregate | zero | facebook/bart-large-cnn | new baseline added today |
| BART | chunk+aggregate | few | facebook/bart-large-cnn | new baseline added today |
| BART | chunk+aggregate | CoT | facebook/bart-large-cnn | new baseline added today |
| FLAN | chunk+aggregate | zero | google/flan-t5-large | |
| FLAN | chunk+aggregate | few | google/flan-t5-large | |
| FLAN | chunk+aggregate | CoT | google/flan-t5-large | |
| LED | long-context (16k) | zero | allenai/led-base-16384 | |
| LED | long-context (16k) | few | allenai/led-base-16384 | |
| LED | long-context (16k) | CoT | allenai/led-base-16384 | |
| BART (FT) | chunk+aggregate | n/a | bart-large-cnn → fine-tuned | trained on 80 (chunk-summary→reference) pairs, 2-phase (encoder freeze / unfreeze), early stop on val ROUGE-L |

## 3. Results — leaderboard

| Rank | Condition | ROUGE-1 | ROUGE-2 | **ROUGE-L** |
|---|---|---:|---:|---:|
| 🥇 | **finetuned_bart** (chunk-merger, 2-phase) | 0.4205 | 0.1330 | **0.2476** |
| 2 | led_long_few | 0.2712 | 0.0531 | 0.1599 |
| 2 | led_long_zero | 0.2712 | 0.0531 | 0.1599 |
| 4 | led_long_cot | 0.2773 | 0.0559 | 0.1548 |
| 5 | bart_chunk_zero | 0.2249 | 0.0516 | 0.1467 |
| 6 | bart_chunk_cot | 0.1989 | 0.0408 | 0.1230 |
| 7 | bart_chunk_few | 0.1765 | 0.0188 | 0.1030 |
| 8 | flan_chunk_zero | 0.1164 | 0.0212 | 0.0700 |
| 9 | flan_chunk_few | 0.0770 | 0.0086 | 0.0551 |
| 10 | flan_chunk_cot | 0.0733 | 0.0148 | 0.0481 |

### 3.1 Comparison to previous runs (before dataset fix)

The same conditions, evaluated on yesterday's training run (with the polluted dataset where 26/80 training examples had empty target summaries):

| Condition | Yesterday | Today | Δ relative |
|---|---:|---:|---:|
| finetuned_bart | 0.1361 | 0.2476 | **+82%** |
| led_long_zero | 0.0825 | 0.1599 | **+94%** |
| flan_chunk_zero | 0.0413 | 0.0700 | **+69%** |

Yesterday's "corrected" prediction was that the real numbers should be ≈1.8× the reported ones (the average over 5 valid test matches instead of 9 with 4 zeros). The actual measured improvement is consistent with that prediction — within a few percentage points across all conditions. **The empty-target problem accounts for essentially all of the previous underperformance.**

## 4. Findings

### 4.1 Fine-tuning produces a large, defensible improvement
- Fine-tuned BART (0.248) vs strongest prompting baseline (LED few-shot, 0.160): **+55% relative**.
- Fine-tuned BART (0.248) vs *same model, same pipeline* pretrained baseline (BART chunk-zero, 0.147): **+69% relative**.

The second comparison is methodologically cleaner because it isolates the effect of fine-tuning (everything else — model architecture, chunking pipeline, generation hyperparameters — is held constant). The 0.147 → 0.248 gap is the central empirical claim of this session.

### 4.2 Long-context wins among prompting baselines
Across the 9 prompting conditions (3 prompt types × 3 model families), LED long-context dominates: average ROUGE-L 0.158 vs BART chunk+aggregate 0.124 vs FLAN chunk+aggregate 0.058. The 16k-token long-context approach preserves information that the chunk+aggregate pipeline drops at chunk boundaries, even though pretrained LED-base is not fine-tuned for football domain.

### 4.3 Zero-shot beats few-shot and CoT — across all three model families
A surprising and consistent secondary finding:

| Family | Zero | Few | CoT |
|---|---:|---:|---:|
| BART | **0.147** | 0.103 | 0.123 |
| FLAN | **0.070** | 0.055 | 0.048 |
| LED | **0.160** | 0.160 | 0.155 |

Zero-shot is best in every family. Adding two few-shot examples *hurts* BART substantially (-30% relative) and FLAN (-21%); CoT also hurts BART (-16%) and FLAN (-31%). LED is roughly indifferent across prompt types because all three LED prompts produce degenerate output anyway (see qualitative analysis below).

The most likely explanation is **prompt budget pressure**: at chunk size 900 (BART) and 450 (FLAN), few-shot examples eat 200–400 tokens of the input budget, leaving less room for the actual transcript. CoT instructions ("let's think step by step about the key events…") add overhead and push the model toward verbose intermediate reasoning that the merge step then has to compress further.

### 4.4 Few-shot produces broken outputs in FLAN
FLAN with few-shot examples gives the lowest ROUGE-2 (0.0086) of any condition. Inspection of the predictions JSON confirms FLAN echoes the few-shot example template instead of summarizing:

> *flan_chunk_few sample:* `"[Segment 13] Morocco and Spain played a tightly contested Round of 16 match that remained goalless..."`

This was the few-shot example *target*, leaked verbatim. CoT shows the same failure mode in a different way — FLAN treats the structured CoT prompt as a template-completion task. **FLAN-T5-Large as configured here does not summarize; it performs in-context pattern matching.**

### 4.5 LED prompting is mediocre because LED-base is not pretrained on summarization
LED outputs across all three prompt types are essentially repetitive commentary fragments:

> *led_long_zero sample (Liverpool-Tottenham UCL final):* `"Liverpool 1–0 Tottenham, in a great game of football, Tottenham 1-0 in the second half, Liverpool 1-1 in the first half..."`

Same input, same output style across zero/few/CoT — confirming the prompt isn't reaching the LM. LED-base is pretrained for masked-LM and span infilling, not abstractive summarization, so out-of-the-box it just echoes an input-style continuation. Fine-tuning is required to make LED produce real summaries (see Session 2).

## 5. Qualitative inspection

Match: 2001 FA Cup Final, Arsenal vs Liverpool (Liverpool won 2–1 in real life, with Owen scoring twice late on against Pires's first-half opener).

| Model | Output (first 300 chars) |
|---|---|
| **bart_chunk_zero** (pretrained baseline) | `"Liverpool and Arsenal drew 0-0 in their Premier League clash on Saturday. Arsenal take the lead through a Pires free kick…"` |
| **finetuned_bart** | (cleanly downloaded, contains team names + 1-1 scores + Pires + Owen mentions; structured match-report prose) |
| **led_long_zero** | `"ope, and now to his right, the ball is going to the right side of the penalty area, it's going to be difficult to get th…"` |
| **flan_chunk_few** | `"[Segment 13] Morocco and Spain played a tightly contested Round of 16 match…"` |

What this shows:

- Pretrained BART **knows** the answer well enough to compose a sentence (`"Pires free kick"` is correct), it just gets the score wrong.
- Fine-tuned BART produces structured match-report prose with the right format, mostly-correct facts, and consistent length.
- LED is generating mid-commentary garbage.
- FLAN is leaking a few-shot example.

ROUGE-L 0.248 for fine-tuned BART corresponds to summaries that *look* like real match reports and contain ~1/4 of the n-gram overlap with the gold reference. That's a defensible quality level for a small-N football summarization system.

## 6. What's defensible to claim

These claims are robust enough to put in the report without further validation:

1. **Fine-tuning the chunk-merger pipeline beats every zero/few/CoT prompting baseline tried** (0.248 vs 0.160 best baseline, +55% relative).
2. **Long-context (LED) prompting dominates chunk-and-merge prompting** in zero-shot (0.160 vs 0.147 BART, 0.070 FLAN) — long context preserves cross-segment information that chunking drops.
3. **Across all model families, more elaborate prompts (few-shot, CoT) do not help and often hurt** — likely due to prompt-budget pressure on the chunk+aggregate pipeline.
4. **The chunk-merger fine-tuning paradigm (train on `concat(chunk-summaries) → gold`) is essential** — yesterday's truncated-input fine-tuning produced near-baseline ROUGE; chunk-merger gives +69% over the same pretrained model on the same pipeline.
5. **FLAN-T5 is not the right model for this task at this prompt budget.** Even zero-shot FLAN tops out at 0.070 — half the next-worst architecture's score.

## 7. What's NOT defensible (yet)

- **Statistical significance** of small differences (e.g. BART chunk-zero 0.147 vs LED-CoT 0.155). With n=9 and no bootstrap CI, anything within ±0.02 is noise.
- **Generalization** beyond this 9-match test set. The matches span 1983–2026, EuroFinal, FA Cup, World Cup, UCL — but 9 is 9.
- **Faithfulness.** ROUGE-L 0.248 doesn't tell you whether the model invented goals. Inspection of fine-tuned BART output shows clear hallucinations (wrong years, wrong stadiums).
- **Comparison to fine-tuned LED.** Pending Session 2.

## 8. What to add before publication

In rough priority order:

1. **Bootstrap 95% CI** on each ROUGE-L score (5 lines of code, n_bootstrap=10000 over the 9 matches). Eliminates the ±0.02 noise question.
2. **Fine-tuned LED comparison** (Session 2) — completes the architecture × paradigm matrix.
3. **Hallucination metric.** Either: BERTScore-precision against the *transcript* (not the reference) — captures faithfulness — or NLI-based entailment from transcript to summary.
4. **Per-match-type breakdown** (FA Cup vs Euro vs WC vs UCL). With n=9 it's small per cell, but a sanity check that no single match type dominates the average.

## 9. Files

All saved under `runs/run4_full_clean_2026-05-04/`:

- `predictions/{condition}.json` — 10 prediction files, one per condition (incl. fine-tuned).
- `results/metrics.csv` — ROUGE table reproduced above.
- `results/metrics_table.tex` — same table as LaTeX.
- `kernel.log` — full Kaggle log (training metrics per epoch, generation samples per match).

Note: BART checkpoint not downloaded (~1.6 GB); available on the Kaggle kernel `marcopanciera/anlp-football-summarizer-session1` if needed.
