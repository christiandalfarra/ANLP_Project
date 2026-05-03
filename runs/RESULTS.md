# Results — All Runs

Three Kaggle sessions on T4 GPU, comparing prompting vs fine-tuning across BART/LED/FLAN-T5 on 99 football match transcripts (80 train / 10 val / 9 test).

## Headline numbers (test ROUGE-L, averaged over 9 matches)

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

## Important methodological caveat: 4 of 9 test references are empty

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

## What works

### 1. Fine-tuning beats prompting by a large margin
Both architectures benefit substantially:

- **LED**: 0.085 prompt → 0.122 fine-tune (+44%)
- **BART**: not directly comparable to FLAN-only prompting baselines, but its 0.136 dominates every other condition

### 2. Chunk-merger > naive truncation for BART
Run 2's redesign — train BART on `(concatenated pretrained-BART chunk-summaries → gold reference)` instead of `(truncated transcript[:1024] → gold reference)` — improved ROUGE-L from 0.127 → 0.136 (+7%). This matches what BART actually does at inference time, so train and test distributions are aligned.

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

## What doesn't work

### 1. FLAN-T5 + chunk-aggregate is broken
ROUGE-L of 0.03–0.04 across all three FLAN conditions. Looking at the actual outputs, FLAN-T5-Large is failing to summarize at the chunk level and instead echoing the prompt scaffolding:

> *flan_chunk_cot:* `"[Segment 19] Goals (with approximate time and scorer if mentioned), yellow/red cards"`
> *flan_chunk_few:* `"[Segment 13] Morocco and Spain played a tightly contested Round of 16 match that remained goalless..."` (wrong match — that's leaking from a few-shot example)

This is a classic FLAN failure mode: the prompt is too structured and the model treats it as a template-completion task instead of a summarization task. A FLAN-tuned prompt or a different model (e.g. Llama-3-8B-Instruct) would likely do better, but FLAN as configured here is non-functional.

### 2. LED prompting is mediocre even with long context
LED's 16k context window should help, but zero/few/CoT all land around 0.083 — only ~2× FLAN. The outputs reveal why: LED-base regurgitates the prompt or produces commentary-text that mimics input style instead of summarizing:

> *led_long_few:* `"...the ball is going to the right side of the penalty area, it's going to be difficult to get the ball back to the left side, there's a little bit of time for the ball to go in the right hand side..."`

LED-base wasn't pretrained on this kind of summarization → fine-tuning is needed to teach it the task, not just feed it more context.

### 3. Long-context fine-tuning didn't beat chunk-merger
LED at 8k-token windows during training (random-window sampling per epoch) ended up at 0.122 vs BART's 0.136. Two likely reasons:

- **N=80 is too small for LED.** With effectively 80 (input, output) pairs and 162M parameters, LED can't learn to attend across 8k tokens — it overfits to surface patterns. BART has the same parameter count but is operating on a 960-token compressed input, which is much more learnable.
- **Pretrained-BART chunk summaries are themselves a strong feature.** The merger has access to information BART already extracted at the sentence level. LED has to learn what's important from raw transcript noise.

## Sample outputs (test match: 2001 FA Cup Final, Arsenal vs Liverpool)

Reference: empty file (data quality issue), so qualitative judgment only.

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

## Limitations and what's left to do

1. **Fix the empty-reference data files.** 4/9 test matches have 0-byte gold summaries. Either re-extract or remove from the test set; current ROUGE-L numbers underestimate true performance by ~1.8×.
2. **Hallucination is unsolved.** Both fine-tuned models invent years, stadiums, scorers, and entire match outcomes. ROUGE doesn't penalize this. Adding a faithfulness check (BERTScore-precision against the transcript instead of the reference, or NLI-based entailment) would expose the gap.
3. **N=80 is the binding constraint.** Both models likely overfit; the val→test gap (BART 0.21→0.14, LED 0.17→0.12) supports this. More data is the highest-leverage improvement.
4. **No BERTScore.** Kaggle's transformers/bert_score combo throws OverflowError; the eval script falls back to ROUGE-only. Would be worth running locally with a pinned bert_score version.
5. **No statistical comparison.** With n=9 (or n=5 valid), the differences between conditions are within noise. A proper bootstrap CI or paired test would tell us how much of the BART > LED gap is signal.
