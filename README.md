# Football Match Summarizer — Applied NLP Project

A university project for the Applied Natural Language Processing course. The goal is to explore different NLP techniques for automatic football match summarization: the model takes as input the full transcription of a match (≈2.5 hours of commentary) and outputs a concise summary with the main events.

Two main approaches are compared:
- **Prompting** — use a pretrained open-source model with zero-shot, few-shot, and chain-of-thought prompting (no training required).
- **Fine-tuning** — fine-tune a summarization model on the football dataset.

Each approach is evaluated with two strategies for handling long inputs:
- **Chunk + Aggregate** — split the transcript into chunks, summarize each, then merge.
- **Long-context (LED)** — feed the full transcript to a model that natively handles long sequences.

---

## Dataset

The dataset lives in `../football_commentary_dataset/` (sibling directory of this repo). It contains **99 football matches** (World Cup, Euro, FA Cup, Premier League — 1983–2018), each with:

| File | Description |
|---|---|
| `data/transcripts/{id}_transcript.txt` | Full plain-text commentary (~300 KB per match) |
| `data/transcripts/{id}_segments.txt` | Time-aligned segments with pitch/energy features |
| `data/transcripts/{id}.json` | Structured JSON with all of the above |
| `data/summaries/{id}.txt` | Ground-truth summary (1–2 paragraphs + key events) |

---

## Project Structure

```
progetto chri/
├── src/
│   ├── data/
│   │   ├── dataset_loader.py      # Load transcripts, segments, summaries
│   │   ├── chunker.py             # Token-count chunking aligned to segment boundaries
│   │   └── splits.py             # Stratified train/val/test split (80/10/9)
│   ├── models/
│   │   ├── prompting/
│   │   │   ├── base_prompter.py   # Abstract base class for all prompters
│   │   │   ├── flan_prompter.py   # FLAN-T5-Large (chunk-based prompting)
│   │   │   └── led_prompter.py    # LED-Base-16384 (long-context prompting)
│   │   └── finetuning/
│   │       ├── trainer.py         # HuggingFace Trainer + early stopping on ROUGE-L
│   │       ├── bart_finetuner.py  # BART-Large-CNN fine-tuning (freeze/unfreeze encoder)
│   │       └── led_finetuner.py   # LED-Base fine-tuning (gradient checkpointing)
│   ├── pipelines/
│   │   ├── chunk_aggregate.py     # 2-stage pipeline: chunk → summarize → merge
│   │   └── longcontext.py         # Single-pass full-transcript pipeline
│   ├── evaluation/
│   │   ├── metrics.py             # ROUGE-1/2/L and BERTScore
│   │   └── evaluate_all.py        # Aggregate all conditions → CSV + LaTeX table
│   └── prompts/
│       ├── zero_shot.py           # Zero-shot templates
│       ├── few_shot.py            # 2-shot templates with example selection
│       └── chain_of_thought.py    # 2-pass CoT templates
├── scripts/
│   ├── generate_splits.py         # One-time: create and save train/val/test split
│   ├── run_prompting.py           # Run prompting experiments, save predictions
│   ├── run_finetuning.py          # Fine-tune BART or LED
│   ├── run_inference_finetuned.py # Inference with fine-tuned checkpoints
│   └── run_evaluation.py          # Compute all metrics, write CSV + LaTeX
├── outputs/
│   ├── splits.json                # Committed — ensures reproducibility
│   ├── predictions/               # Generated summaries per condition (gitignored)
│   └── results/                   # metrics.csv + metrics_table.tex (gitignored)
├── checkpoints/                   # Fine-tuned model weights (gitignored)
├── requirements.txt
└── README.md
```

---

## Models Used

| Role | Model | Params | Notes |
|---|---|---|---|
| Prompting (chunk) | `google/flan-t5-large` | 780M | Instruction-following T5 variant |
| Prompting (long) | `allenai/led-base-16384` | 162M | Longformer encoder-decoder, 16k token context |
| Fine-tuning (chunk) | `facebook/bart-large-cnn` | 406M | Pre-trained on CNN/DailyMail news summarization |
| Fine-tuning (long) | `allenai/led-base-16384` | 162M | Same architecture, fine-tuned on football data |

---

## Compute Requirements

All experiments are designed to run on **free cloud GPUs**:

| Task | Recommended Platform | Notes |
|---|---|---|
| Prompting (FLAN-T5, LED) | Google Colab (T4, 15 GB) | ~1–2h per condition |
| Fine-tuning BART | Google Colab (T4, 15 GB) | fp16=True, batch=1 + grad_accum=8 |
| Fine-tuning LED | Kaggle (T4×2 or P100, 30h/week) | gradient_checkpointing=True required |
| Evaluation (ROUGE) | Any CPU | Fast, no GPU needed |
| Evaluation (BERTScore) | Colab/Kaggle GPU | Slow on CPU |

**Tips for free compute:**
- Mount Google Drive in Colab (`drive.mount('/content/drive')`) and save checkpoints there so they persist across sessions.
- On Kaggle, use "Save & Run All" and attach the output as a dataset for the next notebook.
- Run one condition at a time with `--model`/`--strategy`/`--prompt` flags; predictions are cached to disk so you never recompute a finished condition.

---

## Installation

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `transformers`, `datasets`, `rouge_score`, `bert_score`, `accelerate`, `sentencepiece`.

---

## Full Pipeline — Step by Step

### Step 1 — Generate train/val/test splits

Run **once**. Creates `outputs/splits.json` (80 train / 10 val / 9 test, stratified by competition type). Commit this file for reproducibility.

```bash
python scripts/generate_splits.py
```

Expected output:
```
Total matches: 99
Train: 80  Val: 10  Test: 9
Splits saved to outputs/splits.json
```

---

### Step 2 — Run prompting experiments

Runs inference on the 9 test matches for each prompting condition. Each condition saves a JSON file to `outputs/predictions/`.

**Run all 6 conditions at once:**
```bash
python scripts/run_prompting.py --all
```

**Or run individually:**
```bash
# FLAN-T5, chunk+aggregate
python scripts/run_prompting.py --model flan --strategy chunk --prompt zero
python scripts/run_prompting.py --model flan --strategy chunk --prompt few
python scripts/run_prompting.py --model flan --strategy chunk --prompt cot

# LED, long-context
python scripts/run_prompting.py --model led --strategy long --prompt zero
python scripts/run_prompting.py --model led --strategy long --prompt few
python scripts/run_prompting.py --model led --strategy long --prompt cot
```

Each run skips conditions that already have a saved prediction file, so you can resume safely after a session timeout.

---

### Step 3 — Fine-tune BART (chunk strategy)

Fine-tunes `facebook/bart-large-cnn` on the 80 training matches. Uses a two-phase strategy: encoder frozen for the first 3 epochs, then fully unfrozen. Early stopping on validation ROUGE-L (patience=3).

```bash
python scripts/run_finetuning.py --model bart --output_dir checkpoints/bart
```

Training takes approximately **2–4 hours on Colab T4**. Checkpoint is saved to `checkpoints/bart/`.

**Hyperparameters:** lr=3e-5, batch=1, grad_accum=8, fp16=True, max_input=1024 tokens, max_target=256 tokens.

---

### Step 4 — Fine-tune LED (long-context strategy)

Fine-tunes `allenai/led-base-16384` on full transcripts (capped at 8192 tokens). Requires `gradient_checkpointing=True` to fit on GPU.

```bash
python scripts/run_finetuning.py --model led --output_dir checkpoints/led
```

Training takes approximately **4–8 hours**. Use Kaggle (30h/week free) for this step if Colab times out.

**Hyperparameters:** lr=5e-5, batch=1, grad_accum=8, fp16=True, gradient_checkpointing=True, max_input=8192 tokens.

---

### Step 5 — Inference with fine-tuned models

Generate summaries on the test set using the saved checkpoints.

```bash
python scripts/run_inference_finetuned.py --model bart --checkpoint checkpoints/bart
python scripts/run_inference_finetuned.py --model led  --checkpoint checkpoints/led
```

Saves to:
- `outputs/predictions/finetuned_bart.json`
- `outputs/predictions/finetuned_led.json`

---

### Step 6 — Evaluate all conditions

Computes ROUGE-1, ROUGE-2, ROUGE-L, and BERTScore for every prediction file in `outputs/predictions/`. Writes results to `outputs/results/`.

```bash
# With BERTScore on GPU (recommended)
python scripts/run_evaluation.py --device cuda

# ROUGE only (fast, no GPU needed)
python scripts/run_evaluation.py --no-bertscore
```

Output files:
- `outputs/results/metrics.csv` — full results table
- `outputs/results/metrics_table.tex` — LaTeX table ready to paste into the report

---

## Experimental Conditions Summary

After completing all steps, 8 conditions will be evaluated:

| Condition | Model | Strategy | Prompt |
|---|---|---|---|
| `flan_chunk_zero` | FLAN-T5-Large | Chunk + Aggregate | Zero-shot |
| `flan_chunk_few` | FLAN-T5-Large | Chunk + Aggregate | Few-shot (2-shot) |
| `flan_chunk_cot` | FLAN-T5-Large | Chunk + Aggregate | Chain-of-Thought |
| `led_long_zero` | LED-Base-16384 | Long-context | Zero-shot |
| `led_long_few` | LED-Base-16384 | Long-context | Few-shot |
| `led_long_cot` | LED-Base-16384 | Long-context | CoT |
| `finetuned_bart` | BART-Large-CNN | Chunk + Aggregate | — |
| `finetuned_led` | LED-Base-16384 | Long-context | — |

---

## Evaluation Metrics

| Metric | Library | Description |
|---|---|---|
| ROUGE-1 | `rouge_score` | Unigram overlap between generated and reference summary |
| ROUGE-2 | `rouge_score` | Bigram overlap |
| ROUGE-L | `rouge_score` | Longest common subsequence (used for early stopping) |
| BERTScore F1 | `bert_score` | Semantic similarity using DeBERTa embeddings |

---

## Chunking Details

For chunk-based models, transcripts are split using the `_segments.txt` file (which has time-aligned boundaries) rather than the raw transcript. This ensures chunk boundaries always fall between spoken sentences, not mid-word.

- **FLAN-T5:** max 450 tokens per chunk (512 token limit − prompt overhead)
- **BART (fine-tuned):** max 900 tokens per chunk (1024 token limit − summary space)
- **LED:** no chunking — full transcript in one pass

A 1-segment overlap is added between consecutive chunks to avoid hard context cuts.

---

## Reproducing Results

To ensure full reproducibility:
1. `outputs/splits.json` is committed to the repo — always the same 80/10/9 split.
2. All random seeds are fixed to `42`.
3. Predictions are saved to disk before evaluation — you can re-run metrics without re-running inference.
4. Each script skips already-computed outputs, making it safe to resume after interruption.
