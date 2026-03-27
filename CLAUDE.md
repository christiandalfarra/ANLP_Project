# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Generate train/val/test splits (run once, commit outputs/splits.json)
python scripts/generate_splits.py

# Run prompting experiments
python scripts/run_prompting.py --all                                      # all 6 conditions
python scripts/run_prompting.py --model flan --strategy chunk --prompt zero  # single condition

# Fine-tune models
python scripts/run_finetuning.py --model bart --output_dir checkpoints/bart
python scripts/run_finetuning.py --model led  --output_dir checkpoints/led

# Inference with fine-tuned checkpoints
python scripts/run_inference_finetuned.py --model bart --checkpoint checkpoints/bart
python scripts/run_inference_finetuned.py --model led  --checkpoint checkpoints/led

# Evaluate all saved predictions
python scripts/run_evaluation.py --device cuda       # with BERTScore
python scripts/run_evaluation.py --no-bertscore      # ROUGE only, no GPU needed
```

## Architecture

The project compares 8 experimental conditions across two axes: **approach** (prompting vs. fine-tuning) and **input strategy** (chunk+aggregate vs. long-context LED). All scripts save outputs to disk before evaluation, so any step can be resumed after interruption.

### Data flow

```
../football_commentary_dataset/data/
  transcripts/{id}_segments.txt   ← primary source for chunking (has segment boundaries)
  transcripts/{id}_transcript.txt ← used for long-context (LED) pipeline
  summaries/{id}.txt              ← ground truth for evaluation
```

`src/data/dataset_loader.py` loads all three into a `Match` dataclass. `src/data/splits.py` reads `outputs/splits.json` (80/10/9 stratified split committed to repo).

### Two pipelines

**Chunk + Aggregate** (`src/pipelines/chunk_aggregate.py`): uses `src/data/chunker.py` to split `_segments.txt` into token-count-bounded chunks (aligned to segment boundaries, 1-segment overlap), runs a `generate_fn(text, start, end)` per chunk, then a `merge_fn(list[str])` for the final summary. Post-processes with BoW cosine deduplication.

**Long-context** (`src/pipelines/longcontext.py`): passes the full `_transcript.txt` to a single `generate_fn(text)`. No chunking.

### Prompting vs. fine-tuning wiring

Prompts are pure functions in `src/prompts/` — they return strings, know nothing about models. Prompters in `src/models/prompting/` wrap HuggingFace models and expose a `generate(prompt) -> str` interface. The scripts in `scripts/` wire prompts + prompters + pipelines together.

Fine-tuning uses `src/models/finetuning/trainer.py` (shared `Seq2SeqTrainer` wrapper with early stopping on val ROUGE-L) called by `bart_finetuner.py` (two-phase: encoder frozen → unfrozen) and `led_finetuner.py` (gradient checkpointing, global attention mask on BOS token).

### Token limits per model

| Model | Max input tokens | Chunker threshold |
|---|---|---|
| FLAN-T5-Large | 512 | 450 |
| BART-Large-CNN | 1024 | 900 |
| LED-Base-16384 | 16384 (train: 8192) | no chunking |

### Prediction file naming

`outputs/predictions/{condition}.json` maps `match_id -> generated_summary`. Condition names: `flan_chunk_zero`, `flan_chunk_few`, `flan_chunk_cot`, `led_long_zero`, `led_long_few`, `led_long_cot`, `finetuned_bart`, `finetuned_led`. `run_evaluation.py` auto-discovers all `.json` files in that directory.

### Dataset path

`src/data/dataset_loader.py` resolves the dataset as `../football_commentary_dataset/` relative to `src/data/`. If running from a different working directory (e.g. Colab), override by passing `dataset_dir=` to `load_match()` / `list_match_ids()`.
