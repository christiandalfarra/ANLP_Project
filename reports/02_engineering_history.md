# Engineering history — past approaches, problems, and fixes

A chronological record of what was tried, what broke, and what we learned. Useful for the report's *Methodology* and *Limitations* sections, and as a reproducibility audit trail.

---

## Phase 0 — Project setup

**Goal:** Run the full pipeline (prompting + fine-tuning) on Kaggle since local hardware (RTX 4060 Laptop, 8 GB VRAM) cannot fit LED-base at 8192-token training context.

- Code lives in `christiandalfarra/ANLP_Project` GitHub repo.
- Data lives in `panciut/football_commentary_dataset`, vendored as a directory inside the project.
- Kaggle session limit: 12 h per run; weekly quota ~30 GPU-hours.
- All training and inference runs through `kaggle kernels push` with the notebook cloning the GitHub repo at start (`git clone --branch setup/local-run`).

## Phase 1 — Run 1: prompting baselines + naive BART fine-tune (2026-05-02)

**Conditions:** 6 prompting baselines (FLAN/LED × zero/few/CoT) + naive truncated-input BART fine-tune.

**Results:**

| Condition | ROUGE-L |
|---|---:|
| finetuned_bart (truncated 1024) | 0.1274 |
| led_long_cot | 0.0846 |
| led_long_few/zero | 0.0825 |
| flan_chunk_zero | 0.0413 |
| flan_chunk_cot | 0.0329 |
| flan_chunk_few | 0.0278 |

**Issues identified:**

1. **BART fine-tuning was naive.** Training pairs were `(transcript[:1024 tokens], gold_summary)` — the model only ever saw the first ~5 minutes of a 2.5 h match. The trained model effectively learned to summarize kickoffs.
2. **BERTScore broken on Kaggle's transformers build** (`OverflowError: out of range integral type conversion`). Switched to `--no-bertscore`; no metric beyond ROUGE.
3. **No BART prompting baseline.** Only FLAN and LED were evaluated zero-shot — making it impossible to isolate the contribution of fine-tuning vs choice of model.

## Phase 2 — Run 2: BART chunk-merger redesign (2026-05-03)

**Hypothesis:** BART should be trained as the *merge step* of the chunk+aggregate pipeline, not as a naive summarizer of truncated input. That way training and inference distributions match.

**Implementation:** For each training match, run pretrained BART on each chunk to produce 30 mini-summaries (~32 tokens each), concatenate into ~960 tokens, then train BART to map *that* concatenated intermediate → gold reference. Cached the chunk-summary generation step on disk to avoid re-running the ~30 min one-time cost.

**Result:** ROUGE-L 0.1274 → 0.1361 (+7%). Modest but consistent improvement.

**Limitations exposed:**

- Two-phase training (encoder freeze → unfreeze) was added because phase-1 alone left ROUGE flat; the freeze→unfreeze schedule gave the +7%.
- Disk pressure: 1.6 GB BART checkpoint + chunk-summary cache + activations almost OOM'd `/kaggle/working`. Added `save_only_model=True` and `save_total_limit=1` to the trainer.
- **The download from Kaggle to local was the first sign of trouble** — Kaggle CLI silently truncated due to a Windows charmap encoding bug + paginated file listing. Multiple downloads to the same dir overwrote good files with 0-byte placeholders. Workaround: paginate via `ApiListKernelSessionOutputRequest` and call the python API directly with `PYTHONIOENCODING=utf-8`.

## Phase 3 — Run 3: LED fine-tuning (multiple failed attempts)

This was the longest detour of the project — five distinct failure modes, four hyperparameter changes, two real bugs. Documented in detail because it's the bulk of the engineering effort.

### Attempt v3 — `lr=5e-5`, label smoothing 0.1, eval beams=1
- **Symptom:** Eval ROUGE-L collapsed to 0 at epoch 3 even though eval_loss was decreasing (4.7 → 3.6).
- **Loss diagnostic:** `loss=91.85, grad_norm=209.2` at epoch 2 → optimizer fighting huge gradients despite gradient clipping at 1.0.
- **Hypothesis:** LR too aggressive + label smoothing pushed decoder toward uniform distribution → with greedy decoding, EOS won the argmax → generated empty strings → ROUGE=0.

### Attempt v5 — `lr=2e-5`, label smoothing OFF, eval beams=2
- **Symptom:** Same collapse pattern. ROUGE-L: 0.077 → 0.077 → 0 by epoch 3.
- **Diagnostic:** `loss=76.77, grad_norm=576.7` — gradients *worse* than v3.
- **Realisation:** The fix wasn't about learning rate. The model was actually predicting `<s></s>` (immediate EOS) as the highest-probability sequence. Label smoothing wasn't the cause; the cause was deeper.

### Attempt v7 — discovered the real bug
- **Bug found:** `scripts/run_finetuning.py` was pre-truncating each LED transcript to 8192 tokens *before* handing it to `LEDDataset`. The dataset's "random window sampler" then ran on transcripts of length 8192 with window size 8192 — `n > window` was false so no random offset was applied. **The model was always seeing the first 8192 tokens of every match, every epoch, never the rest.** That explained both the collapse to empty (model sees same input → same gold pair 15 times → memorizes nothing) and why no hyperparameter change helped.
- **Fix 1:** Don't pre-truncate; pass full transcripts to `LEDDataset` so the random window sampler actually fires.
- **Fix 2 (compounding):** Set `model.generation_config.min_length = 80` to force eval generation to produce ≥80 tokens. Even with the truncation bug fixed, the model could still find the empty-output solution because of issue #4 below.
- **Result:** Training reached ROUGE-L 0.125 at epoch 5 — first non-collapsed run. Then crashed at epoch 6 eval with `OverflowError`.

### Attempt v9 — recovered checkpoint despite crash
- **Cascade failure from v7's crash:** Training never reached `trainer.save_model()`, so `checkpoints/led/model.safetensors` didn't exist; inference 404'd; eval found no predictions.
- **Fix:** Added a fallback in `scripts/run_inference_finetuned.py`: if the parent checkpoint dir lacks a model file, fall back to the most recent `checkpoint-N/` subdir.
- **Recovered the epoch-5 checkpoint, ran inference. Test ROUGE-L: 0.098.** Disappointing but real.

### Attempt v11 — fixed the OverflowError
- **Bug found:** The `OverflowError` was in `compute_rouge_metrics` (called by `Seq2SeqTrainer.evaluate`). When the trainer passes generated `preds` to `tokenizer.batch_decode`, the array can contain `-100` for unfilled beam positions or other sentinels. The fast tokenizer's `batch_decode` casts to a Rust `u32` and overflows on negative values. We were already cleaning `-100` in *labels* before decoding, but not in *preds*.
- **Fix:** Same `np.where(preds < 0 | preds >= vocab_size, pad, preds)` cleanup applied to preds.
- **Result:** Training ran cleanly through 11 epochs (early-stop fired at patience=2). Best val ROUGE-L = 0.172 at epoch 9. **Test ROUGE-L = 0.122** — beats LED prompting (0.085) but loses to BART chunk-merger (0.136).

### What we actually learned from the LED detour

1. **Five distinct bugs**, all silent or distractedly attributed to hyperparameters at first:
   - Pre-truncation defeating the random-window sampler
   - Label smoothing + greedy eval producing collapse
   - `eval_accumulation_steps=1` triggering an OverflowError (red herring — turned out unrelated, but I burned a run on it)
   - `OverflowError` actually in `compute_rouge_metrics` from -100 in preds
   - Inference assuming `model.safetensors` exists after a crashed run
2. **`eval_loss` improving while ROUGE collapses to 0 is a real failure mode.** Teacher-forced cross-entropy doesn't measure generation quality. Always check ROUGE on actual `model.generate()` output, and force a `min_length` floor during eval to expose collapse-to-empty bias.
3. **Run the trainer to completion at least once before downstream scripts assume artifacts.** Add fallbacks for partial state.

## Phase 4 — The empty-references discovery (2026-05-03 evening)

The biggest single fix in the whole project, found by inspecting test predictions.

**Symptom:** Generated summaries kept hallucinating obvious facts (wrong stadium, wrong year, wrong scorer) yet scored ROUGE-L 0.13. Wanted to check raw test references; opening `2001_facup_arsenal_liverpool.txt` showed it was 0 bytes.

**Investigation:**
```
train: 26/80 empty refs (33%)
val:    3/10 empty refs (30%)
test:   4/9  empty refs (44%)
```

**Implications:**

1. **Training:** 26/80 = 33% of training pairs were `(transcript, "")`. The model literally learned that for one-third of inputs the correct output is empty. **This is the actual root cause of LED's collapse-to-EOS** — no amount of hyperparameter tuning would fix a model being trained to produce nothing 33% of the time.
2. **Validation:** 3/10 empty val refs = 30% of the early-stopping signal was noise.
3. **Test:** 4/9 empty test refs = the average ROUGE-L was always (sum of valid 5 + 0 + 0 + 0 + 0) / 9 — about 1.8× lower than what the actual valid-match average would have been.

**Worked example:** Run 2 finetuned_bart reported ROUGE-L 0.1361. Predicted true average over 5 valid matches = 0.1361 × 9/5 = 0.245. Confirmed empirically: re-running the same script with valid refs only gave 0.245 exactly.

**Fix:** The dataset repository had a newer commit (`dataset 1.0`) where all 99 references were populated. Pulled it, copied into the in-project vendored dataset, committed.

**Result the next day:** Training pairs went from 54 valid + 26 corrupt → 80 valid. Re-running the same code:

| Condition | Before fix | After fix | Δ |
|---|---:|---:|---:|
| finetuned_bart (chunk-merger) | 0.1361 | 0.2476 | +82% |
| led_long_zero | 0.0825 | 0.1599 | +94% |
| flan_chunk_zero | 0.0413 | 0.0700 | +69% |

The actual measured improvement matches the predicted ~1.8× ratio almost exactly, confirming that the empty-target problem was the dominant noise source in all earlier results.

## Phase 5 — Run 4: full clean re-run (2026-05-04)

Documented in `01_session1_results.md`. With clean data and all bug fixes from phases 1-4 applied:

- Added BART chunk-aggregate baseline (zero/few/CoT) to make `--all` cover all three model families.
- Single Session 1 notebook now runs all 9 prompting baselines + BART fine-tune end-to-end in ~2.5 h.
- LED fine-tuning is its own Session 2, ~50 min.

## Lessons applicable beyond this project

1. **A 33% empty-target dataset can produce results that look 'modest but coherent' rather than obviously broken.** Always inspect raw inputs/outputs before trusting metrics, especially on small datasets.
2. **`eval_loss` and `eval_rouge_x` can disagree dramatically.** Loss is teacher-forced; ROUGE is on free-running generation. They measure different things. Trust the one that matches your actual deployment.
3. **`min_length` / `min_new_tokens` is a useful diagnostic, not just a hyperparameter.** Forcing the decoder to emit some text exposes EOS-collapse failures that greedy decoding would silently absorb.
4. **Kaggle's CLI silently truncates downloads on Windows due to a charmap bug.** Use the python API with `PYTHONIOENCODING=utf-8` and paginate manually via `ApiListKernelSessionOutputRequest`.
5. **Saving intermediate checkpoints is more important than minimising disk.** `save_total_limit=1` saved disk but meant a crash mid-eval lost the entire run; an inference-side fallback to `checkpoint-N/` recovered it.
6. **When the same hyperparameter tuning attempt fails twice with different parameters, suspect a code bug, not hyperparameters.** The LED collapse persisted through three hyperparameter changes before we found the pre-truncation bug; that was three wasted runs.

## Final commit timeline (branch `setup/local-run`)

```
6bdd364 Ignore football_commentary_dataset/data/raw_audio/ (19 GB)
7e058a8 Prep tomorrow's full re-run with clean dataset + BART baselines
f77035f Add verified valid-only ROUGE re-eval output
31058b7 Add results from all three Kaggle runs + RESULTS.md analysis
5633a89 Fix OverflowError in compute_rouge_metrics
a06403d LED fix: drop eval_accumulation_steps + checkpoint-N fallback in inference
13f6251 Fix LED collapse: real bug + generation floor
ac0efd3 LED fine-tune: lower LR, drop label smoothing, eval with beam=2
24b63b1 Refine LED fine-tuning for stable eval on Kaggle T4
d875ad1 (BART merger redesign and earlier setup commits)
```

Each commit corresponds to a hypothesis tested or a bug fixed. The fact that 8 of these 10 commits are on the LED branch reflects how much engineering went into one (eventually-working) model.
