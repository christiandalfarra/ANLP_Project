# Analysis Plan — Getting Top Marks

This document describes all additional analyses to perform on top of the base pipeline (prompting + fine-tuning + ROUGE/BERTScore evaluation). Each analysis includes motivation, concrete implementation steps, expected output, and where to include it in the report.

---

## Analysis 1 — Hallucination / Factual Consistency

> **Claude Code prompt to implement this:**
> *"Implement Analysis 1 from analysis.md. Create `scripts/run_analysis_hallucination.py` that: (1) loads all 8 prediction JSON files from `outputs/predictions/`, (2) parses ground-truth summaries and generated summaries to extract scores (regex on patterns like `2–1`, `3-0`) and scorer names (words preceding/following 'goal' or 'scored'), (3) flags mismatches per condition, (4) prints a hallucination rate table and saves it to `outputs/results/hallucination.csv`."*

**Why it matters:** The most critical failure mode in summarization is generating events that never happened (invented goals, wrong scores, non-existent scorers). Examiners expect you to go beyond ROUGE and address faithfulness.

**Steps:**

1. For each of the 9 test matches, extract the ground-truth score and goalscorers from `data/summaries/{id}.txt` using a regex on patterns like `"2–1"`, `"scored"`, `"goal"`.
2. For each generated summary (all 8 conditions), extract the same fields.
3. Flag mismatches: wrong score, wrong scorer name, goal time off by more than 10 minutes, or events not present in the ground truth at all.
4. Build a simple per-condition table: number of hallucinated facts / total facts mentioned.
5. Manually verify the 9 test summaries — it is a small enough set to read in full.

**Output:** Table `hallucination_rate` per condition in the report. Example:

| Condition | Hallucinated facts | Total facts | Rate |
|---|---|---|---|
| flan_chunk_zero | 3 | 18 | 16.7% |
| finetuned_bart | 1 | 21 | 4.8% |

**Where to include:** Dedicated subsection in Results. This is often the most-discussed finding.

---

## Analysis 2 — Performance Breakdown by Match Type

> **Claude Code prompt to implement this:**
> *"Implement Analysis 2 from analysis.md. Create `scripts/run_analysis_breakdown.py` that: (1) loads `outputs/splits.json` and `outputs/predictions/*.json`, (2) categorises each test match by competition type (infer from match ID: wc=World Cup, euro=Euro, fa_cup=FA Cup, else Premier League) and by total goals from the ground-truth summary, (3) computes ROUGE-L per category per condition using `src/evaluation/metrics.py`, (4) saves a breakdown CSV to `outputs/results/breakdown_by_type.csv` and prints a formatted table. Also check if matches with year < 2000 in the ID score lower on average and print a note."*

**Why it matters:** Aggregate ROUGE hides systematic failures. A model that scores well on average but collapses on low-scoring or high-scoring matches is not robust.

**Steps:**

1. Categorize the 9 test matches along two axes:
   - **Competition type:** World Cup, Euro, FA Cup, Premier League (from match ID naming convention).
   - **Match drama:** extract total goals from ground truth summary. Define: low-scoring (0–1 goals total), medium (2–3), high-scoring (4+).
2. For each category, compute ROUGE-L for every condition separately.
3. Plot a grouped bar chart: x-axis = competition type, y-axis = ROUGE-L, bars grouped by condition.
4. Check whether matches from earlier years (pre-2000) score lower — Whisper transcription quality degrades on older audio. Flag this as a data quality finding if confirmed.

**Output:** Breakdown table + bar chart. Hypothesis to test: *"Fine-tuned models generalize better across match types than zero-shot prompting."*

**Where to include:** Results section, after the main metrics table.

---

## Analysis 3 — Chunk Size Ablation

> **Claude Code prompt to implement this:**
> *"Implement Analysis 3 from analysis.md. Add a `--max_tokens` argument to `scripts/run_prompting.py` that overrides the default chunk size, and make it pass through to `run_chunk_aggregate` in `src/pipelines/chunk_aggregate.py`. Then create `scripts/run_analysis_ablation_chunks.py` that runs the FLAN-T5 zero-shot chunk+aggregate pipeline on the test set three times with max_tokens in [256, 450, 900], recording ROUGE-1/2/L and average inference time per match for each, and saves results to `outputs/results/ablation_chunk_size.csv`."*

**Why it matters:** The chunk size is a key hyperparameter with no obvious optimal value. An ablation directly justifies your design choice and shows scientific rigor.

**Steps:**

1. For FLAN-T5 chunk+aggregate, run the pipeline on the 9 test matches with three chunk sizes:
   - Small: 256 tokens
   - Medium: 450 tokens (current default)
   - Large: 900 tokens (pushes against the 512 token model limit — requires truncation of each chunk)
2. Record ROUGE-1/2/L for each size.
3. Also record: average number of chunks per match and total inference time per match.
4. Plot: ROUGE-L (y) vs. chunk size (x). Expect a sweet-spot curve — too small loses context per chunk, too large truncates.

**Implementation:** In `scripts/run_prompting.py`, add `--max_tokens` argument and pass it through to `run_chunk_aggregate`. The `chunker.py` already accepts `max_tokens` as a parameter.

**Output:** Line chart + table with ROUGE and inference time per chunk size.

**Where to include:** Hyperparameter analysis subsection. One of the cleanest graphs in the report.

---

## Analysis 4 — Acoustic Feature-Guided Chunking (Original Contribution)

> **Claude Code prompt to implement this:**
> *"Implement Analysis 4 from analysis.md. (1) Add a `filter_high_energy_segments(segments, top_k_ratio=0.4)` function to `src/data/chunker.py` that keeps only the top-40% highest-energy segments. (2) Create `scripts/run_analysis_acoustic.py` that runs the FLAN-T5 zero-shot chunk+aggregate pipeline on the test set twice — once with all segments and once with energy-filtered segments — computes ROUGE-1/2/L for both, and saves a comparison to `outputs/results/acoustic_vs_uniform.csv`. (3) For the match `2014_wc_brazil_germany` (or the first available test match if not in test set), plot energy over time (x=segment start time in seconds, y=energy value) using matplotlib, mark goal times extracted from the ground-truth summary with vertical red lines, and save the figure to `outputs/results/energy_plot.png`."*

**Why it matters:** Your dataset uniquely contains per-segment pitch and energy values from the original Whisper transcription pipeline. No existing summarization paper exploits this. Even a negative result is publishable.

**Hypothesis:** High-energy / high-pitch moments (crowd noise, excited commentary) correlate with key events (goals, red cards). Filtering to these segments before feeding the model may improve summary quality.

**Steps:**

1. Add `chunk_by_energy` to `src/data/chunker.py`:
   ```python
   def filter_high_energy_segments(segments, top_k_ratio=0.4):
       """Keep only the top 40% highest-energy segments."""
       if not segments:
           return segments
       threshold = sorted(s.energy for s in segments)[int(len(segments) * (1 - top_k_ratio))]
       return [s for s in segments if s.energy >= threshold]
   ```
2. Run the FLAN-T5 zero-shot chunk+aggregate pipeline twice on the test set:
   - Baseline: all segments (current behaviour)
   - Filtered: only top-40% energy segments passed to chunker
3. Compare ROUGE-1/2/L between the two.
4. As a secondary analysis, plot energy over time for one match (x = time in seconds, y = energy). Manually annotate known goal times from the ground truth summary and check whether they coincide with energy peaks.

**Output:**
- Comparison table: uniform chunking vs. energy-guided chunking.
- One annotated energy-over-time plot for a high-scoring match (e.g. Brazil 1–7 Germany 2014).

**Where to include:** Dedicated subsection titled *"Exploiting Acoustic Features for Event-Guided Summarization"*. This is the most original contribution of the project and should be highlighted.

---

## Analysis 5 — Faithfulness vs. Coverage Trade-off (BERTScore Precision vs. Recall)

> **Claude Code prompt to implement this:**
> *"Implement Analysis 5 from analysis.md. Update `scripts/run_evaluation.py` to save BERTScore precision and recall separately in `outputs/results/metrics.csv` (they are already returned by `compute_bertscore` in `src/evaluation/metrics.py`). Then create `scripts/run_analysis_bertscore_scatter.py` that reads `outputs/results/metrics.csv`, produces a matplotlib scatter plot with BERTScore Precision on the x-axis and BERTScore Recall on the y-axis, one labeled point per condition, and saves it to `outputs/results/bertscore_scatter.png`."*

**Why it matters:** BERTScore F1 is the harmonic mean of precision and recall, but they measure different things. Precision = how much of the generated text is grounded in the reference (faithfulness). Recall = how much of the reference is covered (completeness). Fine-tuned models and prompted models likely trade these off differently.

**Steps:**

1. From `src/evaluation/metrics.py`, `compute_bertscore` already returns precision, recall, and F1 separately.
2. For every condition, plot a scatter chart: x = BERTScore Precision, y = BERTScore Recall, one point per condition. Label each point.
3. Expected pattern: prompting approaches cluster toward higher recall (verbose, covers more) but lower precision (less grounded). Fine-tuning approaches cluster toward higher precision.

**Output:** Scatter plot + interpretation paragraph.

**Where to include:** Analysis subsection, after the main results table.

---

## Analysis 6 — Prompt Sensitivity

> **Claude Code prompt to implement this:**
> *"Implement Analysis 6 from analysis.md. (1) Add two new prompt variant functions to `src/prompts/zero_shot.py`: `build_chunk_prompt_v2` (uses 'Write a match report based on the following commentary excerpt') and `build_chunk_prompt_v3` (uses 'You are a football journalist. Extract the main events from the commentary below'). (2) Create `scripts/run_analysis_prompt_sensitivity.py` that runs the FLAN-T5 chunk+aggregate pipeline on the test set with all three zero-shot prompt variants, computes ROUGE-L per match per variant, computes the standard deviation of ROUGE-L across the 3 variants for each match, and saves results to `outputs/results/prompt_sensitivity.csv`."*

**Why it matters:** A good prompting approach should be robust to minor wording changes. If ROUGE varies significantly with small prompt edits, the approach is unreliable — an important limitation to report.

**Steps:**

1. Create 3 variants of the zero-shot chunk prompt in `src/prompts/zero_shot.py`:
   - Variant A (current): *"Summarize the football commentary. Focus on goals, key moments..."*
   - Variant B: *"Write a match report based on the following commentary excerpt..."*
   - Variant C: *"You are a football journalist. Extract the main events from the commentary below..."*
2. Run all three on the 9 test matches using FLAN-T5 chunk+aggregate.
3. Compute ROUGE-L per variant. Compute standard deviation across variants per match.
4. If std > 0.05 ROUGE-L points on average, conclude that prompting is sensitive to phrasing — a valid limitation.

**Output:** Table with ROUGE-L for each prompt variant + std across variants.

**Where to include:** Limitations section or dedicated prompt analysis subsection.

---

## Analysis 7 — Training Dynamics (Fine-tuning Only)

> **Claude Code prompt to implement this:**
> *"Implement Analysis 7 from analysis.md. Create `scripts/run_analysis_training_curves.py` that: (1) reads `checkpoints/bart/trainer_state.json` and `checkpoints/led/trainer_state.json` (HuggingFace saves this automatically during training), (2) extracts per-epoch train loss and eval rougeL from the `log_history` field, (3) for each model plots train loss and eval ROUGE-L on the same chart with dual y-axes, marks the early stopping epoch with a vertical dashed line, and saves the figure to `outputs/results/training_curves_bart.png` and `outputs/results/training_curves_led.png`."*

**Why it matters:** With only 80 training samples, overfitting is a real risk. Showing training curves demonstrates you understand and monitored this.

**Steps:**

1. The HuggingFace `Seq2SeqTrainer` in `src/models/finetuning/trainer.py` already logs metrics per epoch (set `logging_steps=10`). Enable `report_to="none"` to keep it local.
2. After fine-tuning, read the `trainer_state.json` file saved in the checkpoint directory. It contains per-epoch train loss and validation ROUGE-L.
3. Plot two curves on the same chart: train ROUGE-L and val ROUGE-L vs. epoch number. Mark the early stopping epoch.
4. Do this for both BART and LED.

**Output:** Two line charts (one per model). Look for: val ROUGE-L plateauing or declining while train ROUGE-L continues to rise (overfitting signal).

**Where to include:** Fine-tuning subsection of Results.

---

## Analysis 8 — Qualitative Case Study

> **Claude Code prompt to implement this:**
> *"Implement Analysis 8 from analysis.md. Create `scripts/run_analysis_case_study.py` that: (1) loads `outputs/results/metrics.csv` and identifies the test match with the highest ROUGE-L and the one with the lowest ROUGE-L averaged across all conditions, (2) for those two matches, prints a side-by-side comparison of: ground-truth summary, best prompting output (highest ROUGE-L among prompting conditions), best fine-tuned output, (3) saves the comparison as a plain text file to `outputs/results/case_study.txt` with clear section headers. For the worst-case match, also plot its energy-over-time signal (reusing the logic from Analysis 4) and save to `outputs/results/case_study_worst_energy.png`."*

**Why it matters:** Numbers alone don't tell a story. A qualitative case study makes the report readable and gives examiners something to anchor the quantitative results to.

**Steps:**

1. From the 9 test matches, identify:
   - **Best case:** the match where your best overall condition achieves the highest ROUGE-L.
   - **Worst case:** the match where all conditions score lowest.
2. For both matches, produce a side-by-side comparison table:
   - Ground truth summary
   - Best prompting output (e.g. `flan_chunk_cot`)
   - Best fine-tuned output (e.g. `finetuned_bart`)
3. Annotate: what did the model get right (correct score, correct scorers), what did it miss (substitutions, red cards), what did it invent.
4. For the worst case, investigate *why*: is it a low-quality transcription? An unusual match format? Very low energy levels throughout (no crowd noise)?

**Output:** One page in the report with the two case studies. Include the energy-over-time plot for the worst case to visualize transcription quality.

**Where to include:** Qualitative Analysis section, before Conclusion.

---

## Analysis 9 — Efficiency vs. Quality Trade-off

> **Claude Code prompt to implement this:**
> *"Implement Analysis 9 from analysis.md. Modify `scripts/run_prompting.py` and `scripts/run_inference_finetuned.py` to record, for each test match: wall-clock inference time (using `time.time()`), peak GPU memory in MB (using `torch.cuda.reset_peak_memory_stats()` before and `torch.cuda.max_memory_allocated() / 1e6` after), and number of model forward passes. Save these per-condition averages to `outputs/results/efficiency.csv`. Also update `src/evaluation/evaluate_all.py` to join efficiency.csv with metrics.csv when both exist, so the final output table includes the efficiency columns."*

**Why it matters:** Practical NLP requires balancing accuracy with cost. Comparing inference time and memory alongside ROUGE gives a complete picture.

**Steps:**

1. For each condition, record during inference:
   - Total inference time for all 9 test matches (seconds)
   - Peak GPU memory (MB) — use `torch.cuda.max_memory_allocated()` before and after
   - Average number of model forward passes per match
2. Add these columns to the main results table.

**Implementation:** Wrap each pipeline call in `scripts/run_prompting.py` and `scripts/run_inference_finetuned.py`:
```python
import time, torch
torch.cuda.reset_peak_memory_stats()
start = time.time()
summary = run_pipeline(...)
elapsed = time.time() - start
peak_mem_mb = torch.cuda.max_memory_allocated() / 1e6
```

**Output:** Extended results table with columns: ROUGE-L, BERTScore F1, Inference time (s/match), Peak VRAM (MB).

**Where to include:** Main results table.

---

## Implementation Priority Order

Run these analyses in this order — earlier items unblock later ones and have the highest mark impact:

| Priority | Analysis | Effort | Mark Impact |
|---|---|---|---|
| 1 | Acoustic feature-guided chunking (#4) | Medium | Very High — original contribution |
| 2 | Hallucination analysis (#1) | Low | Very High — expected at top level |
| 3 | Chunk size ablation (#3) | Low | High — justifies design choices |
| 4 | Training dynamics (#7) | Low | High — shows fine-tuning rigor |
| 5 | Qualitative case study (#8) | Low | High — makes report readable |
| 6 | Performance by match type (#2) | Medium | Medium-High |
| 7 | Faithfulness vs. coverage (#5) | Low | Medium — already computed by BERTScore |
| 8 | Prompt sensitivity (#6) | Medium | Medium |
| 9 | Efficiency trade-off (#9) | Low | Medium — polishing |

---

## Files to Add/Modify

| File | Purpose |
|---|---|
| `src/data/chunker.py` | Add `filter_high_energy_segments()` for Analysis 4 |
| `src/evaluation/metrics.py` | Return BERTScore P/R/F1 separately (already done) |
| `scripts/run_prompting.py` | Add `--max_tokens` flag for Analysis 3; add timing/memory logging for Analysis 9 |
| `scripts/run_analysis.py` | New script: orchestrates all post-hoc analyses (hallucination, energy plot, breakdown by type) |
| `notebooks/analysis.ipynb` | Optional: visualizations (energy plot, training curves, scatter plots) |
