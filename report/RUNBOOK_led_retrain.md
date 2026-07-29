# Runbook — clean LED retrain (Task A)

**Goal.** The reported `finetuned_led` result comes from `run3`, which was trained
on the *poisoned* dataset (33% empty references) and only re-scored on clean
references. That makes the #1-vs-#2 (BART-vs-LED) comparison unfair on the
training side. This runbook retrains LED on the clean `football_commentary_dataset 1.0`
and folds the result back into the report.

Everything except the GPU training runs locally. Training **must** run on a
Kaggle T4 (LED at 8192 tokens + gradient checkpointing does not fit on CPU/MPS).

---

## 0. Preconditions

- The Kaggle notebook clones `christiandalfarra/ANLP_Project@setup/local-run`.
  **That branch must contain the clean dataset** (all 99 references ≥ 200 B).
  The notebook's new preflight cell (`scripts/check_dataset.py`) aborts training
  if it does not — so if the preflight fails, push the clean dataset to that
  branch first.
- Config is already pinned in `src/models/finetuning/led_finetuner.py`
  (lr `2e-5`, warmup 100, max input 8192, max target 256, patience 2). Do not
  change it — the report Methodology now documents exactly these values.

## 1. Run on Kaggle (~3–4 h)

Open `notebooks/kaggle_session2_led_finetuning.ipynb`:

- Accelerator → **GPU T4 x2**, Internet → **On**.
- **Save & Run All (Commit)**.

The notebook now: preflight → train → inference → eval → **bundle**. The final
cell writes and prints `outputs/run5_led_clean_<date>.zip`.

## 2. Download and unpack locally

Download that zip from the Kaggle output panel and unpack into `runs/`:

```bash
mkdir -p runs/run5_led_clean_<date>
unzip ~/Downloads/run5_led_clean_<date>.zip -d runs/run5_led_clean_<date>
# should contain: predictions/finetuned_led.json, results/metrics.csv, trainer_state.json
```

The analysis scripts auto-detect `runs/run5_led_clean_*/predictions/finetuned_led.json`
and use it in place of run3 — no path edits needed.

## 3. Regenerate all analysis outputs (local)

```bash
python report/compute_analysis.py     # -> results/faithfulness.csv, percondition_cis.csv
python report/compute_semantic.py     # -> results/semantic.csv   (needs bert_score)
cd report && python make_led_curve.py         # auto-reads run5 trainer_state.json
python make_leaderboard_chart.py
python make_inversion_chart.py
```

Note the new **clean-trained LED numbers** printed by `compute_analysis.py`:
`finetuned_led` ROUGE-L, its 95% CI, the paired-bootstrap Δ vs `finetuned_bart`
(and whether it is now significant), plus faithfulness and semantic rows.

## 4. Update `report/main.tex`

Replace the run3 LED figures with the run5 ones. The values live in:

- **Table `tab:main`** (leaderboard): `finetuned_led` R-1/R-2/R-L + CI. Update the
  paired-bootstrap footnote (`Δ = +0.029, CI [-0.014,+0.070]`) with the new Δ/CI,
  and flip "not significant" ↔ "significant" if it changed.
- **Remove the `$^{\dagger}$` footnote** on the `finetuned_led` row — it no longer
  applies (clean training), and delete the "trained before the empty-reference
  fix but re-scored" caveat wherever it appears (Observation 2, Engineering §,
  Conclusions bullet 2).
- **Table `tab:faith`** and **Table `tab:semantic`**: `finetuned_led` row.
- **Observation (2)** prose ("0.248 vs 0.219"): update the LED number and the
  significance verdict.
- **Abstract**: only if the LED number moves enough to change a stated claim.
- **Limitations "val→test gap"**: update the LED val/test numbers if quoted.
- **Engineering §**: the sentence "we later re-scored them ... 0.219" becomes a
  clean-training result, not a re-score. Keep the *five-silent-failures* history
  (still true and valuable) — only the provenance of the final number changes.

## 5. Recompile

No LaTeX toolchain is installed locally — recompile `report/main.tex` on Overleaf
(or wherever `main.pdf` is built). Confirm the two new tables/figures from Tasks B
and C (prosody subsection + figure, NER-precision column) render correctly too.

---

### If the retrain changes the ranking

If clean-trained LED overtakes BART, the honest framing already in the report
still holds: at n=9 the two fine-tuned pipelines are within noise. Update the
"nominally best" wording to match whichever is nominally ahead, and keep the
"statistically indistinguishable" conclusion — that is the defensible claim
either way.
