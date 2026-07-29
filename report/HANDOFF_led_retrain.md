# Handoff — execute the clean LED retrain (Task A), local GPU

This file is written so a **fresh Claude Code instance running on the GPU PC** can
drive Task A to completion. Training is done locally (not Kaggle), so — unlike the
browser flow — a Claude instance with terminal access can run *every* step itself,
including the training command. The human only needs to: provide the GPU machine,
approve the long-running training command, and recompile the PDF at the end.

## Kickoff prompt (paste into the new Claude instance on the GPU PC)

> I'm in the ANLP_Project repo on a machine with a CUDA GPU (~15 GB VRAM).
> Execute Task A from `report/HANDOFF_led_retrain.md`: retrain LED on the clean
> dataset, regenerate all analysis outputs, and update `report/main.tex`. Run the
> steps yourself (the training command is long-running — launch it in the
> background and wait). When done, commit and push a branch so I can pull it and
> recompile `main.pdf` on Overleaf. Read `report/RUNBOOK_led_retrain.md` for the
> detailed report-edit reference.

---

## Background (why this task exists)

The report's `finetuned_led` result comes from `run3`, trained on a *poisoned*
dataset (33% of references were empty 0-byte files) and only re-scored on clean
references afterwards. That makes the #1-vs-#2 (BART-vs-LED) comparison unfair on
the training side. Task A retrains LED on the clean
`football_commentary_dataset 1.0` (already committed in-repo) and folds the result
into the report.

## Requirements on the GPU PC

- **CUDA GPU, ~15 GB VRAM** (T4-class). The config uses batch 1 + gradient
  checkpointing + fp16 to fit; a smaller card may OOM. fp16 is enabled only when
  CUDA is present — do not train on CPU/MPS (it will not fit / will crawl).
- Python env with `pip install -r requirements.txt`. The `spacy`/`matplotlib`/
  `bert_score` deps are only needed for the analysis phase (steps 4–5), not
  training; installing everything up front is simplest. Also run
  `python -m spacy download en_core_web_sm` before step 4.
- Internet (step 5 downloads roberta-large + deberta-mnli for the semantic metrics).

---

## Steps (a Claude instance on the GPU PC can run all of these)

### 1. Get the repo and confirm the dataset is clean

```bash
git pull            # or clone the repo, then cd into it
python scripts/check_dataset.py
# expect: "OK: all 99 matches ... reference >= 200B."
```

If the preflight FAILS, stop — the dataset is stale; get the clean
`football_commentary_dataset 1.0` before training.

### 2. Train LED (long-running, ~hours on a T4; launch in background)

```bash
python scripts/run_finetuning.py --model led --output_dir checkpoints/led
```

This has a built-in guard that aborts if any train/val reference is empty. It
trains lr 2e-5, warmup 100, max input 8192, max target 256, patience 2, up to 15
epochs (early-stops). Best model by val ROUGE-L is saved to `checkpoints/led/`.

### 3. Inference on the 9 test matches (GPU, ~10–20 min)

```bash
python scripts/run_inference_finetuned.py --model led --checkpoint checkpoints/led
# writes outputs/predictions/finetuned_led.json
```

### 4. Bundle the run into `runs/run5_led_clean_<date>/`

The `trainer_state.json` (full per-epoch log_history, needed for the training
curve) lives in the newest checkpoint dir — copy it too:

```bash
DATE=$(date +%Y-%m-%d)
RUN="runs/run5_led_clean_$DATE"
mkdir -p "$RUN/predictions"
cp outputs/predictions/finetuned_led.json "$RUN/predictions/"
cp "$(ls -t checkpoints/led/checkpoint-*/trainer_state.json | head -1)" "$RUN/trainer_state.json"
ls -R "$RUN"   # expect predictions/finetuned_led.json and trainer_state.json
```

### 5. Regenerate all analysis outputs (auto-detect run5 — no path edits)

```bash
python -m spacy download en_core_web_sm   # if not already
python report/compute_analysis.py     # -> report/results/faithfulness.csv, percondition_cis.csv
python report/compute_semantic.py     # -> report/results/semantic.csv  (needs internet + bert_score)
cd report
python make_led_curve.py              # auto-reads run5 trainer_state.json
python make_leaderboard_chart.py
python make_inversion_chart.py
cd ..
```

Capture the new `finetuned_led` numbers `compute_analysis.py` prints: ROUGE-1/2/L,
its 95% CI, the paired-bootstrap Δ vs `finetuned_bart` and whether it is now
significant, plus the faithfulness and semantic rows.

### 6. Update `report/main.tex` (Claude does this)

Replace every run3 LED figure with the run5 one:

- **Table `tab:main`** (leaderboard): `finetuned_led` R-1/R-2/R-L + CI. Update the
  caption's paired-bootstrap footnote (currently `Δ = +0.029, CI [-0.014,+0.070]`)
  and flip "not significant" ↔ "significant" if it changed.
- **Delete the `$^{\dagger}$` footnote** on the `finetuned_led` row and the
  "trained before the empty-reference fix but re-scored" caveat everywhere it
  appears (Observation 2, Engineering §, Conclusions bullet 2) — LED is now
  clean-trained.
- **Table `tab:faith`** and **Table `tab:semantic`**: the `finetuned_led` row.
- **Observation (2)** prose ("0.248 vs 0.219 ... paired bootstrap ..."): update the
  LED number and the significance verdict.
- **Engineering §**: "we later re-scored them ... 0.219" becomes a clean-training
  result, not a re-score. KEEP the five-silent-failures history — only the
  provenance of the final number changes.
- **Abstract** / **Limitations val→test gap**: update only if a quoted LED number
  moved enough to change a stated claim.

If clean-trained LED now *beats* BART: keep the "statistically indistinguishable at
n=9" conclusion (defensible either way); just swap which model is "nominally best".

### 7. Commit, push, hand back

```bash
git checkout -b task-a/led-clean-retrain
git add runs/run5_led_clean_* report/results report/figures report/main.tex
git commit -m "Task A: clean LED retrain (run5); update report LED numbers"
git push -u origin task-a/led-clean-retrain
```

Tell the human to pull that branch on the main machine and recompile
`report/main.tex` on Overleaf (no LaTeX toolchain is installed locally). Confirm
the Task B/C additions still render (prosody subsection + figure, NER-precision
column in Table 5).

---

## Definition of done

- `runs/run5_led_clean_<date>/` committed (predictions + trainer_state).
- `report/results/*.csv` regenerated with clean LED numbers.
- `report/figures/led_training_curve.pdf` regenerated from run5.
- `main.tex` LED numbers updated, dagger caveat removed, significance verdict
  correct.
- Branch pushed; `main.pdf` recompiled on Overleaf and visually checked.
