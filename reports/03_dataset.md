# Dataset — `football_commentary_dataset 1.0`

A custom-built corpus of 99 football matches with audio commentary transcripts, sentence-level segments with prosodic features, and human-written reference summaries. Used as both training and evaluation data for the ANLP project.

Source: `https://github.com/panciut/football_commentary_dataset`. Local in-project copy is committed at `football_commentary_dataset/data/`.

---

## 1. Composition

99 matches in total, splits committed at `outputs/splits.json`:

| Split | n |
|---|---:|
| train | 80 |
| val | 10 |
| test | 9 |

### 1.1 Year coverage

Matches span **1983 to 2026** (the dataset is in active development; the most recent additions are 2025–2026 PL / UCL fixtures). Heavy concentration on 2018–2024 (Euro 2024, World Cup 2018/2022, recent UCL).

### 1.2 Competition mix

| Competition | n | Notes |
|---|---:|---|
| World Cup (`wc`) | 32 | group + KO stages, 2010–2022 |
| UEFA Champions League (`ucl`) | 17 | group + KO, 2017–2026 |
| Premier League (`pl`) | 15 | recent fixtures, 2011–2026 |
| Euros (`euro`) | 14 | tournament group/KO, 2004–2024 |
| UCL final (`uclfinal`) | 8 | finals only |
| WC qualifiers (`wcqual`) | 5 | 2010, 2026 |
| FA Cup (`facup`) | 2 | 1983 final, 2001 final |
| Nations League (`nl`) | 2 | |
| UEL final (`uelfinal`) | 1 | |
| Euro final (`eurofinal`) | 1 | 2024 |
| WC final (`wcfinal`) | 1 | |
| UECL final (`ueclfinal`) | 1 | |

The mix is England-heavy (most fixtures involve a home-nations team), with a few neutral fixtures (Spain–Germany, Argentina–Poland, etc.).

## 2. File layout per match

For each match `{id}` (e.g. `2001_facup_arsenal_liverpool`), three files exist:

```
football_commentary_dataset/data/
├── transcripts/
│   ├── {id}_transcript.txt   ~130 KB avg, range 9 KB–271 KB
│   ├── {id}_segments.txt     ~225 KB avg, range 139 KB–502 KB
│   └── {id}.json             ~580 KB avg, range 348 KB–1.3 MB
└── summaries/
    └── {id}.txt              ~1.2 KB avg, range 637 B–2.3 KB
```

Plus an out-of-tree `data/raw_audio/` directory of MP3 files (~19 GB total, gitignored — not needed at runtime).

### 2.1 Transcript (`{id}_transcript.txt`)

Plain text — the full commentary audio transcribed end-to-end via Whisper, no segmentation, no markup. Average length ~130 KB ≈ ~25,000 words for a 90-minute match.

Sample (first 300 chars of `2001_facup_arsenal_liverpool_transcript.txt`):

> *Vengas Arsenal look like this, Seaman in goal, a back four of Dixon, Keown, Adams and Cole. In midfield we have Ljungberg, Grimondi, Vieira and Perez, an upfront two more French World Cup stars Sylvain Wiltor and Thierry Henri. The substitutes for Arsenal today are Manninga, Parler, Burgkamp, Kanu...*

Note that Whisper transcription introduces typos: "Vengas" → "Wenger's", "Manninga" → "Manninger", "Burgkamp" → "Bergkamp". These propagate through every downstream pipeline and are part of the evaluation noise.

### 2.2 Segments (`{id}_segments.txt`)

Same content as the transcript, but split into Whisper-decoded utterances with start/end timestamps and prosodic features (pitch in Hz, energy as RMS) per segment.

Sample:

```
[0.00 - 6.80]  P:1767.91 E:0.031016 |  Vengas Arsenal look like this, Seaman in goal, a back four of Dixon, Keown, Adams and Cole.
[6.80 - 14.24] P:1900.59 E:0.046202 |  In midfield we have Ljungberg, Grimondi, Vieira and Perez, an upfront two more French World Cup stars
[14.24 - 21.04] P:1904.70 E:0.048367 |  Sylvain Wiltor and Thierry Henri. The substitutes for Arsenal today are Manninga,
```

This is the **primary input** to the chunk+aggregate pipeline (`src/data/chunker.py` chunks at segment boundaries up to a token budget). The pitch/energy fields are not used by any current model but are available for prosody-aware extensions.

### 2.3 Metadata (`{id}.json`)

Per-segment structured JSON with the same content as the segments file plus full token-level alignment (start/end per token, confidence scores). Used by `src/data/dataset_loader.py` to build the `Match` and `Segment` dataclasses. Not human-readable but the ground truth for any post-hoc time-alignment queries.

### 2.4 Reference summary (`{id}.txt`)

A short human-written match report of typically 600–2300 characters (~150–500 words), structured as:

> Line 1: `{TeamA} {scoreA}–{scoreB} {TeamB}`
> Line 2: `{Competition}, {Stadium}, {City} – {Date}`
> Lines 3+: 2–4 paragraph match report covering the result, key events (goals, cards, substitutions), and a brief tactical or contextual note.

Sample (`2001_facup_arsenal_liverpool.txt`):

> *Arsenal 1–2 Liverpool*
> *2001 FA Cup Final, Millennium Stadium, Cardiff – 12 May 2001*
>
> *Liverpool completed a dramatic late comeback to win the 2001 FA Cup Final against Arsenal. Arsenal opened the scoring in the 72nd minute when Freddie Ljungberg finished after good build-up play to put the Gunners ahead. Arsenal maintained the lead for much of the match and created several chances, but could not extend their advantage…*

## 3. Why the in-project copy

`src/data/dataset_loader.py` resolves the dataset in three places (in order):
1. `FOOTBALL_DATASET_DIR` env var.
2. `<repo>/football_commentary_dataset/` — the in-project vendored copy.
3. `../football_commentary_dataset/` — sibling layout used in local development.

The Kaggle notebook clones the GitHub repo and then uses path #2. **This is why the dataset must be committed inside the ANLP_Project repo** — Kaggle has no internet access to clone a separate repo without explicit dataset attachment.

Trade-off: 90 MB of transcripts + segments + JSON metadata bloat the repo, but the alternative (a Kaggle Dataset attachment workflow per kernel run) was more brittle.

## 4. Splits

`outputs/splits.json` was generated once by `scripts/generate_splits.py` and is committed. The split is fixed and reproducible — every run uses the same train/val/test partition.

| Split | n | Selection |
|---|---:|---|
| train | 80 | bulk of the dataset |
| val | 10 | for early stopping during fine-tuning |
| test | 9 | for final evaluation |

The 9 test matches:

```
2001_facup_arsenal_liverpool         FA Cup Final
2021_euro_england_ukraine            Euro 2020 KO
2019_uclfinal_liverpool_tottenham    UCL Final
2022_wc_poland_argentina             WC Group D
1983_facup_final_brighton_manchesterunited  FA Cup Final
2024_euro_england_slovakia           Euro 2024 KO
2026_ucl_athleticclub_arsenal        UCL Group
2022_wc_england_senegal              WC R16
2021_euro_czechrepublic_england      Euro 2020 KO
```

The split mixes competitions, eras (1983–2026), and result types (one-sided wins, late comebacks, penalties, low-scoring draws). Useful for testing generalisation but **9 is small** — see Limitations in `01_session1_results.md`.

## 5. The empty-references incident

Yesterday's runs (2026-05-03) used dataset version 0.7, which had **33 of 99 reference summaries as 0-byte files** (26 train / 3 val / 4 test). The local copy was stale; the upstream `panciut/football_commentary_dataset` repo had a newer commit (`dataset 1.0`) where all 99 references were filled in.

The empty references caused two compounding problems:

1. **Training poisoned.** 33% of training pairs were `(transcript, "")`. The model literally learned that for one-third of inputs the correct output is empty. Especially harmful for LED, which exhibited persistent "collapse to `<s></s>`" behaviour for several training runs.
2. **Evaluation deflated.** ROUGE on an empty reference is exactly 0 for any non-empty prediction. The reported test ROUGE-L was the average over 9 matches, 4 of which always scored 0 — a ~1.8× understate of the true valid-match average.

After pulling `dataset 1.0` and re-running:

| Condition | v0.7 (broken) | v1.0 (clean) | Δ |
|---|---:|---:|---:|
| finetuned_bart | 0.1361 | 0.2476 | +82% |
| led_long_zero | 0.0825 | 0.1599 | +94% |
| flan_chunk_zero | 0.0413 | 0.0700 | +69% |

Discussed in detail in `02_engineering_history.md` § Phase 4.

## 6. Stats summary

| Stat | Value |
|---|---|
| Matches | 99 |
| Years covered | 1983–2026 (43 years) |
| Competitions | 12 (WC, UCL, PL, Euro, FA Cup, etc.) |
| Train / val / test | 80 / 10 / 9 |
| Avg transcript size | 130 KB (~25k words) |
| Avg segments size | 225 KB |
| Avg reference summary | 1.2 KB (~150–500 words) |
| Total dataset on disk (excl. raw audio) | ~91 MB |
| Total raw audio (gitignored) | ~19 GB |

## 7. Strengths and limitations

### Strengths

- **End-to-end pipeline**: raw audio → Whisper transcripts → segment-aligned timed text → human reference summaries. Reproducible from source.
- **Diverse competitions and eras** (43-year span). Tests generalisation across decades of commentary style and broadcast quality.
- **Prosodic features available** (pitch, energy per segment). Currently unused but enable future prosody-aware extensions.
- **Both segmented and unsegmented inputs** are available, supporting both chunk+aggregate (segments) and long-context (transcript) pipelines without preprocessing.

### Limitations

- **Small (n=99)**, especially for fine-tuning. With only 80 training pairs, models with hundreds of millions of parameters overfit quickly.
- **Transcription noise**: Whisper makes systematic errors on player names ("Bergkamp" → "Burgkamp", "Wenger's" → "Vengas"). These propagate through every model output and depress every metric that compares surface tokens.
- **Older audio is harder.** Pre-2010 matches have lower-quality recordings; transcription quality degrades correspondingly. The 1983 FA Cup final transcript is noticeably noisier than the 2024 Euro Final.
- **English-language commentary only.** Non-English-speaking teams (Spain, Germany, Argentina) appear, but the commentary is always English broadcast audio.
- **Very recent matches** (2025–2026) included for variety, but reference summaries for these are necessarily based on retrospective writing — and Whisper transcription quality on the most recent matches has not been spot-checked.
- **Versioning was loose** — the empty-references issue went undiagnosed for ~36 hours and silently destroyed two days of training results. A pinned dataset version with checksums would have caught it immediately.

## 8. Recommended workflow if revising the dataset

1. **Pin a version** in the project (e.g. as a git submodule with a commit SHA). Avoids silent staleness.
2. **Add a sanity-check script** that asserts:
   - Every match has all four files (transcript, segments, json, summary).
   - Every reference summary is ≥ 200 bytes.
   - Every transcript is ≥ 1000 chars.
   Run it as a pre-flight in every notebook.
3. **Filter at load time** by reference quality (e.g. drop matches with reference < 300 bytes), to avoid future poisoning if upstream regresses.
4. **Document Whisper version** used to produce transcripts, so re-runs are reproducible if the audio is re-transcribed.
