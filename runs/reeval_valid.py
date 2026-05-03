"""Re-evaluate all predictions on the 5 test matches that have non-empty references."""
import json, os
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
splits = json.load(open(r'outputs/splits.json'))
test = splits['test']
SUMM = r'../football_commentary_dataset/data/summaries'
valid = [m for m in test if os.path.getsize(f'{SUMM}/{m}.txt') > 0]
print(f"{len(valid)}/{len(test)} test matches with non-empty references:")
for m in valid: print(f"  - {m}")
print()
refs = {m: open(f'{SUMM}/{m}.txt', encoding='utf-8').read().strip() for m in valid}

print(f"{'Condition':38s} {'R1':>7s} {'R2':>7s} {'RL':>7s}")
print('-' * 65)
rows = []
for run in ['run1_baseline_2026-05-02', 'run2_bart_merger_2026-05-03', 'run3_led_finetuned_2026-05-03']:
    p = f'runs/{run}/predictions'
    if not os.path.isdir(p): continue
    for f in sorted(os.listdir(p)):
        d = json.load(open(os.path.join(p, f), encoding='utf-8'))
        rs = []
        for m in valid:
            if m in d:
                s = scorer.score(refs[m], d[m])
                rs.append((s['rouge1'].fmeasure, s['rouge2'].fmeasure, s['rougeL'].fmeasure))
        if not rs: continue
        n = len(rs)
        r1 = sum(x[0] for x in rs)/n
        r2 = sum(x[1] for x in rs)/n
        rl = sum(x[2] for x in rs)/n
        cond = f.replace('.json', '')
        prefix = run.split('_')[0]
        rows.append((rl, prefix, cond, r1, r2))

for rl, prefix, cond, r1, r2 in sorted(rows, key=lambda x: -x[0]):
    label = f"{prefix}/{cond}"[:37]
    print(f"{label:38s} {r1:7.3f} {r2:7.3f} {rl:7.3f}")
