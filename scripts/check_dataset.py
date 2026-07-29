"""Dataset preflight: fail loudly if any split references are missing or empty.

The single most costly bug in this project was a stale dataset copy with 33% of
reference summaries as 0-byte files, which silently taught the models to emit
nothing and deflated every metric ~1.8x for two days. This script makes that
class of failure impossible to miss: run it before any training run.

Usage:
    python scripts/check_dataset.py                 # check all split matches
    python scripts/check_dataset.py --min-bytes 200 # custom floor
    python scripts/check_dataset.py --splits train val
Exit code is non-zero if any check fails.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.dataset_loader import DATASET_DIR
from src.data.splits import load_splits

SUMM = os.path.join(DATASET_DIR, "data", "summaries")
TRANS = os.path.join(DATASET_DIR, "data", "transcripts")


def check(splits_to_check, min_bytes):
    splits = load_splits()
    problems = []
    checked = 0
    for split in splits_to_check:
        for mid in splits[split]:
            checked += 1
            summ = os.path.join(SUMM, mid + ".txt")
            trans = os.path.join(TRANS, mid + "_transcript.txt")
            segs = os.path.join(TRANS, mid + "_segments.txt")
            if not os.path.exists(summ):
                problems.append(f"[{split}] {mid}: summary file MISSING")
            elif os.path.getsize(summ) < min_bytes:
                problems.append(f"[{split}] {mid}: summary only "
                                f"{os.path.getsize(summ)}B (< {min_bytes})")
            if not os.path.exists(trans):
                problems.append(f"[{split}] {mid}: transcript file MISSING")
            if not os.path.exists(segs):
                problems.append(f"[{split}] {mid}: segments file MISSING")
    return checked, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-bytes", type=int, default=200,
                    help="minimum acceptable reference summary size (bytes)")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = ap.parse_args()

    print(f"Dataset: {DATASET_DIR}")
    checked, problems = check(args.splits, args.min_bytes)
    if problems:
        print(f"\nFAILED: {len(problems)} problem(s) across {checked} matches:")
        for p in problems:
            print("  -", p)
        print("\nDo NOT train on this dataset — pull the clean version first.")
        sys.exit(1)
    print(f"OK: all {checked} matches in {args.splits} have transcript, "
          f"segments, and reference >= {args.min_bytes}B.")


if __name__ == "__main__":
    main()
