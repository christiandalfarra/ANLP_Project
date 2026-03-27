"""
One-time script to generate and save train/val/test splits.
Run this once, then commit outputs/splits.json for reproducibility.

Usage:
    python scripts/generate_splits.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.dataset_loader import list_match_ids
from src.data.splits import create_splits, save_splits


def main():
    match_ids = list_match_ids()
    print(f"Total matches: {len(match_ids)}")

    splits = create_splits(match_ids, val_size=10, test_size=9, seed=42)
    print(f"Train: {len(splits['train'])}  Val: {len(splits['val'])}  Test: {len(splits['test'])}")

    save_splits(splits)

    print("\nTest set matches:")
    for mid in splits["test"]:
        print(f"  {mid}")


if __name__ == "__main__":
    main()
