"""
Stratified train/val/test split for the football dataset.

Stratification is done by competition type inferred from match IDs:
  - world_cup (wc)
  - euro (euro / ec)
  - fa_cup (fa_cup)
  - other (premier league, qualifiers, etc.)

Split: 80 train / 10 val / 9 test  (totals ~99 matches).
"""

import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple

SPLITS_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "splits.json")


def _competition_key(match_id: str) -> str:
    mid = match_id.lower()
    if "wc" in mid or "world_cup" in mid:
        return "world_cup"
    if "euro" in mid or "_ec_" in mid:
        return "euro"
    if "fa_cup" in mid or "fac" in mid:
        return "fa_cup"
    return "other"


def create_splits(
    match_ids: List[str],
    val_size: int = 10,
    test_size: int = 9,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """
    Create stratified train/val/test splits.
    Returns dict with keys 'train', 'val', 'test'.
    """
    rng = random.Random(seed)

    # Group by competition
    by_competition: Dict[str, List[str]] = defaultdict(list)
    for mid in match_ids:
        by_competition[_competition_key(mid)].append(mid)

    # Shuffle within each group
    for group in by_competition.values():
        rng.shuffle(group)

    # Flatten in a stratified order (round-robin across groups)
    groups = list(by_competition.values())
    stratified: List[str] = []
    max_len = max(len(g) for g in groups)
    for i in range(max_len):
        for g in groups:
            if i < len(g):
                stratified.append(g[i])

    # Assign splits
    test = stratified[:test_size]
    val = stratified[test_size:test_size + val_size]
    train = stratified[test_size + val_size:]

    return {"train": train, "val": val, "test": test}


def save_splits(splits: Dict[str, List[str]], path: str = SPLITS_FILE) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)
    print(f"Splits saved to {path}")
    print(f"  train={len(splits['train'])}  val={len(splits['val'])}  test={len(splits['test'])}")


def load_splits(path: str = SPLITS_FILE) -> Dict[str, List[str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Splits file not found at {path}. "
            "Run `python scripts/generate_splits.py` first."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
