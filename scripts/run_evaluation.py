"""
Run evaluation across all saved prediction files.

Usage:
    # CPU (no BERTScore model download needed if skipped)
    python scripts/run_evaluation.py

    # With BERTScore on GPU
    python scripts/run_evaluation.py --bertscore --device cuda

    # Skip BERTScore (faster, ROUGE only)
    python scripts/run_evaluation.py --no-bertscore

Output:
    outputs/results/metrics.csv
    outputs/results/metrics_table.tex
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.dataset_loader import load_match
from src.data.splits import load_splits
from src.evaluation.evaluate_all import evaluate_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-bertscore", dest="bertscore", action="store_false",
                        help="Skip BERTScore computation (faster)")
    parser.add_argument("--device", default="cpu",
                        help="Device for BERTScore: 'cuda' or 'cpu'")
    parser.set_defaults(bertscore=True)
    args = parser.parse_args()

    splits = load_splits()
    test_ids = splits["test"]

    print(f"Test set: {len(test_ids)} matches")
    references = {}
    for mid in test_ids:
        m = load_match(mid)
        references[mid] = m.summary or ""

    evaluate_all(
        test_ids=test_ids,
        references=references,
        use_bertscore=args.bertscore,
        bertscore_device=args.device,
    )


if __name__ == "__main__":
    main()
