"""
Aggregate evaluation: load all saved predictions and compute ROUGE + BERTScore
for every experimental condition. Outputs metrics.csv and a LaTeX table.
"""

import csv
import json
import os
from typing import Dict, List

from src.evaluation.metrics import compute_rouge, compute_bertscore


PREDICTIONS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "predictions")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "results")


def load_predictions(predictions_dir: str = PREDICTIONS_DIR) -> Dict[str, Dict[str, str]]:
    """
    Load all prediction JSON files.
    Returns dict: condition_name -> {match_id: generated_summary}
    """
    all_preds = {}
    for fname in sorted(os.listdir(predictions_dir)):
        if fname.endswith(".json"):
            condition = fname[:-5]
            with open(os.path.join(predictions_dir, fname), "r", encoding="utf-8") as f:
                all_preds[condition] = json.load(f)
    return all_preds


def evaluate_all(
    test_ids: List[str],
    references: Dict[str, str],  # match_id -> ground truth summary
    use_bertscore: bool = True,
    bertscore_device: str = "cpu",
    predictions_dir: str = PREDICTIONS_DIR,
    results_dir: str = RESULTS_DIR,
) -> Dict[str, Dict]:
    """
    Evaluate all conditions and write results to CSV + LaTeX.

    Args:
        test_ids: List of match IDs in the test set.
        references: Dict mapping match_id to ground truth summary.
        use_bertscore: Whether to compute BERTScore (slow without GPU).
        bertscore_device: 'cuda' or 'cpu'.
    """
    os.makedirs(results_dir, exist_ok=True)
    all_preds = load_predictions(predictions_dir)

    results = {}
    for condition, preds in all_preds.items():
        pred_list = [preds.get(mid, "") for mid in test_ids]
        ref_list = [references.get(mid, "") for mid in test_ids]

        rouge = compute_rouge(pred_list, ref_list)
        metrics = {"condition": condition, **rouge}

        if use_bertscore:
            bs = compute_bertscore(pred_list, ref_list, device=bertscore_device)
            metrics.update(bs)

        results[condition] = metrics
        print(f"{condition}: ROUGE-1={rouge['rouge1']:.4f}  ROUGE-2={rouge['rouge2']:.4f}  ROUGE-L={rouge['rougeL']:.4f}")

    # Write CSV
    csv_path = os.path.join(results_dir, "metrics.csv")
    fieldnames = list(next(iter(results.values())).keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results.values())
    print(f"\nResults saved to {csv_path}")

    # Write LaTeX table
    _write_latex(results, os.path.join(results_dir, "metrics_table.tex"))

    return results


def _write_latex(results: Dict[str, Dict], path: str) -> None:
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"\textbf{Condition} & \textbf{ROUGE-1} & \textbf{ROUGE-2} & \textbf{ROUGE-L} & \textbf{BERTScore F1} \\",
        r"\hline",
    ]
    for cond, m in sorted(results.items()):
        r1 = f"{m.get('rouge1', 0):.4f}"
        r2 = f"{m.get('rouge2', 0):.4f}"
        rl = f"{m.get('rougeL', 0):.4f}"
        bs = f"{m.get('bertscore_f1', 0):.4f}" if "bertscore_f1" in m else "-"
        lines.append(rf"{cond.replace('_', ' ')} & {r1} & {r2} & {rl} & {bs} \\")
    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\caption{Comparison of summarization approaches on the test set.}",
        r"\label{tab:results}",
        r"\end{table}",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"LaTeX table saved to {path}")
