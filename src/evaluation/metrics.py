"""
Evaluation metrics: ROUGE-1/2/L and BERTScore.
"""

from typing import List, Dict
import numpy as np

from rouge_score import rouge_scorer as rs


def compute_rouge(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    Compute ROUGE-1, ROUGE-2, ROUGE-L (F1) averaged over all prediction/reference pairs.
    """
    scorer = rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rl = [], [], []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref.strip(), pred.strip())
        r1.append(scores["rouge1"].fmeasure)
        r2.append(scores["rouge2"].fmeasure)
        rl.append(scores["rougeL"].fmeasure)
    return {
        "rouge1": float(np.mean(r1)),
        "rouge2": float(np.mean(r2)),
        "rougeL": float(np.mean(rl)),
    }


def compute_bertscore(
    predictions: List[str],
    references: List[str],
    model_type: str = "microsoft/deberta-xlarge-mnli",
    batch_size: int = 4,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Compute BERTScore (precision, recall, F1) averaged over all pairs.
    Uses batches to avoid OOM on Colab.
    """
    from bert_score import score as bert_score_fn

    P, R, F1 = bert_score_fn(
        predictions,
        references,
        model_type=model_type,
        batch_size=batch_size,
        device=device,
        verbose=False,
    )
    return {
        "bertscore_precision": float(P.mean()),
        "bertscore_recall": float(R.mean()),
        "bertscore_f1": float(F1.mean()),
    }
