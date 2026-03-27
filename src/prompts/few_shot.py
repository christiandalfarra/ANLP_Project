"""
Few-shot (2-shot) prompt templates for football match summarization.

Examples are selected once from the training split and fixed for all runs
to ensure reproducibility. Selection strategy: pick one high-scoring match
and one low-scoring match by scanning summary text for score patterns.
"""

import re
from typing import List, Tuple, Optional


def _extract_goals(summary: str) -> int:
    """Rough estimate of total goals from a summary string."""
    # Look for patterns like "3-1", "2–0", "1–1"
    scores = re.findall(r"(\d+)\s*[-–]\s*(\d+)", summary)
    if scores:
        a, b = scores[0]
        return int(a) + int(b)
    return 0


def select_examples(
    train_ids: List[str],
    summaries: dict,  # match_id -> summary str
    n: int = 2,
) -> List[str]:
    """
    Select `n` diverse example match IDs from train_ids.
    Strategy: pick the highest-scoring and lowest-scoring match.
    """
    scored = sorted(train_ids, key=lambda mid: _extract_goals(summaries.get(mid, "")))
    if len(scored) < n:
        return scored

    # Lowest-scoring (index 0) and highest-scoring (index -1)
    selected = [scored[0], scored[-1]]
    return selected[:n]


CHUNK_FEW_SHOT_TEMPLATE = (
    "You are a sports journalist. Summarize football match commentary excerpts.\n\n"
    "Here are two examples:\n\n"
    "{examples}"
    "Now summarize the following excerpt ({start:.0f}s - {end:.0f}s):\n"
    "Commentary:\n{text}\n\n"
    "Summary:"
)

EXAMPLE_TEMPLATE = (
    "Example {n}:\n"
    "Commentary:\n{commentary}\n\n"
    "Summary:\n{summary}\n\n"
    "---\n\n"
)

MERGE_FEW_SHOT_TEMPLATE = (
    "You are a sports journalist. Combine partial match summaries into a full match report.\n\n"
    "Here is an example:\n\n"
    "Partial summaries:\n{example_partials}\n\n"
    "Final match report:\n{example_final}\n\n"
    "---\n\n"
    "Now combine these partial summaries:\n{partial_summaries}\n\n"
    "Final match report:"
)


def build_chunk_prompt(
    text: str,
    start: float,
    end: float,
    examples: List[Tuple[str, str]],  # list of (commentary_excerpt, summary)
) -> str:
    examples_str = ""
    for i, (comm, summ) in enumerate(examples):
        examples_str += EXAMPLE_TEMPLATE.format(n=i + 1, commentary=comm[:800], summary=summ)
    return CHUNK_FEW_SHOT_TEMPLATE.format(
        examples=examples_str, text=text, start=start, end=end
    )


def build_merge_prompt(
    partial_summaries: List[str],
    example_partials: Optional[List[str]] = None,
    example_final: Optional[str] = None,
) -> str:
    joined = "\n\n---\n\n".join(f"[Segment {i+1}]\n{s}" for i, s in enumerate(partial_summaries))

    if example_partials and example_final:
        ex_joined = "\n\n---\n\n".join(
            f"[Segment {i+1}]\n{s}" for i, s in enumerate(example_partials)
        )
        return MERGE_FEW_SHOT_TEMPLATE.format(
            example_partials=ex_joined,
            example_final=example_final,
            partial_summaries=joined,
        )

    # Fall back to zero-shot merge if no example provided
    from src.prompts.zero_shot import build_merge_prompt as zs_merge
    return zs_merge(partial_summaries)
