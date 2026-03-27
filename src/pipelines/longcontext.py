"""
Long-context (LED) pipeline.

Feeds the full match transcript in a single inference pass.
The transcript is truncated to the model's maximum input length if needed.
"""

from typing import Callable
from src.data.dataset_loader import Match


def run_longcontext(
    match: Match,
    generate_fn: Callable[[str], str],
) -> str:
    """
    Single-pass summarization for a full match transcript.

    Args:
        match: Match object.
        generate_fn: Function(full_text) -> summary string.

    Returns:
        Generated summary string.
    """
    return generate_fn(match.transcript)
