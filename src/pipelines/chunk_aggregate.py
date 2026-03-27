"""
Chunk + Aggregate pipeline.

Stage 1: Split transcript into token-count-based chunks (aligned to segment boundaries).
         Summarize each chunk independently with a prompter or fine-tuned model.
Stage 2: Concatenate intermediate summaries and merge into a final summary.

Post-processing: remove adjacent near-duplicate sentences.
"""

from typing import List, Tuple, Callable, Optional
import re

from src.data.dataset_loader import Match
from src.data.chunker import chunk_text


def _cosine_sim(a: str, b: str) -> float:
    """Simple bag-of-words cosine similarity for deduplication."""
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    return len(intersection) / ((len(tokens_a) * len(tokens_b)) ** 0.5)


def _deduplicate_sentences(text: str, threshold: float = 0.92) -> str:
    """Remove consecutive near-duplicate sentences."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if not sentences:
        return text
    kept = [sentences[0]]
    for sent in sentences[1:]:
        if _cosine_sim(sent, kept[-1]) < threshold:
            kept.append(sent)
    return " ".join(kept)


def run_chunk_aggregate(
    match: Match,
    tokenizer,
    generate_fn: Callable[[str, float, float], str],
    merge_fn: Callable[[List[str]], str],
    max_tokens_per_chunk: int,
    overlap: int = 1,
) -> str:
    """
    Full chunk-and-aggregate pipeline for a single match.

    Args:
        match: Match object with segments.
        tokenizer: HuggingFace tokenizer for counting tokens.
        generate_fn: Function(text, start, end) -> chunk_summary string.
        merge_fn: Function(list_of_summaries) -> final_summary string.
        max_tokens_per_chunk: Token budget per chunk.
        overlap: Number of overlap segments between chunks.

    Returns:
        Final merged summary string.
    """
    chunks = chunk_text(match.segments, tokenizer, max_tokens_per_chunk, overlap)

    if not chunks:
        return ""

    # Stage 1: summarize each chunk
    chunk_summaries = []
    for i, (text, start, end) in enumerate(chunks):
        print(f"  Chunk {i+1}/{len(chunks)} ({start:.0f}s - {end:.0f}s, ~{len(text.split())} words)")
        summary = generate_fn(text, start, end)
        chunk_summaries.append(summary)

    if len(chunk_summaries) == 1:
        return _deduplicate_sentences(chunk_summaries[0])

    # Stage 2: merge
    print(f"  Merging {len(chunk_summaries)} chunk summaries...")
    final = merge_fn(chunk_summaries)
    return _deduplicate_sentences(final)
