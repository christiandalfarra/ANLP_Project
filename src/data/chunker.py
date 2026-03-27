"""
Token-count-based chunking of match transcripts, aligned to segment boundaries.

Each chunk is a list of Segment objects. The caller is responsible for joining
segment texts for model input.
"""

from typing import List, Tuple
from transformers import PreTrainedTokenizerBase

from src.data.dataset_loader import Segment


def chunk_segments(
    segments: List[Segment],
    tokenizer: PreTrainedTokenizerBase,
    max_tokens: int,
    overlap: int = 1,
) -> List[List[Segment]]:
    """
    Split segments into chunks where each chunk's token count <= max_tokens.
    Adds `overlap` segments from the previous chunk at the start of each new chunk
    to avoid hard cuts mid-context.

    Args:
        segments: List of Segment objects for the full match.
        tokenizer: HuggingFace tokenizer used to count tokens.
        max_tokens: Maximum number of tokens per chunk.
        overlap: Number of trailing segments from the previous chunk to prepend.

    Returns:
        List of chunks, each chunk being a list of Segment objects.
    """
    chunks: List[List[Segment]] = []
    current_chunk: List[Segment] = []
    current_token_count = 0

    for seg in segments:
        seg_tokens = len(tokenizer.encode(seg.text, add_special_tokens=False))

        if current_token_count + seg_tokens > max_tokens and current_chunk:
            chunks.append(current_chunk)
            # Start new chunk with overlap from the end of the previous chunk
            current_chunk = current_chunk[-overlap:] if overlap > 0 else []
            current_token_count = sum(
                len(tokenizer.encode(s.text, add_special_tokens=False))
                for s in current_chunk
            )

        current_chunk.append(seg)
        current_token_count += seg_tokens

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def chunk_text(
    segments: List[Segment],
    tokenizer: PreTrainedTokenizerBase,
    max_tokens: int,
    overlap: int = 1,
) -> List[Tuple[str, float, float]]:
    """
    Convenience wrapper: returns list of (text, start_time, end_time) tuples.
    """
    raw_chunks = chunk_segments(segments, tokenizer, max_tokens, overlap)
    result = []
    for chunk in raw_chunks:
        text = " ".join(s.text for s in chunk)
        start = chunk[0].start
        end = chunk[-1].end
        result.append((text, start, end))
    return result
