"""
Zero-shot prompt templates for football match summarization.
"""


CHUNK_SYSTEM = (
    "You are a sports journalist. Summarize the football match commentary excerpt below. "
    "Focus on goals, key moments, cards, and substitutions. "
    "Be concise and factual."
)

CHUNK_TEMPLATE = (
    "{system}\n\n"
    "Commentary ({start:.0f}s - {end:.0f}s):\n{text}\n\n"
    "Summary:"
)

MERGE_TEMPLATE = (
    "You are a sports journalist. "
    "Below are partial summaries from different segments of the same football match. "
    "Combine them into a single coherent match report. "
    "Remove any repetition. Preserve all goals, key events, and the final result.\n\n"
    "Partial summaries:\n{partial_summaries}\n\n"
    "Final match report:"
)

LONGCONTEXT_TEMPLATE = (
    "You are a sports journalist. Summarize the following full football match commentary. "
    "Focus on goals, key moments, cards, substitutions, and the final result. "
    "Be concise and factual.\n\n"
    "Commentary:\n{text}\n\n"
    "Summary:"
)


def build_chunk_prompt(text: str, start: float, end: float) -> str:
    return CHUNK_TEMPLATE.format(system=CHUNK_SYSTEM, text=text, start=start, end=end)


def build_merge_prompt(partial_summaries: list[str]) -> str:
    joined = "\n\n---\n\n".join(f"[Segment {i+1}]\n{s}" for i, s in enumerate(partial_summaries))
    return MERGE_TEMPLATE.format(partial_summaries=joined)


def build_longcontext_prompt(text: str) -> str:
    return LONGCONTEXT_TEMPLATE.format(text=text)
