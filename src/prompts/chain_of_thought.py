"""
Chain-of-Thought (CoT) prompting for football match summarization.

Two-pass approach:
  Pass 1: Extract key events (goals, cards, substitutions) from commentary.
  Pass 2: Write a coherent match summary using those events.
"""

from typing import List


PASS1_TEMPLATE = (
    "You are a football analyst. Read the commentary excerpt below and list all key events "
    "you can identify: goals (with approximate time and scorer if mentioned), yellow/red cards, "
    "substitutions, and any other notable moments.\n\n"
    "Commentary ({start:.0f}s - {end:.0f}s):\n{text}\n\n"
    "Key events:"
)

PASS2_TEMPLATE = (
    "You are a sports journalist. Using the list of key events below, "
    "write a concise summary paragraph for this segment of the football match "
    "({start:.0f}s - {end:.0f}s).\n\n"
    "Key events:\n{events}\n\n"
    "Summary paragraph:"
)

LONGCONTEXT_PASS1_TEMPLATE = (
    "You are a football analyst. Read the full match commentary below and list all key events: "
    "goals (with approximate time and scorer if mentioned), yellow/red cards, "
    "substitutions, and any other notable moments.\n\n"
    "Commentary:\n{text}\n\n"
    "Key events:"
)

LONGCONTEXT_PASS2_TEMPLATE = (
    "You are a sports journalist. Using the list of key events below, "
    "write a complete match report covering the full game.\n\n"
    "Key events:\n{events}\n\n"
    "Match report:"
)

MERGE_TEMPLATE = (
    "You are a sports journalist. "
    "Below are partial summaries from different segments of the same football match. "
    "Combine them into a single coherent match report. "
    "Remove any repetition. Preserve all goals, key events, and the final result.\n\n"
    "Partial summaries:\n{partial_summaries}\n\n"
    "Final match report:"
)


def build_pass1_prompt(text: str, start: float, end: float) -> str:
    return PASS1_TEMPLATE.format(text=text, start=start, end=end)


def build_pass2_prompt(events: str, start: float, end: float) -> str:
    return PASS2_TEMPLATE.format(events=events, start=start, end=end)


def build_longcontext_pass1_prompt(text: str) -> str:
    return LONGCONTEXT_PASS1_TEMPLATE.format(text=text)


def build_longcontext_pass2_prompt(events: str) -> str:
    return LONGCONTEXT_PASS2_TEMPLATE.format(events=events)


def build_merge_prompt(partial_summaries: List[str]) -> str:
    joined = "\n\n---\n\n".join(f"[Segment {i+1}]\n{s}" for i, s in enumerate(partial_summaries))
    return MERGE_TEMPLATE.format(partial_summaries=joined)
