"""
Load football match transcripts, segments, and summaries from the dataset directory.
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict


DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "football_commentary_dataset")


@dataclass
class Segment:
    start: float
    end: float
    text: str
    pitch: Optional[float] = None
    energy: Optional[float] = None


@dataclass
class Match:
    match_id: str
    transcript: str
    segments: List[Segment]
    summary: Optional[str] = None


def _parse_segments_file(path: str) -> List[Segment]:
    """Parse the _segments.txt format: [start - end] P:pitch E:energy | text"""
    segments = []
    pattern = re.compile(
        r"\[(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\]\s*P:([\d.]+)\s*E:([\d.]+)\s*\|\s*(.*)"
    )
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = pattern.match(line)
            if m:
                segments.append(Segment(
                    start=float(m.group(1)),
                    end=float(m.group(2)),
                    pitch=float(m.group(3)),
                    energy=float(m.group(4)),
                    text=m.group(5).strip(),
                ))
    return segments


def list_match_ids(dataset_dir: str = DATASET_DIR) -> List[str]:
    """Return sorted list of match IDs based on available summary files."""
    summaries_dir = os.path.join(dataset_dir, "data", "summaries")
    ids = []
    for fname in os.listdir(summaries_dir):
        if fname.endswith(".txt"):
            ids.append(fname[:-4])
    return sorted(ids)


def load_match(match_id: str, dataset_dir: str = DATASET_DIR) -> Match:
    """Load a single match by ID."""
    transcripts_dir = os.path.join(dataset_dir, "data", "transcripts")
    summaries_dir = os.path.join(dataset_dir, "data", "summaries")

    # Transcript
    transcript_path = os.path.join(transcripts_dir, f"{match_id}_transcript.txt")
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = f.read().strip()

    # Segments
    segments_path = os.path.join(transcripts_dir, f"{match_id}_segments.txt")
    segments = _parse_segments_file(segments_path)

    # Summary (optional — test set matches still need to be loaded for inference)
    summary = None
    summary_path = os.path.join(summaries_dir, f"{match_id}.txt")
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = f.read().strip()

    return Match(match_id=match_id, transcript=transcript, segments=segments, summary=summary)


def load_all_matches(dataset_dir: str = DATASET_DIR) -> Dict[str, Match]:
    """Load all matches. Returns dict keyed by match_id."""
    ids = list_match_ids(dataset_dir)
    matches = {}
    for mid in ids:
        try:
            matches[mid] = load_match(mid, dataset_dir)
        except FileNotFoundError as e:
            print(f"[WARN] Skipping {mid}: {e}")
    return matches
