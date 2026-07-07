"""
Text segmentation for the auto-SR pipeline.

The original story and each participant's recall transcript are split
into ``num_segments`` overlapping word-based segments. Overlap is used to
reduce artificial breaks at segment boundaries (Herrmann, 2025).
"""

from __future__ import annotations

from typing import List


def segment_text(text: str, num_segments: int = 10, overlap_ratio: float = 0.2) -> List[str]:
    """Split ``text`` into ``num_segments`` overlapping word-based segments.

    Parameters
    ----------
    text : str
        Input text (e.g., the original story, or a participant's recall
        transcript). Words are assumed to be whitespace-separated.
    num_segments : int, default 10
        Number of segments to produce. The published study used 10
        segments for both the story and the recall transcripts.
    overlap_ratio : float, default 0.2
        Proportion of overlap between adjacent segments (0-1). The
        published study used 20% overlap.

    Returns
    -------
    list of str
        ``num_segments`` text segments. If the input has fewer words than
        ``num_segments``, the full text is returned as the first segment
        and the remaining segments are empty strings.
    """
    words = text.split()
    total_words = len(words)

    if total_words < num_segments:
        return [text] + [""] * (num_segments - 1)

    step = total_words / (1 + (1 - overlap_ratio) * (num_segments - 1))
    segment_len = int(step / (1 - overlap_ratio))

    starts = [
        round(i * (total_words - segment_len) / (num_segments - 1))
        for i in range(num_segments)
    ]

    return [" ".join(words[s:s + segment_len]) for s in starts]
