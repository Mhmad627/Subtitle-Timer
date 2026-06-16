"""Merge per-frame OCR observations into timed subtitle blocks.

OCR output varies slightly from frame to frame (a flicker, a misread letter),
so we group consecutive observations using fuzzy string matching rather than
exact equality. Each resulting block has a start time, end time, and the text.
"""

from rapidfuzz import fuzz


class SubtitleBlock:
    """A single subtitle entry with start/end times (seconds) and text."""

    def __init__(self, start, end, text):
        self.start = start
        self.end = end
        self.text = text

    def __repr__(self):
        return f"SubtitleBlock({self.start:.2f}-{self.end:.2f}: {self.text!r})"


def _similar(a, b, threshold):
    return fuzz.ratio(a, b) >= threshold


def deduplicate(observations, similarity_threshold=85.0, fps=2.0):
    """Group consecutive similar observations into timed subtitle blocks.

    Args:
        observations: List of (timestamp_seconds, text) tuples in time order.
        similarity_threshold: 0-100 fuzzy match cutoff. Consecutive frames
            whose text is at least this similar are merged into one block.
        fps: Sampling rate, used to estimate how long the final frame of a
            block stays on screen (one sample interval).

    Returns:
        List of SubtitleBlock, in time order.
    """
    if not observations:
        return []

    # Duration of a single sampling interval, added to the last frame of a
    # block so its end time reflects the time the subtitle was actually shown.
    frame_duration = 1.0 / fps if fps > 0 else 0.0

    blocks = []
    cur_start, cur_text = observations[0]
    cur_end = cur_start

    for timestamp, text in observations[1:]:
        if _similar(text, cur_text, similarity_threshold):
            # Same subtitle still on screen — extend the block.
            cur_end = timestamp
        else:
            blocks.append(SubtitleBlock(cur_start, cur_end + frame_duration, cur_text))
            cur_start, cur_text = timestamp, text
            cur_end = timestamp

    blocks.append(SubtitleBlock(cur_start, cur_end + frame_duration, cur_text))
    return blocks
