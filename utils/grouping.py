"""Turn per-frame yes/no detections into timed subtitle blocks.

The detector answers "is a subtitle visible?" once per sampled frame. This
module groups runs of consecutive positive frames into SubtitleBlock entries,
bridging short detector flickers and dropping one-off false positives.
"""


class SubtitleBlock:
    """A single subtitle entry with start/end times (seconds) and text."""

    def __init__(self, start, end, text):
        self.start = start
        self.end = end
        self.text = text

    def __repr__(self):
        return f"SubtitleBlock({self.start:.2f}-{self.end:.2f}: {self.text!r})"


def group_detections(times, fps, gap_tolerance=1, min_count=2, text="..."):
    """Group detection timestamps into timed subtitle blocks.

    Args:
        times: Sorted timestamps (seconds) of frames where a subtitle was
            detected.
        fps: The sampling rate the timestamps came from, used to know how far
            apart consecutive samples are and how long the final frame of a
            block stays on screen.
        gap_tolerance: How many consecutive missed samples are allowed inside
            one block. Bridges single-frame detector flickers without gluing
            two different subtitles together.
        min_count: Minimum number of detections for a block to be kept.
            Filters out isolated false positives.
        text: Placeholder text written into every block (there is no text
            recognition — this tool times subtitles, it doesn't read them).

    Returns:
        List of SubtitleBlock in time order.
    """
    if not times or fps <= 0:
        return []

    interval = 1.0 / fps
    # Two detections belong to the same block if the gap between them is at
    # most (gap_tolerance + 1) sample intervals (with a little float slack).
    max_gap = (gap_tolerance + 1) * interval + 1e-6

    blocks = []
    start = prev = times[0]
    count = 1

    def flush():
        if count >= min_count:
            blocks.append(SubtitleBlock(start, prev + interval, text))

    for t in times[1:]:
        if t - prev <= max_gap:
            prev = t
            count += 1
        else:
            flush()
            start = prev = t
            count = 1

    flush()
    return blocks
