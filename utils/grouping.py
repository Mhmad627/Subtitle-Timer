"""Turn per-frame classifications into timed subtitle blocks.

The detector labels each sampled frame "no" / "N" / "S". Runs of consecutive
frames with the same label become one SubtitleBlock (bridging short detector
flickers, dropping one-off false positives). A label change is a block
boundary — that's how "time it until the sub changes" works.

S blocks are then expanded: each becomes three lines with identical timing,
prefixed with fixed ASS override tags (fade + stacked positions).

Back-to-back subtitles of the same class (no gap — e.g. a speaker's next
sentence) are separated by the `is_new` flag each detection carries, set by
detector/text_change.py when the on-screen text visibly changes.
"""

# The three lines every S block expands into, in top-to-bottom order.
# ASS override tags: 300 ms fade-out + fixed positions for a 1080p script.
S_LINE_TAGS = (
    r"{\fad(0,300)\pos(538,932)}",
    r"{\fad(0,300)\pos(538,985)}",
    r"{\fad(0,300)\pos(538,1054)}",
)


class SubtitleBlock:
    """A subtitle entry: start/end times (seconds), text, and class label."""

    def __init__(self, start, end, text, label="N"):
        self.start = start
        self.end = end
        self.text = text
        self.label = label

    def __repr__(self):
        return (
            f"SubtitleBlock({self.label} "
            f"{self.start:.2f}-{self.end:.2f}: {self.text!r})"
        )


def group_detections(detections, fps, gap_tolerance=1, min_count=2, text="..."):
    """Group labelled detections into timed subtitle blocks.

    Args:
        detections: Time-ordered list of (timestamp, label, is_new) for
            frames where a subtitle was detected (label "N" or "S" — never
            "no"). is_new=True marks a frame whose text differs from the
            previous positive frame, forcing a block boundary there.
        fps: The sampling rate the timestamps came from.
        gap_tolerance: How many consecutive missed samples are allowed inside
            one block. Bridges single-frame detector flickers without gluing
            two different subtitles together.
        min_count: Minimum number of detections for a block to be kept.
            Filters out isolated false positives.
        text: Placeholder text written into every block (this tool times
            subtitles, it doesn't read them).

    Returns:
        List of SubtitleBlock in time order, one per subtitle.
    """
    if not detections or fps <= 0:
        return []

    interval = 1.0 / fps
    # Two detections continue the same block if the gap between them is at
    # most (gap_tolerance + 1) sample intervals (with a little float slack).
    max_gap = (gap_tolerance + 1) * interval + 1e-6

    blocks = []
    start, prev = detections[0][0], detections[0][0]
    label = detections[0][1]
    count = 1

    def flush():
        if count >= min_count:
            blocks.append(SubtitleBlock(start, prev + interval, text, label=label))

    for t, lab, is_new in detections[1:]:
        if lab == label and t - prev <= max_gap and not is_new:
            prev = t
            count += 1
        else:
            flush()
            start = prev = t
            label = lab
            count = 1

    flush()
    return blocks


def expand_s_blocks(blocks, tags=S_LINE_TAGS):
    """Replace each S block with len(tags) lines sharing its timing.

    N blocks pass through untouched; each S block becomes one line per ASS
    tag, in order, all with the original start/end times.
    """
    out = []
    for block in blocks:
        if block.label == "S":
            for tag in tags:
                out.append(SubtitleBlock(block.start, block.end, tag, label="S"))
        else:
            out.append(block)
    return out
