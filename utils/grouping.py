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


def group_detections(detections, fps, max_gap=0.3, min_duration=0.25, text="..."):
    """Group labelled detections into timed subtitle blocks.

    Args:
        detections: Time-ordered list of (timestamp, label, is_new) for
            frames where a subtitle was detected (label "N" or "S" — never
            "no"). is_new=True marks a frame whose text differs from the
            previous positive frame, forcing a block boundary there.
        fps: The sampling rate the timestamps came from.
        max_gap: Longest silence (seconds) bridged inside one block. Covers
            detector flicker during fades/animations; the text-change
            splitter still separates different subtitles inside a bridged
            gap. Always at least one sample interval.
        min_duration: Blocks shorter than this (seconds) are dropped —
            isolated false positives, not real subtitles.
        text: Placeholder text written into every block (this tool times
            subtitles, it doesn't read them).

    Returns:
        List of SubtitleBlock in time order, one per subtitle.
    """
    if not detections or fps <= 0:
        return []

    interval = 1.0 / fps
    bridge = max(max_gap, interval) + 1e-6

    blocks = []
    start, prev = detections[0][0], detections[0][0]
    label = detections[0][1]

    def flush():
        end = prev + interval
        if end - start >= min_duration:
            blocks.append(SubtitleBlock(start, end, text, label=label))

    for t, lab, is_new in detections[1:]:
        if lab == label and t - prev <= bridge and not is_new:
            prev = t
        else:
            flush()
            start = prev = t
            label = lab

    flush()
    return blocks


def fix_seven_endings(blocks, shift=0.03, tol=1e-3):
    """Shift block ends whose centiseconds end in 7 back by 0.03 s.

    User-observed quirk of the source videos: whenever a block's end time
    falls on a centisecond whose last digit is 7 (0:00:00.07, 0:00:30.57
    in an ASS-style editor display, ...), it is 3 centiseconds late — a
    frame-time rounding artifact. This moves such ends 0.03 s earlier so
    they end in 4, and moves the start of any block that began at that
    exact boundary back with it, so back-to-back lines stay contiguous
    (S triplets share their timing, so all three lines shift together).
    A block too short to survive the shift is left untouched.

    Mutates `blocks` (time-ordered) in place; returns how many ends moved.
    """
    shifted = 0
    for i, block in enumerate(blocks):
        # Round to centiseconds the way a subtitle editor displays them.
        if int(block.end * 100 + 0.5) % 10 != 7:
            continue
        if block.end - shift <= block.start:
            continue  # would invert the block; leave it alone
        old_end = block.end
        block.end -= shift
        shifted += 1
        for later in blocks[i + 1:]:
            if abs(later.start - old_end) <= tol:
                later.start -= shift
    return shifted


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
