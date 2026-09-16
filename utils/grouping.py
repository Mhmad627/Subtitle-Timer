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


def _ends_in_seven(t):
    """True if `t`'s centisecond (hundredths of a second) digit is 7, the
    way a subtitle editor displays it (e.g. 0:00:30.57)."""
    return int(round(t * 100)) % 10 == 7


def fix_seven_boundaries(blocks, shift=0.03, tol=1e-3):
    """Shift any block boundary landing on a .x7 centisecond back by 0.03 s.

    User-observed quirk of the source videos: whenever a timestamp — a
    block's start OR its end — falls on a centisecond whose last digit is 7
    (0:00:00.07, 0:00:30.57 in an ASS-style editor display, ...), it is 3
    centiseconds late, a frame-time rounding artifact. Every boundary is
    checked independently and shifted 0.03 s earlier so it ends in 4.

    Two touching blocks (one's end equal to the next one's start) share the
    same value and therefore the same digit, so when that boundary
    qualifies both sides shift by the same amount and stay touching — no
    separate "propagate to the neighbor" step is needed.

    IMPORTANT: `blocks` must be ONE region's own time-ordered, non-
    overlapping output from group_detections — the start-shift guard below
    treats `blocks[i - 1].end` as "the previous subtitle's end" to avoid
    creating an overlap. That's only true within a single region's own
    timeline. Two independently-classified regions (e.g. N and N2, or N and
    S) can legitimately be visible at once, so a block from one region
    merged next to a longer-running block from another would wrongly be
    treated as "still inside the previous subtitle" and its start would
    never be corrected. Call this per region before merging; use
    fix_seven_ends (which only ever looks at a block's own start, so it has
    no such assumption) for a later, merged-list-safe end-only pass.

    Guards: a block is never shrunk past inversion (end vs. its own start),
    and a start is never pulled earlier than the immediately preceding
    block's already-resolved end, so blocks never end up overlapping or
    negative.

    Mutates `blocks` (time-ordered by start) in place; returns how many
    boundaries moved.
    """
    moved = 0
    for i, block in enumerate(blocks):
        floor = blocks[i - 1].end if i > 0 else 0.0
        if _ends_in_seven(block.start) and block.start - shift >= floor - tol:
            block.start -= shift
            moved += 1
        if _ends_in_seven(block.end) and block.end - shift > block.start:
            block.end -= shift
            moved += 1
    return moved


def fix_seven_ends(blocks, shift=0.03):
    """Shift any block's END landing on a .x7 centisecond back by 0.03 s.

    Unlike the start-shift in fix_seven_boundaries, this only ever compares
    a block against its own start, never a neighbor — so it's safe to run
    on a list merging multiple independently-timed regions that may
    overlap in time. Meant as a final pass after add_s_delay, to catch a
    freshly-delayed S end that now lands on .x7 (which an earlier,
    per-region fix_seven_boundaries call couldn't have seen, since the
    delay happens after regions are merged).

    Mutates `blocks` in place; returns how many ends moved.
    """
    moved = 0
    for block in blocks:
        if _ends_in_seven(block.end) and block.end - shift > block.start:
            block.end -= shift
            moved += 1
    return moved


def add_s_delay(blocks, delay=0.16):
    """Add `delay` seconds to the end of every S block.

    User-observed quirk: the S banner's on-screen end lags the model's
    detected end by ~16 centiseconds. Applied after expand_s_blocks, so it
    reaches all three tagged lines identically (they share timing).

    Mutates `blocks` in place; returns how many ends were delayed.
    """
    delayed = 0
    for block in blocks:
        if block.label == "S":
            block.end += delay
            delayed += 1
    return delayed


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
