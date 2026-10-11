"""Subtitle Timer — CLI entry point.

Detect WHEN burned-in subtitles are on screen and write a timed .ass (or
.srt) with placeholder text. There is no text recognition. Frames are classified into
three classes: "no" (nothing), "N" (normal subtitle -> one timed block), and
"S" (special style -> several ASS-tagged lines, three by default, with
identical timing). The
classifier is either a simple built-in heuristic (N only) or a user-trained
ML model (see TRAINING.md).

Each style has its own crop box: the N box(es) only produce N blocks and the
S box only S blocks, so one run covers both styles. A second N box
(--crop-n2) is optional, for videos that show normal subtitles in two
different places.

Pipeline:
    sample frames -> crop each region's box -> classify no/N/S per crop
    -> group same-class runs into blocks -> expand S blocks -> write .ass

Usage:
    python main.py --input video.mp4
    python main.py --input video.mp4 --crop-n 0.1 0.8 0.8 0.15 --fps 5
    python main.py --input video.mp4 --crop-n 0 0.75 1 0.25 --crop-s 0 0.55 1 0.2
    python main.py --input video.mp4 --crop-n 0 0.75 1 0.25 --crop-n2 0 0.02 1 0.15
    python main.py --input video.mp4 --model subtitle_detector.onnx
"""

import argparse
import os
import sys

from detector.frame_sampler import sample_frames, video_size
from detector.subtitle_region import DEFAULT_CROP, crop_rect
from detector.presence import load_detector
from detector.text_change import TextChangeSplitter
from utils.grouping import (
    S_LINE_TAGS,
    add_s_delay,
    expand_s_blocks,
    fix_seven_boundaries,
    fix_seven_ends,
    group_detections,
    line_problem,
    s_tags_problem,
)
from output.ass_writer import write_ass
from output.srt_writer import write_srt


class PipelineCancelled(Exception):
    """Raised to abort a run mid-pipeline (used by the GUI's Cancel button)."""


def _check_cancel(should_cancel):
    if should_cancel is not None and should_cancel():
        raise PipelineCancelled()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="subtitle-timer",
        description="Detect burned-in subtitle timing and write an .ass "
                    "(or .srt) file.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to the output file: .ass, or .srt for SubRip (default: "
             "an .ass with the same name as the video).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=0.0,
        help="Frame sampling rate. 0 = every frame, frame-accurate timing "
             "(default). A rate like 5 is faster but times only to ±1/rate s.",
    )
    parser.add_argument(
        "--crop-n",
        dest="crop_n",
        type=float,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        default=list(DEFAULT_CROP),
        help="N-subtitle box as fractions of the frame (default: 0 0.75 1 "
             "0.25, the full-width bottom quarter).",
    )
    parser.add_argument(
        "--crop-n2",
        dest="crop_n2",
        type=float,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        default=None,
        help="Second N-subtitle box, for videos with normal subtitles in "
             "two places (default: no second N box).",
    )
    parser.add_argument(
        "--crop-s",
        dest="crop_s",
        type=float,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        default=None,
        help="S-subtitle box as fractions of the frame (default: no S box — "
             "S detection off).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to a trained .onnx presence model (default: use the "
             "built-in edge-density heuristic).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Detection threshold 0-1; lower catches more, with more false "
             "positives (default: 0.5).",
    )
    parser.add_argument(
        "--max-gap",
        type=float,
        default=0.3,
        help="Longest silence in seconds bridged inside one block; covers "
             "detector flicker during fades (default: 0.3).",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.25,
        help="Drop blocks shorter than this many seconds — isolated false "
             "positives (default: 0.25).",
    )
    parser.add_argument(
        "--text",
        default="...",
        help="Placeholder text written into N blocks (default: '...'). "
             "S blocks get the --s-tags lines instead.",
    )
    parser.add_argument(
        "--s-tags",
        dest="s_tags",
        nargs="+",
        metavar="LINE",
        default=list(S_LINE_TAGS),
        help="The lines every S block expands into, top to bottom — as many "
             "as needed (default: the three ASS fade + position tags in "
             "utils/grouping.py).",
    )
    parser.add_argument(
        "--split-iou",
        type=float,
        default=0.5,
        help="Split a block when the bright-text mask overlap between "
             "consecutive frames drops below this IoU — separates "
             "back-to-back subtitles with no gap (0 disables; default: 0.5).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output file if it already exists (default: refuse — "
             "the default <video>.ass is also where Aegisub saves an edited "
             "script, and overwriting it would lose that work).",
    )
    args = parser.parse_args(argv)
    # Strip like the GUI does, so both front ends write the same lines.
    args.s_tags = [tag.strip() for tag in args.s_tags]
    problem = s_tags_problem(args.s_tags)
    if problem:
        parser.error(f"--s-tags: {problem}")
    problem = line_problem(args.text)
    if problem:
        parser.error(f"--text {problem}")
    return args


def default_output_path(input_path):
    base, _ = os.path.splitext(input_path)
    return base + ".ass"


def resolve_output_path(output, input_path):
    """The file to write: `output`, with .ass added unless it already ends
    in .ass or .srt (the only formats written — "Episode.01" or "subs.txt"
    would otherwise hide an ASS script under an extension Aegisub doesn't
    recognize), or default_output_path when it's empty."""
    if not output:
        return default_output_path(input_path)
    if os.path.splitext(output)[1].lower() not in (".ass", ".srt"):
        return output + ".ass"
    return output


def write_output(blocks, output_path, video_path):
    """Write `blocks` as SubRip if `output_path` ends in .srt, else as ASS.

    ASS is the default because Aegisub merges overlapping lines (the
    stacked S lines) when it handles an .srt. The ASS script resolution is
    the video's size (1920x1080 if it can't be read).
    """
    if output_path.lower().endswith(".srt"):
        write_srt(blocks, output_path)
    else:
        write_ass(blocks, output_path, resolution=video_size(video_path) or (1920, 1080))


def run_detection(args, should_cancel=None):
    """Run per-frame classification and return a list of SubtitleBlock.

    Each crop box is a (region_id, crop_rect, target_class) triple: region_id
    is just a label for logging/bookkeeping (e.g. "N2"), while target_class
    is the class name the box's detections become ("N" or "S"). This split
    matters because two boxes can target the same class (a second N box) —
    a detector that tells classes apart (the trained model) only accepts a
    box's target_class, never its region_id, so "N2" isn't a class the model
    needs to know about. The heuristic can't tell styles apart at all, so
    any positive counts as the box's target_class regardless.

    N runs become one placeholder block each; S runs are expanded into one
    line per S tag, all with identical timing (args.s_tags, defaulting to
    the three utils/grouping.S_LINE_TAGS).

    `should_cancel` is an optional zero-arg callable; when it returns True
    the run aborts with PipelineCancelled (checked once per sampled frame).
    """
    # Checked before the long frame loop, not after it: the N placeholder
    # text and the S lines go into the output file once it's done.
    problem = line_problem(args.text)
    if problem:
        raise ValueError(f"Block text {problem}")
    s_tags = getattr(args, "s_tags", None)
    if s_tags is None:  # an empty list is a caller bug; let the check reject it
        s_tags = S_LINE_TAGS
    problem = s_tags_problem(s_tags)
    if problem:
        raise ValueError(problem)
    detector = load_detector(args.model, threshold=args.threshold)
    engine = "trained model" if args.model else "built-in heuristic (all N)"
    rate = "every frame" if args.fps <= 0 else f"{args.fps} fps"
    print(f"Detector: {engine} (threshold={args.threshold})")

    regions = [("N", tuple(args.crop_n), "N")]
    if getattr(args, "crop_n2", None) is not None:
        regions.append(("N2", tuple(args.crop_n2), "N"))
    if getattr(args, "crop_s", None) is not None:
        regions.append(("S", tuple(args.crop_s), "S"))
    for region_id, crop, _target in regions:
        print(f"  {region_id} box: {crop}")
    print(f"Sampling {args.input} at {rate}...")

    # Text-change splitting applies to N boxes only: the S banner is bright
    # with dark text, which the glyph mask can't track. Each N-class box
    # gets its own splitter instance — they watch unrelated screen areas,
    # so one box's glyph mask must never be compared against another's.
    splitters = {
        region_id: TextChangeSplitter(iou_threshold=args.split_iou)
        for region_id, _crop, target in regions
        if target == "N" and args.split_iou > 0
    }

    detections = {region_id: [] for region_id, _crop, _target in regions}
    visible = {region_id: False for region_id, _crop, _target in regions}
    # The real sampling interval is measured from the first two timestamps
    # (fps=0 means native rate, which only the video itself knows).
    first_ts = second_ts = None
    for timestamp, frame in sample_frames(args.input, fps=args.fps):
        _check_cancel(should_cancel)
        if first_ts is None:
            first_ts = timestamp
        elif second_ts is None:
            second_ts = timestamp
        for region_id, crop, target in regions:
            cropped = crop_rect(frame, crop)
            label, _confidence = detector.classify(cropped)
            hit = label != "no" and (
                label == target or not detector.distinguishes_styles
            )

            is_new = False
            if hit:
                splitter = splitters.get(region_id)
                if splitter is not None:
                    # The splitter keeps its memory across "no" flickers so
                    # a text change can't hide inside a gap that grouping
                    # later bridges.
                    is_new = splitter.update(cropped)
                detections[region_id].append((timestamp, target, is_new))

            if hit != visible[region_id]:
                visible[region_id] = hit
                state = "appeared" if hit else "gone"
                print(f"  [{timestamp:7.2f}s] {region_id} subtitle {state}")
            elif is_new:
                print(f"  [{timestamp:7.2f}s] {region_id} subtitle changed")

    effective_fps = args.fps
    if second_ts is not None and second_ts > first_ts:
        effective_fps = 1.0 / (second_ts - first_ts)

    total = sum(len(hits) for hits in detections.values())
    print(f"{total} positive frames. Grouping into blocks...")
    # Group each box's stream separately (two boxes can be on screen at the
    # same time), then interleave the blocks by start time. The .x7 boundary
    # fix runs here too, per region: it needs each region's own genuinely
    # sequential, non-overlapping timeline to tell "the previous subtitle"
    # from "an unrelated, still-running block from a different box" — a
    # distinction that's lost once regions are merged.
    blocks = []
    n_shifted = 0
    for region_id, _crop, _target in regions:
        region_blocks = group_detections(
            detections[region_id],
            fps=effective_fps,
            max_gap=args.max_gap,
            min_duration=args.min_duration,
            text=args.text,
        )
        n_shifted += fix_seven_boundaries(region_blocks)
        blocks.extend(region_blocks)
    blocks.sort(key=lambda b: b.start)
    n_s = sum(1 for b in blocks if b.label == "S")
    if n_s:
        print(f"Expanding {n_s} S block(s) into {len(s_tags)} tagged line(s) each...")
    blocks = expand_s_blocks(blocks, tags=s_tags)
    n_delayed = add_s_delay(blocks)
    if n_delayed:
        print(f"Delayed {n_delayed} S line end(s) by 0.16 s.")
    # A delayed S end can land on .x7 too; fix_seven_ends only ever looks at
    # a block's own start, so it's safe to run on the merged, multi-region
    # list (unlike fix_seven_boundaries's start-shift, above).
    n_shifted += fix_seven_ends(blocks)
    if n_shifted:
        print(f"Moved {n_shifted} block boundary(ies) landing on a .x7 "
              "centisecond back by 0.03 s.")
    return blocks


def main(argv=None):
    args = parse_args(argv)

    if not os.path.isfile(args.input):
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        return 1
    if args.model and not os.path.isfile(args.model):
        print(f"Error: model file not found: {args.model}", file=sys.stderr)
        return 1

    output_path = resolve_output_path(args.output, args.input)
    if os.path.exists(output_path) and not args.overwrite:
        print(f"Error: {output_path} already exists (it may be a script you "
              "edited in Aegisub). Pass --overwrite to replace it, or choose "
              "another --output.", file=sys.stderr)
        return 1

    blocks = run_detection(args)

    print(f"Writing {len(blocks)} subtitle blocks to {output_path}...")
    write_output(blocks, output_path, args.input)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
