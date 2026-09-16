"""Subtitle Timer — CLI entry point.

Detect WHEN burned-in subtitles are on screen and write a timed .srt with
placeholder text. There is no text recognition. Frames are classified into
three classes: "no" (nothing), "N" (normal subtitle -> one timed block), and
"S" (special style -> three ASS-tagged lines with identical timing). The
classifier is either a simple built-in heuristic (N only) or a user-trained
ML model (see TRAINING.md).

Each style has its own crop box: the N box only produces N blocks and the S
box only S blocks, so one run covers both styles.

Pipeline:
    sample frames -> crop each style's box -> classify no/N/S per crop
    -> group same-label runs into blocks -> expand S blocks -> write .srt

Usage:
    python main.py --input video.mp4
    python main.py --input video.mp4 --crop-n 0.1 0.8 0.8 0.15 --fps 5
    python main.py --input video.mp4 --crop-n 0 0.75 1 0.25 --crop-s 0 0.55 1 0.2
    python main.py --input video.mp4 --model subtitle_detector.onnx
"""

import argparse
import os
import sys

from detector.frame_sampler import sample_frames
from detector.subtitle_region import DEFAULT_CROP, crop_rect
from detector.presence import load_detector
from detector.text_change import TextChangeSplitter
from utils.grouping import (
    add_s_delay,
    expand_s_blocks,
    fix_seven_boundaries,
    group_detections,
)
from output.srt_writer import write_srt


class PipelineCancelled(Exception):
    """Raised to abort a run mid-pipeline (used by the GUI's Cancel button)."""


def _check_cancel(should_cancel):
    if should_cancel is not None and should_cancel():
        raise PipelineCancelled()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="subtitle-timer",
        description="Detect burned-in subtitle timing and write an .srt file.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to the output .srt file (default: same name as the video).",
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
             "S blocks get the fixed ASS tag lines instead.",
    )
    parser.add_argument(
        "--split-iou",
        type=float,
        default=0.5,
        help="Split a block when the bright-text mask overlap between "
             "consecutive frames drops below this IoU — separates "
             "back-to-back subtitles with no gap (0 disables; default: 0.5).",
    )
    return parser.parse_args(argv)


def default_output_path(input_path):
    base, _ = os.path.splitext(input_path)
    return base + ".srt"


def run_detection(args, should_cancel=None):
    """Run per-frame classification and return a list of SubtitleBlock.

    Each style has its own crop box (args.crop_n / args.crop_s; crop_s may
    be None to skip S). Per sampled frame, each box's crop is classified
    independently. A detector that can tell N from S (the trained model)
    only accepts its box's own class — so an N subtitle bleeding into an
    overlapping S box isn't counted twice; the heuristic can't tell styles
    apart, so any positive counts as the box's class.

    N runs become one placeholder block each; S runs are expanded into three
    ASS-tagged lines with identical timing (utils/grouping.S_LINE_TAGS).

    `should_cancel` is an optional zero-arg callable; when it returns True
    the run aborts with PipelineCancelled (checked once per sampled frame).
    """
    detector = load_detector(args.model, threshold=args.threshold)
    engine = "trained model" if args.model else "built-in heuristic (all N)"
    rate = "every frame" if args.fps <= 0 else f"{args.fps} fps"
    print(f"Detector: {engine} (threshold={args.threshold})")

    regions = [("N", tuple(args.crop_n))]
    if getattr(args, "crop_s", None) is not None:
        regions.append(("S", tuple(args.crop_s)))
    for region_label, crop in regions:
        print(f"  {region_label} box: {crop}")
    print(f"Sampling {args.input} at {rate}...")

    # Text-change splitting applies to the N box only: the S banner is
    # bright with dark text, which the glyph mask can't track.
    splitter = (
        TextChangeSplitter(iou_threshold=args.split_iou)
        if args.split_iou > 0 else None
    )

    detections = {region_label: [] for region_label, _ in regions}
    visible = {region_label: False for region_label, _ in regions}
    # The real sampling interval is measured from the first two timestamps
    # (fps=0 means native rate, which only the video itself knows).
    first_ts = second_ts = None
    for timestamp, frame in sample_frames(args.input, fps=args.fps):
        _check_cancel(should_cancel)
        if first_ts is None:
            first_ts = timestamp
        elif second_ts is None:
            second_ts = timestamp
        for region_label, crop in regions:
            cropped = crop_rect(frame, crop)
            label, _confidence = detector.classify(cropped)
            hit = label != "no" and (
                label == region_label or not detector.distinguishes_styles
            )

            is_new = False
            if hit:
                if region_label == "N" and splitter is not None:
                    # The splitter keeps its memory across "no" flickers so
                    # a text change can't hide inside a gap that grouping
                    # later bridges.
                    is_new = splitter.update(cropped)
                detections[region_label].append((timestamp, region_label, is_new))

            if hit != visible[region_label]:
                visible[region_label] = hit
                state = "appeared" if hit else "gone"
                print(f"  [{timestamp:7.2f}s] {region_label} subtitle {state}")
            elif is_new:
                print(f"  [{timestamp:7.2f}s] {region_label} subtitle changed")

    effective_fps = args.fps
    if second_ts is not None and second_ts > first_ts:
        effective_fps = 1.0 / (second_ts - first_ts)

    total = sum(len(hits) for hits in detections.values())
    print(f"{total} positive frames. Grouping into blocks...")
    # Group each box's stream separately (N and S can be on screen at the
    # same time), then interleave the blocks by start time.
    blocks = []
    for region_label, _ in regions:
        blocks.extend(group_detections(
            detections[region_label],
            fps=effective_fps,
            max_gap=args.max_gap,
            min_duration=args.min_duration,
            text=args.text,
        ))
    blocks.sort(key=lambda b: b.start)
    n_s = sum(1 for b in blocks if b.label == "S")
    if n_s:
        print(f"Expanding {n_s} S block(s) into 3 tagged lines each...")
    blocks = expand_s_blocks(blocks)
    n_delayed = add_s_delay(blocks)
    if n_delayed:
        print(f"Delayed {n_delayed} S line end(s) by 0.16 s.")
    n_shifted = fix_seven_boundaries(blocks)
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

    output_path = args.output or default_output_path(args.input)

    blocks = run_detection(args)

    print(f"Writing {len(blocks)} subtitle blocks to {output_path}...")
    write_srt(blocks, output_path)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
