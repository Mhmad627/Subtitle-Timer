"""Subtitle Timer — CLI entry point.

Detect WHEN burned-in subtitles are on screen and write a timed .srt with
placeholder text. There is no text recognition. Frames are classified into
three classes: "no" (nothing), "N" (normal subtitle -> one timed block), and
"S" (special style -> three ASS-tagged lines with identical timing). The
classifier is either a simple built-in heuristic (N only) or a user-trained
ML model (see TRAINING.md).

Pipeline:
    sample frames -> crop the subtitle box -> classify no/N/S per frame
    -> group same-label runs into blocks -> expand S blocks -> write .srt

Usage:
    python main.py --input video.mp4
    python main.py --input video.mp4 --crop 0.1 0.8 0.8 0.15 --fps 5
    python main.py --input video.mp4 --model subtitle_detector.onnx
"""

import argparse
import os
import sys

from detector.frame_sampler import sample_frames
from detector.subtitle_region import DEFAULT_CROP, crop_rect
from detector.presence import load_detector
from detector.text_change import TextChangeSplitter
from utils.grouping import expand_s_blocks, group_detections
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
        default=5.0,
        help="Frame sampling rate; higher = tighter timing, slower (default: 5).",
    )
    parser.add_argument(
        "--crop",
        type=float,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        default=list(DEFAULT_CROP),
        help="Subtitle box as fractions of the frame (default: 0 0.75 1 0.25, "
             "the full-width bottom quarter).",
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
        "--gap-tolerance",
        type=int,
        default=1,
        help="Missed samples allowed inside one block before it splits "
             "(default: 1).",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=2,
        help="Minimum detections for a block to be kept; filters one-frame "
             "false positives (default: 2).",
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

    N runs become one placeholder block each; S runs are expanded into three
    ASS-tagged lines with identical timing (utils/grouping.S_LINE_TAGS).

    `should_cancel` is an optional zero-arg callable; when it returns True
    the run aborts with PipelineCancelled (checked once per sampled frame).
    """
    detector = load_detector(args.model, threshold=args.threshold)
    engine = "trained model" if args.model else "built-in heuristic (all N)"
    print(f"Detector: {engine} (threshold={args.threshold})")
    print(f"Sampling {args.input} at {args.fps} fps, crop={tuple(args.crop)}...")

    splitter = (
        TextChangeSplitter(iou_threshold=args.split_iou)
        if args.split_iou > 0 else None
    )

    detections = []
    current = "no"
    for timestamp, frame in sample_frames(args.input, fps=args.fps):
        _check_cancel(should_cancel)
        cropped = crop_rect(frame, args.crop)
        label, _confidence = detector.classify(cropped)

        is_new = False
        if label == "no":
            if splitter is not None:
                splitter.reset()
        else:
            if splitter is not None:
                if label == "S":
                    # Text-change splitting applies to N only: the S banner
                    # is bright with dark text, which the bright-pixel mask
                    # can't track, so S runs are never split.
                    splitter.reset()
                else:
                    is_new = splitter.update(cropped)
            detections.append((timestamp, label, is_new))

        if label != current:
            current = label
            state = "gone" if label == "no" else f"{label} appeared"
            print(f"  [{timestamp:7.2f}s] subtitle {state}")
        elif is_new:
            print(f"  [{timestamp:7.2f}s] subtitle changed ({label})")

    print(f"{len(detections)} positive frames. Grouping into blocks...")
    blocks = group_detections(
        detections,
        fps=args.fps,
        gap_tolerance=args.gap_tolerance,
        min_count=args.min_count,
        text=args.text,
    )
    n_s = sum(1 for b in blocks if b.label == "S")
    if n_s:
        print(f"Expanding {n_s} S block(s) into 3 tagged lines each...")
    return expand_s_blocks(blocks)


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
