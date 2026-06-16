"""Subtitle Timer & Extractor — CLI entry point.

Extract subtitles (text + timing) from a video and write a standard .srt.

Two methods:
  * Visual OCR  — reads burned-in/hardcoded subtitles from video frames.
  * Audio       — transcribes the audio track with OpenAI Whisper.

In `auto` mode (default), OCR runs first; if it finds fewer than
--min-subtitles blocks, the program falls back to Whisper automatically.

Usage:
    python main.py --input video.mp4
    python main.py --input video.mp4 --mode auto --whisper-model base
    python main.py --input video.mp4 --mode audio --whisper-model small
"""

import argparse
import os
import sys

from detector.frame_sampler import sample_frames
from detector.subtitle_region import crop_region
from detector.ocr_reader import OCRReader
from utils.deduplicator import deduplicate
from output.srt_writer import write_srt
from audio import whisper_transcriber


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="subtitle-extractor",
        description="Extract subtitles from a video into an .srt file.",
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
        default=2.0,
        help="Frame sampling rate in frames per second for OCR (default: 2).",
    )
    parser.add_argument(
        "--region",
        type=float,
        default=0.25,
        help="Bottom crop fraction where subtitles live, for OCR (default: 0.25).",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "ocr", "audio"],
        default="auto",
        help="auto: OCR then Whisper fallback; ocr: OCR only; audio: Whisper only.",
    )
    parser.add_argument(
        "--whisper-model",
        choices=["tiny", "base", "small", "medium"],
        default="base",
        help="Whisper model size (default: base).",
    )
    parser.add_argument(
        "--min-subtitles",
        type=int,
        default=5,
        help="Min OCR blocks before falling back to Whisper in auto mode (default: 5).",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["en"],
        help="Language codes for OCR (default: en).",
    )
    parser.add_argument(
        "--similarity",
        type=float,
        default=85.0,
        help="Fuzzy match threshold (0-100) for merging OCR frames (default: 85).",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for OCR if available.",
    )
    return parser.parse_args(argv)


def default_output_path(input_path):
    base, _ = os.path.splitext(input_path)
    return base + ".srt"


def run_ocr(args):
    """Run the visual OCR pipeline and return a list of SubtitleBlock."""
    print(f"Loading OCR engine (languages={args.languages}, gpu={args.gpu})...")
    reader = OCRReader(languages=args.languages, gpu=args.gpu)

    print(f"Sampling frames from {args.input} at {args.fps} fps...")
    observations = []
    for timestamp, frame in sample_frames(args.input, fps=args.fps):
        cropped = crop_region(frame, region=args.region)
        text = reader.read(cropped)
        if text:
            observations.append((timestamp, text))
            print(f"  [{timestamp:7.2f}s] {text}")

    print(f"Collected {len(observations)} text observations. Deduplicating...")
    return deduplicate(observations, similarity_threshold=args.similarity, fps=args.fps)


def run_audio(args):
    """Run the Whisper audio transcription and return a list of SubtitleBlock."""
    print(f"Transcribing audio with Whisper (model={args.whisper_model})...")
    blocks = whisper_transcriber.transcribe(args.input, model_name=args.whisper_model)
    print(f"Whisper produced {len(blocks)} subtitle blocks.")
    return blocks


def select_blocks(args):
    """Pick a method based on --mode and return the subtitle blocks to write."""
    if args.mode == "ocr":
        return run_ocr(args)

    if args.mode == "audio":
        return run_audio(args)

    # auto: OCR first, fall back to Whisper if it found too little.
    blocks = run_ocr(args)
    if len(blocks) >= args.min_subtitles:
        print(f"OCR found {len(blocks)} blocks (>= {args.min_subtitles}). Using OCR.")
        return blocks

    print(
        f"OCR found only {len(blocks)} block(s) (< {args.min_subtitles}). "
        "Falling back to Whisper audio transcription."
    )
    return run_audio(args)


def main(argv=None):
    args = parse_args(argv)

    if not os.path.isfile(args.input):
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        return 1

    output_path = args.output or default_output_path(args.input)

    blocks = select_blocks(args)

    print(f"Writing {len(blocks)} subtitle blocks to {output_path}...")
    write_srt(blocks, output_path)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
