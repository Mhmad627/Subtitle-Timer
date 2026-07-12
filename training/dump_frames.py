"""Dump cropped frames from videos to build the training dataset.

Samples frames the same way the app does, crops the subtitle box, and saves
each crop as a PNG into dataset/unsorted/. You then sort the images into
dataset/no/ (no subtitle), dataset/N/ (normal subtitle) and dataset/S/
(the special stacked-lines style) — see TRAINING.md.

Usage:
    python training/dump_frames.py --input video1.mp4 video2.mp4
    # custom box, or a sparse 1-per-second dump instead of every frame:
    python training/dump_frames.py --input video.mp4 --crop 0.1 0.8 0.8 0.15 --fps 1
"""

import argparse
import os
import sys

# Make the repo root importable when run as `python training/dump_frames.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from detector.frame_sampler import sample_frames
from detector.subtitle_region import DEFAULT_CROP, crop_rect


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, nargs="+",
                        help="One or more video files to sample.")
    parser.add_argument("--crop", type=float, nargs=4, metavar=("X", "Y", "W", "H"),
                        default=list(DEFAULT_CROP),
                        help="Subtitle box as frame fractions — use roughly the "
                             "same box you position in the app.")
    parser.add_argument("--fps", type=float, default=0.0,
                        help="Frames per second to dump; 0 = every native "
                             "frame (default). Use e.g. 1 for a sparse dump.")
    parser.add_argument("--out", default=os.path.join("dataset", "unsorted"),
                        help="Output folder (default: dataset/unsorted).")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    total = 0
    for video in args.input:
        if not os.path.isfile(video):
            print(f"Skipping (not found): {video}")
            continue
        stem = os.path.splitext(os.path.basename(video))[0]
        count = 0
        for timestamp, frame in sample_frames(video, fps=args.fps):
            crop = crop_rect(frame, args.crop)
            name = f"{stem}_{int(timestamp * 1000):08d}ms.png"
            # cv2.imwrite silently fails on non-ASCII filenames on Windows
            # (Japanese video names!); encode in memory and write with
            # Python's open(), which handles Unicode paths properly.
            ok, buffer = cv2.imencode(".png", crop)
            if not ok:
                print(f"  failed to encode frame at {timestamp:.1f}s")
                continue
            with open(os.path.join(args.out, name), "wb") as f:
                f.write(buffer.tobytes())
            count += 1
        print(f"{video}: {count} crops")
        total += count

    print(f"\nDumped {total} images to {args.out}.")
    print("Next: sort them into dataset/no/, dataset/N/ and dataset/S/, "
          "then run training/train.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
