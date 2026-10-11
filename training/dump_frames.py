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
# Only then: the GUI imports this module as part of the training package and
# must not have its import path changed.
if __name__ == "__main__":
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


def dump_video(video, regions, fps, out, should_cancel=None):
    """Crop every sampled frame of `video` and save each crop as a PNG.

    Args:
        video: Path to the video file.
        regions: List of (region_id, rect) pairs. region_id goes into the
            filename ({stem}_{region_id}_{ms}ms.png) so crops from several
            boxes don't collide; None keeps the plain {stem}_{ms}ms.png
            name (the single-box CLI case).
        fps: Sampling rate; 0 = every native frame.
        out: Output folder (must exist).
        should_cancel: Optional zero-arg callable; when it returns True the
            dump stops early (checked once per sampled frame). Used by the
            GUI's Cancel button.

    Returns:
        Number of crops written.
    """
    stem = os.path.splitext(os.path.basename(video))[0]
    count = 0
    # True while the cursor sits at the end of an unterminated "\r" progress
    # line; anything else printed must start a new line first, or the next
    # progress update would overwrite it.
    on_progress_line = False
    for frames, (timestamp, frame) in enumerate(sample_frames(video, fps=fps), 1):
        if should_cancel is not None and should_cancel():
            break
        for region_id, rect in regions:
            crop = crop_rect(frame, rect)
            tag = f"_{region_id}" if region_id else ""
            name = f"{stem}{tag}_{int(timestamp * 1000):08d}ms.png"
            # cv2.imwrite silently fails on non-ASCII filenames on Windows
            # (Japanese video names!); encode in memory and write with
            # Python's open(), which handles Unicode paths properly.
            ok, buffer = cv2.imencode(".png", crop)
            if not ok:
                if on_progress_line:
                    print()
                    on_progress_line = False
                print(f"  failed to encode frame at {timestamp:.1f}s")
                continue
            with open(os.path.join(out, name), "wb") as f:
                f.write(buffer.tobytes())
            count += 1
        if frames % 100 == 0:
            print(f"\r  {timestamp:7.1f}s: {count} crops", end="")
            on_progress_line = True
    if on_progress_line:
        print()
    print(f"{video}: {count} crops")
    return count


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    total = 0
    for video in args.input:
        if not os.path.isfile(video):
            print(f"Skipping (not found): {video}")
            continue
        total += dump_video(video, [(None, args.crop)], args.fps, args.out)

    print(f"\nDumped {total} images to {args.out}.")
    print("Next: sort them into dataset/no/, dataset/N/ and dataset/S/, "
          "then run training/train.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
