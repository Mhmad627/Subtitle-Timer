# Subtitle Timer

Detect **when** burned-in subtitles are on screen and write a timed `.srt`
with placeholder text. You position a crop box over the subtitle area and a
classifier labels every sampled frame:

- **N** — normal subtitle → one timed block
- **S** — special stacked style → three lines with identical timing and ASS
  tags (`{\fad(0,300)\pos(...)}`, positions in `utils/grouping.py`)
- **no** — nothing → ignored

Runs of same-label frames become blocks ("time it until the sub changes").

Two classifiers, one interface:

- **Built-in heuristic** — edge-density check, works out of the box on
  high-contrast subs (labels everything N). A stand-in and baseline.
- **Your trained model** — a small 3-class CNN you train on your own videos
  (see [TRAINING.md](TRAINING.md)), exported to ONNX and run through
  OpenCV's `dnn` module. No PyTorch needed at runtime.

## Install

```bash
pip install -r requirements.txt
```

## Usage

### GUI

```bash
python gui.py
```

Browse for a video, scrub to a moment where a subtitle is visible, drag the
green box onto the subtitle area (corners resize it), and click **Detect
Subtitles**. Pick your trained `.onnx` in the model field once you have one.

### Command line

```bash
python main.py --input video.mp4
python main.py --input video.mp4 --crop 0.1 0.8 0.8 0.15 --fps 5
python main.py --input video.mp4 --model subtitle_detector.onnx
```

### CLI options

| Flag              | Default          | Description                                                |
| ----------------- | ---------------- | ---------------------------------------------------------- |
| `--input`         | (required)       | Path to the input video file.                              |
| `--output`        | `<input>.srt`    | Path to the output `.srt` file.                            |
| `--fps`           | `5`              | Sampling rate; higher = tighter timing, slower.            |
| `--crop`          | `0 0.75 1 0.25`  | Subtitle box `X Y W H` as fractions of the frame.          |
| `--model`         | (heuristic)      | Trained `.onnx` presence model.                            |
| `--threshold`     | `0.5`            | Detection score cutoff (0–1); lower catches more.          |
| `--gap-tolerance` | `1`              | Missed samples allowed inside one block.                   |
| `--min-count`     | `2`              | Minimum detections to keep a block (filters one-offs).     |
| `--text`          | `...`            | Placeholder text for N blocks (S blocks get ASS tag lines). |
| `--split-iou`     | `0.5`            | Text-change split sensitivity for back-to-back subs (0 = off). |

## How it works

```
Video file
  → Sample frames at N fps (OpenCV)          detector/frame_sampler.py
  → Crop the user-positioned subtitle box    detector/subtitle_region.py
  → Classify each frame: no / N / S          detector/presence.py
  → Spot text changes (back-to-back subs)    detector/text_change.py
  → Group runs into timed blocks             utils/grouping.py
  → Expand S blocks into 3 tagged lines      utils/grouping.py
  → Write .srt                               output/srt_writer.py
```

Back-to-back N subtitles (a new sentence replacing the previous one with no
gap) are separated by comparing bright-pixel masks between consecutive
frames: the glyph pattern of an unchanged subtitle is pixel-stable even
while the video moves behind it, so a mask-overlap drop means new text.
S runs are never split this way (their bright banner hides text changes
from the mask); an S block ends on a class change or a gap.

## Project structure

```
subtitle-timer/
├── main.py                       # CLI entry point + pipeline
├── gui.py                        # Tkinter GUI (movable crop box, cancel, log)
├── detector/
│   ├── frame_sampler.py          # Extract frames at N fps
│   ├── subtitle_region.py        # Fraction-rect crop
│   └── presence.py               # Heuristic + ONNX detectors (one interface)
├── utils/
│   └── grouping.py               # Detections → timed SubtitleBlocks
├── output/
│   └── srt_writer.py             # Assemble + write the .srt
├── training/
│   ├── dump_frames.py            # Build the dataset from your videos
│   └── train.py                  # Train tiny CNN + export ONNX
└── TRAINING.md                   # Step-by-step model training guide
```

## Building a Windows .exe

```bash
pip install pyinstaller
pyinstaller subtitle_extractor.spec --noconfirm
```

Result: `dist/SubtitleExtractor/SubtitleExtractor.exe` (onedir — ship the
whole folder). Only OpenCV is bundled, so the app stays small; the trained
model is a separate small `.onnx` file you can swap without rebuilding.

## Roadmap

- [x] Timing pipeline with pluggable classifier (heuristic baseline)
- [x] GUI: movable/resizable crop box, scrubber, cancel, live log
- [x] Training kit: dataset dumper + tiny 3-class CNN (no/N/S) + ONNX export
- [x] S blocks → three ASS-tagged lines with shared timing
- [x] Zero-gap boundary splitting (bright-mask IoU between frames)
- [ ] Train the real model on target videos
