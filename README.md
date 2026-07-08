# Subtitle Timer

Detect **when** burned-in subtitles are on screen and write a timed `.srt`
with placeholder text. Built for one consistent subtitle style: you position
a crop box over the subtitle area, a detector answers "subtitle visible?"
per sampled frame, and runs of positives become timed blocks.

Two detectors, one interface:

- **Built-in heuristic** — edge-density check, works out of the box on
  high-contrast subs. A stand-in and baseline.
- **Your trained model** — a small CNN you train on your own videos
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
| `--text`          | `...`            | Placeholder text written into each block.                  |

## How it works

```
Video file
  → Sample frames at N fps (OpenCV)          detector/frame_sampler.py
  → Crop the user-positioned subtitle box    detector/subtitle_region.py
  → Detector: subtitle visible? yes/no       detector/presence.py
  → Group positive runs into timed blocks    utils/grouping.py
  → Write .srt (placeholder text)            output/srt_writer.py
```

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

- [x] Timing pipeline with pluggable presence detector (heuristic baseline)
- [x] GUI: movable/resizable crop box, scrubber, cancel, live log
- [x] Training kit: dataset dumper + tiny CNN + ONNX export
- [ ] Train the real model on target videos
- [ ] Boundary refinement (re-check frames around block edges at full fps)
