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
| `--fps`           | `0`              | Sampling rate; 0 = every frame (frame-accurate timing).    |
| `--crop`          | `0 0.75 1 0.25`  | Subtitle box `X Y W H` as fractions of the frame.          |
| `--model`         | (heuristic)      | Trained `.onnx` presence model.                            |
| `--threshold`     | `0.5`            | Detection score cutoff (0–1); lower catches more.          |
| `--max-gap`       | `0.3`            | Longest silence (s) bridged inside one block.              |
| `--min-duration`  | `0.25`           | Drop blocks shorter than this (s).                         |
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
gap) are separated by comparing glyph masks between consecutive frames.
The mask is *bright pixels near saturated pixels* — the white glyph core
hugging its colored border — which excludes bright scene backgrounds that
would otherwise hide the change. An unchanged subtitle is pixel-stable even
while the video moves behind it, so a mask-overlap drop means new text.
S runs are never split this way (their bright banner hides text changes);
an S block ends on a class change or a gap.

## Design decisions

- **Time subtitles, don't read them.** OCR quality on stylized hardsubs was
  the original approach and proved unreliable; timing + placeholder text
  slots straight into a subtitle editor (Aegisub), where text entry is fast
  but manual timing is the tedious part.
- **Framework-free deployment.** The model is trained in PyTorch but
  exported to ONNX and executed with OpenCV's `dnn` module, so the shipped
  app carries no ML framework. The `.onnx` file is swappable without
  rebuilding the app.
- **Training and inference share one contract.** Class order and input
  preprocessing are constants in `detector/presence.py`, imported by the
  trainer; the export ends with a cv2.dnn-vs-PyTorch parity check.
- **Classical CV where ML isn't needed.** Zero-gap boundaries between
  consecutive N subtitles are found by comparing bright-pixel glyph masks
  across frames — simpler and more predictable than learning it.

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
