# Subtitle Timer & Extractor

Extract subtitles (text + timing) from video files and write a standard `.srt`.

Two methods:

1. **Visual OCR** — detect burned-in / hardcoded subtitles directly from video
   frames. **(Phase 1 — built)**
2. **Audio fallback** — if OCR finds fewer than `--min-subtitles` blocks,
   transcribe the audio with OpenAI Whisper. **(Phase 2 — built)**

## Install

```bash
pip install -r requirements.txt
```

FFmpeg must also be installed on your system (needed for Phase 2 audio
extraction).

## Usage

### GUI

```bash
python gui.py
```

A window opens where you can browse for a video, choose the mode / Whisper
model / OCR options, and click **Extract Subtitles**. Progress shows in the log
pane; the work runs on a background thread so the window stays responsive.

### Command line

```bash
python main.py --input video.mp4
python main.py --input video.mp4 --output subs.srt --mode auto --whisper-model base
python main.py --input video.mp4 --mode audio --whisper-model small
```

### CLI options

| Flag              | Default          | Description                                            |
| ----------------- | ---------------- | ----------------------------------------------------- |
| `--input`         | (required)       | Path to the input video file.                         |
| `--output`        | `<input>.srt`    | Path to the output `.srt` file.                       |
| `--mode`          | `auto`           | `auto` (OCR then Whisper), `ocr` only, or `audio` only. |
| `--whisper-model` | `base`           | Whisper size: `tiny` / `base` / `small` / `medium`.   |
| `--min-subtitles` | `5`              | OCR block threshold before falling back to Whisper.   |
| `--fps`           | `2`              | Frame sampling rate for OCR (frames per second).      |
| `--region`        | `0.25`           | Bottom crop fraction where subtitles live (OCR).      |
| `--languages`     | `en`             | OCR language codes (space-separated).                 |
| `--similarity`    | `85`             | Fuzzy match threshold (0–100) for merging OCR frames. |
| `--gpu`           | off              | Use GPU for OCR if available.                          |

## How it works (Phase 1)

```
Video file
  → Extract frames at N fps (OpenCV)        detector/frame_sampler.py
  → Crop bottom 25% of each frame           detector/subtitle_region.py
  → Run EasyOCR on the cropped region       detector/ocr_reader.py
  → Collect (timestamp, text) pairs         main.py
  → Deduplicate via fuzzy matching          utils/deduplicator.py
  → Write .srt                              output/srt_writer.py
```

## Project structure

```
subtitle-timer/
├── main.py                       # CLI entry point + pipeline
├── detector/
│   ├── frame_sampler.py          # Extract frames at N fps
│   ├── subtitle_region.py        # Crop the subtitle region
│   └── ocr_reader.py             # Run EasyOCR
├── audio/
│   └── whisper_transcriber.py    # Phase 2: audio fallback (stub)
├── output/
│   └── srt_writer.py             # Assemble + write the .srt
└── utils/
    └── deduplicator.py           # Merge frames into timed blocks
```

## Audio fallback (Phase 2)

In `auto` mode the program runs OCR first. If it produces fewer than
`--min-subtitles` blocks, it extracts the audio with FFmpeg (16 kHz mono WAV
to a temp file) and transcribes it with Whisper, then deletes the temp file.
Whisper segments already carry start/end times and clean text, so they map
straight to `.srt` blocks with no deduplication.

```
Run OCR pipeline
  → got >= --min-subtitles blocks?
      ├── YES → write .srt from OCR
      └── NO  → run Whisper → write .srt from Whisper
```

FFmpeg must be installed and on your PATH for the audio path.

## Building a Windows .exe

A windowed `.exe` (no terminal) can be built with PyInstaller:

```bash
pip install pyinstaller
pyinstaller subtitle_extractor.spec --noconfirm
```

The result is `dist/SubtitleExtractor/SubtitleExtractor.exe` (a onedir bundle —
ship the whole `SubtitleExtractor` folder, not just the .exe). Because it
bundles PyTorch + EasyOCR + Whisper, the folder is large (several GB) and the
first build takes a while. FFmpeg still needs to be installed on the target
machine (or place `ffmpeg.exe` next to the app) for the audio path.

## Roadmap

- [x] **Phase 1** — visual OCR pipeline → `.srt`
- [x] **Phase 2** — Whisper audio fallback (auto / ocr / audio modes)
- [x] **Phase 3a** — Tkinter GUI + Windows .exe build
- [ ] **Phase 3b** — OCR confidence scoring, language detection
```
