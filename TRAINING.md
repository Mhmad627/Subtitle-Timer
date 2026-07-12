# Training your subtitle classifier

The app asks one question per sampled frame: **"what is in the crop box?"**
with three answers:

| Class | Meaning | What the app writes |
| ----- | ------- | ------------------- |
| `no`  | no subtitle | nothing |
| `N`   | normal subtitle | one timed block with placeholder text |
| `S`   | special stacked style | three lines, identical timing, ASS tags `{\fad(0,300)\pos(538,932)}` / `(538,985)` / `(538,1054)` |

You are training a small 3-class image classifier to answer it for *your*
specific subtitle styles. Because the styles are consistent, this is an easy
ML task — a small CNN trained from scratch on your CPU handles it. The one
requirement: **N and S must look different on screen** (color, font, or
position); the model can only learn what's visible.

## 1. Collect crops

Dump cropped frames from a few of your real videos (use roughly the same box
position you use in the app):

```bash
python training/dump_frames.py --input video1.mp4 video2.mp4 video3.mp4
# custom box / rate:
python training/dump_frames.py --input video.mp4 --crop 0.1 0.8 0.8 0.15 --fps 1
```

Images land in `dataset/unsorted/`.

## 2. Sort them (the actual "labeling")

Create `dataset/no/`, `dataset/N/` and `dataset/S/`, then move each image
into the right folder. Windows Explorer with large thumbnails makes this
quick. (The whole `dataset/` folder is gitignored — it never gets committed.)

**How many?**

| Folder | Aim for | Notes |
| ------ | ------- | ----- |
| `no`   | 500–1000 | Needs the most variety: ordinary scenes, busy scenes, credits, bright objects where subs usually sit — the frames that would fool a naive detector. |
| `N`    | 300–1000 | Across many different scenes/videos. |
| `S`    | 300+     | From as many **different songs/videos** as you can — 300 frames of one opening teaches the model that song, not the style. |

Rules of thumb:

- **Variety beats volume.** 400 images from 8 videos outperform 1500 from one.
- Start at the low end, train, test on a real video, then add images from
  whatever it gets wrong (step 5). Two or three loops beats collecting
  thousands up front.
- Rough balance is enough; exact 1:1:1 doesn't matter.

## 3. Train

```bash
pip install -r training/requirements.txt
python training/train.py
```

Prints loss + validation accuracy per epoch (a few minutes on CPU) and writes
`subtitle_detector.onnx`. For consistent styles expect **99%+ validation
accuracy** — if you get less, look for mislabeled images or too little
variety. The script ends with a sanity check that OpenCV's `cv2.dnn` (what
the app uses) gives the same probabilities as PyTorch.

## 4. Use it

- **GUI**: put `subtitle_detector.onnx` in the model field (Browse...).
- **CLI**: `python main.py --input video.mp4 --model subtitle_detector.onnx`

Until a model is selected, the built-in heuristic runs — it can't tell S
from N, so it labels everything N.

## 5. Iterate

If the model misfires on some video, dump frames from *that* video, sort the
misclassified ones into the right folders, and retrain. Two or three of these
loops usually gets you a very reliable classifier.

## How the pieces fit

```
training/train.py  →  subtitle_detector.onnx  →  cv2.dnn (in the app)
        └── imports CLASS_NAMES + MODEL_INPUT_W/H from detector/presence.py
            so training always matches inference (class order & preprocessing)
```

The model input is the crop resized to 256×64 RGB in [0,1]; output is a
softmax over (no, N, S). The app takes the argmax, and its threshold option
turns hesitant N/S calls into "no".

The three S tag lines live in `S_LINE_TAGS` in [utils/grouping.py](utils/grouping.py)
if you ever need to adjust the positions.

## Back-to-back subtitles

The model only answers no/N/S per frame; when one sentence instantly
replaces another, the boundary between them is found separately by
[detector/text_change.py](detector/text_change.py): it compares bright-pixel
masks between consecutive frames and splits the block when the glyph
pattern changes (`--split-iou`, default 0.5; 0 disables). This assumes the
subtitle text is bright (white/yellow); if your style is dark or heavily
colored, tell the splitter via its `brightness` parameter — or splitting
silently stays off and blocks just merge like before.

## Notes for the CV write-up

Things this project demonstrates that are worth naming: multi-class dataset
design with hard negatives, augmentation for robustness to box placement,
train/val methodology, ONNX export, and framework-free deployment (a
PyTorch-trained model shipped in a ~170 MB app via OpenCV's dnn instead of
bundling torch).
