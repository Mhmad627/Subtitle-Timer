# Training your subtitle-presence model

The app asks one question per sampled frame: **"is a subtitle visible in the
crop box?"** You are training a tiny binary image classifier to answer it for
*your* specific subtitle style. Because the style is consistent (same font,
color, position), this is one of the easiest possible ML tasks — a small CNN
trained from scratch on your CPU handles it; no GPU or pretrained models
needed.

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

Create `dataset/yes/` and `dataset/no/`, then move each image:

- `yes/` — a subtitle is visible (even partially)
- `no/`  — no subtitle

Windows Explorer with large thumbnails makes this a few minutes of work.
Guidelines:

- Aim for **300–1000 images per class**. More helps, balance matters less
  than variety.
- Use **several different videos** so the model learns the subtitle, not one
  video's look.
- Include **hard negatives** in `no/`: busy scenes, bright objects in the
  subtitle area, credits, on-screen signs — the things that would fool a
  naive detector.

## 3. Train

```bash
pip install -r training/requirements.txt
python training/train.py
```

Prints loss + validation accuracy per epoch (a few minutes on CPU) and writes
`subtitle_detector.onnx`. For a consistent subtitle style expect **99%+
validation accuracy** — if you get less, you likely have mislabeled images or
too little variety.

The script ends with a sanity check that OpenCV's `cv2.dnn` (what the app
uses) gives the same answer as PyTorch on the exported model.

## 4. Use it

- **GUI**: put `subtitle_detector.onnx` in the model field (Browse...).
- **CLI**: `python main.py --input video.mp4 --model subtitle_detector.onnx`

## 5. Iterate

If the model misfires on some video, dump frames from *that* video, sort the
misclassified ones into the right folders, and retrain. Two or three of these
loops usually gets you a very reliable detector.

## How the pieces fit

```
training/train.py  →  subtitle_detector.onnx  →  cv2.dnn (in the app)
        └── imports MODEL_INPUT_W/H from detector/presence.py so training
            preprocessing always matches inference preprocessing
```

The model input is the crop resized to 256×64 RGB in [0,1]; output is one
sigmoid probability. The app's threshold option maps straight onto that
probability.

## Notes for the CV write-up

Things this project demonstrates that are worth naming: dataset design with
hard negatives, augmentation for robustness to box placement, train/val
methodology, ONNX export, and framework-free deployment (PyTorch-trained
model shipped in a 100 MB app via OpenCV's dnn instead of bundling torch).
