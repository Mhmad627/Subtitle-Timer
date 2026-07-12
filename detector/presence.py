"""Frame-level subtitle classification.

The pipeline asks one question per sampled frame: "what is visible in the
crop box?" with three possible answers:

* "no" — no subtitle
* "N"  — a normal subtitle
* "S"  — the special subtitle style that is written out as three stacked
         ASS-tagged lines (see utils/grouping.py)

Two implementations:

* HeuristicDetector — no ML. Edge-density stand-in so the app works before
  the model exists. It cannot tell S from N, so everything it finds is "N".
* OnnxDetector — the user-trained 3-class classifier, run through OpenCV's
  dnn module so the app needs no PyTorch at runtime.

Both expose `classify(image) -> (label, confidence)`.
"""

import cv2
import numpy as np

# Class order is the contract between training and inference: the model's
# softmax output must be in this order, and dataset folders use these names.
CLASS_NAMES = ("no", "N", "S")

# Input geometry / preprocessing contract for the ONNX model. training/train.py
# imports these so training and inference can never drift apart:
# resize crop to (W, H), BGR -> RGB, scale pixels to [0, 1], NCHW batch of 1.
MODEL_INPUT_W = 256
MODEL_INPUT_H = 64


class HeuristicDetector:
    """Edge-density placeholder until the trained model exists.

    Subtitle text produces a burst of sharp edges inside the crop box; an
    empty background usually doesn't. Crude — busy scenes fool it, and it
    cannot distinguish the S style, so every detection is labelled "N".
    """

    # Edge densities at or above this map to score 1.0. Text in a tight crop
    # typically lands in the 0.02-0.15 range; empty frames well below 0.01.
    _FULL_SCALE_DENSITY = 0.05

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def classify(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        density = float(np.count_nonzero(edges)) / edges.size
        score = min(1.0, density / self._FULL_SCALE_DENSITY)
        if score >= self.threshold:
            return "N", score
        return "no", 1.0 - score


class OnnxDetector:
    """Runs the trained ONNX classifier (see training/train.py).

    The model must take a (1, 3, MODEL_INPUT_H, MODEL_INPUT_W) RGB tensor in
    [0, 1] and output softmax probabilities in CLASS_NAMES order.
    """

    def __init__(self, model_path, threshold=0.5):
        self.threshold = threshold
        self._net = cv2.dnn.readNetFromONNX(model_path)

    def classify(self, image):
        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1.0 / 255.0,
            size=(MODEL_INPUT_W, MODEL_INPUT_H),
            swapRB=True,  # BGR -> RGB
        )
        self._net.setInput(blob)
        probs = self._net.forward().reshape(-1)
        idx = int(np.argmax(probs))
        label = CLASS_NAMES[idx]
        confidence = float(probs[idx])
        # A hesitant subtitle call is treated as "no subtitle".
        if label != "no" and confidence < self.threshold:
            return "no", confidence
        return label, confidence


def load_detector(model_path=None, threshold=0.5):
    """Return the trained model if a path is given, else the heuristic."""
    if model_path:
        return OnnxDetector(model_path, threshold=threshold)
    return HeuristicDetector(threshold=threshold)
