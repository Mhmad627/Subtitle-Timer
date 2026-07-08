"""Frame-level subtitle presence detection.

The pipeline asks one question per sampled frame: "is a subtitle visible in
the crop?" Two implementations answer it:

* HeuristicDetector — no ML. An edge-density stand-in so the app works end to
  end before the model exists, and a baseline for the model to beat.
* OnnxDetector — the user-trained classifier, run through OpenCV's dnn module
  so the app needs no PyTorch at runtime.

Both expose `score(image) -> float in [0, 1]` and `detect(image) -> bool`.
"""

import cv2
import numpy as np

# Input geometry / preprocessing contract for the ONNX model. training/train.py
# imports these so training and inference can never drift apart:
# resize crop to (W, H), BGR -> RGB, scale pixels to [0, 1], NCHW batch of 1.
MODEL_INPUT_W = 256
MODEL_INPUT_H = 64


class HeuristicDetector:
    """Edge-density placeholder until the trained model exists.

    Subtitle text produces a burst of sharp edges inside the crop box; an
    empty background usually doesn't. Crude — busy scenes fool it — but it
    keeps the pipeline testable and gives the ML model a baseline.
    """

    # Edge densities at or above this map to score 1.0. Text in a tight crop
    # typically lands in the 0.02-0.15 range; empty frames well below 0.01.
    _FULL_SCALE_DENSITY = 0.05

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def score(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        density = float(np.count_nonzero(edges)) / edges.size
        return min(1.0, density / self._FULL_SCALE_DENSITY)

    def detect(self, image):
        return self.score(image) >= self.threshold


class OnnxDetector:
    """Runs a trained ONNX presence classifier (see training/train.py).

    The model must take a (1, 3, MODEL_INPUT_H, MODEL_INPUT_W) RGB tensor in
    [0, 1] and output a single sigmoid probability.
    """

    def __init__(self, model_path, threshold=0.5):
        self.threshold = threshold
        self._net = cv2.dnn.readNetFromONNX(model_path)

    def score(self, image):
        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1.0 / 255.0,
            size=(MODEL_INPUT_W, MODEL_INPUT_H),
            swapRB=True,  # BGR -> RGB
        )
        self._net.setInput(blob)
        return float(self._net.forward().reshape(-1)[0])

    def detect(self, image):
        return self.score(image) >= self.threshold


def load_detector(model_path=None, threshold=0.5):
    """Return the trained model if a path is given, else the heuristic."""
    if model_path:
        return OnnxDetector(model_path, threshold=threshold)
    return HeuristicDetector(threshold=threshold)
