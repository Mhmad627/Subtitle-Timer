"""Detect when the subtitle TEXT changes between consecutive frames.

Presence classification alone can't separate two subtitles shown
back-to-back with no gap (a speaker's next sentence replacing the previous
one instantly). This module sees that boundary: subtitle glyphs are bright,
pixel-stable shapes, so we threshold the crop to a bright-pixel mask and
compare consecutive masks. While one subtitle stays on screen the mask
barely moves (even if the video behind it does); when the text changes, the
glyph pattern — and therefore the mask — changes a lot.

Graceful degradation: if the subtitle style isn't bright enough to mask
(fewer than `min_pixels` bright pixels), comparison is skipped and blocks
simply aren't split — the pre-splitter behaviour.
"""

import cv2
import numpy as np


class TextChangeSplitter:
    """Stateful comparator fed one positive-frame crop at a time."""

    def __init__(self, iou_threshold=0.5, brightness=200, min_pixels=40):
        """
        Args:
            iou_threshold: Split when the overlap (intersection over union)
                between consecutive bright masks falls below this. Lower =
                split less eagerly.
            brightness: Grayscale value (0-255) a pixel must reach to count
                as subtitle text. 200 captures white/yellow subs; dark or
                saturated-color styles fall below it and disable splitting.
            min_pixels: Minimum bright pixels in BOTH masks to attempt a
                comparison — guards against judging from noise.
        """
        self.iou_threshold = iou_threshold
        self.brightness = brightness
        self.min_pixels = min_pixels
        self._prev = None

    def reset(self):
        """Forget the previous frame (call when no subtitle is visible)."""
        self._prev = None

    def update(self, crop):
        """Feed the next subtitle-positive crop. Returns True if its text
        differs from the previous positive frame's (i.e. a new subtitle)."""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        mask = gray >= self.brightness

        prev, self._prev = self._prev, mask
        if prev is None or prev.shape != mask.shape:
            return False

        a = int(np.count_nonzero(mask))
        b = int(np.count_nonzero(prev))
        if a < self.min_pixels or b < self.min_pixels:
            return False

        intersection = int(np.count_nonzero(mask & prev))
        union = a + b - intersection
        return (intersection / union) < self.iou_threshold
