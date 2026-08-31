"""Detect when the subtitle TEXT changes between consecutive frames.

Presence classification alone can't separate two subtitles shown
back-to-back with no gap (a speaker's next sentence replacing the previous
one instantly). This module sees that boundary by masking the glyph pixels
each frame and comparing consecutive masks: while one subtitle stays on
screen its glyphs are pixel-stable even though the video moves behind them;
when the text changes, the mask changes a lot.

A plain brightness threshold is NOT enough for a glyph mask — bright scene
backgrounds (white walls, skin) dominate it and hide text changes. The N
style is a white core wrapped in a thick saturated border, so the mask is:
bright pixels NEAR saturated pixels. Background whites have no saturated
border around them and drop out. Measured on real crops, same-text pairs
score ~0.95 IoU and changed-text pairs ~0.15.

Graceful degradation: styles without a saturated border (or too small to
produce `min_pixels` mask pixels) never split — blocks just merge as before.
"""

import cv2
import numpy as np


class TextChangeSplitter:
    """Stateful comparator fed one positive-frame crop at a time."""

    # How far (in pixels) a bright pixel may sit from the saturated border
    # and still count as a glyph core. 7x7 = 3px reach.
    _DILATE_KERNEL = np.ones((7, 7), np.uint8)

    def __init__(self, iou_threshold=0.5, brightness=200, saturation=100,
                 min_pixels=300):
        """
        Args:
            iou_threshold: Split when the overlap (intersection over union)
                between consecutive glyph masks falls below this. Lower =
                split less eagerly.
            brightness: Grayscale value (0-255) a pixel must reach to count
                as a glyph core.
            saturation: HSV saturation a pixel must reach to count as the
                colored glyph border.
            min_pixels: Minimum mask pixels in BOTH frames to attempt a
                comparison — guards against judging from noise. A real
                glyph mask is thousands of pixels; a mask built from noise
                pixels crossing the saturation threshold is at most a few
                hundred, and comparing those random masks splits one
                subtitle into chains of duplicate blocks.
        """
        self.iou_threshold = iou_threshold
        self.brightness = brightness
        self.saturation = saturation
        self.min_pixels = min_pixels
        self._prev = None
        self._prev_changed = False

    def reset(self):
        """Forget the previous frame (call on a class change)."""
        self._prev = None
        self._prev_changed = False

    def _glyph_mask(self, crop):
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # Smooth the saturation plane before thresholding: on outline colors
        # whose saturation rides near the threshold (e.g. tan), compression
        # noise otherwise flips individual pixels across it every frame,
        # producing a small random mask each frame — which reads as constant
        # text change and chops one subtitle into duplicate blocks.
        sat = cv2.GaussianBlur(hsv[:, :, 1], (5, 5), 0)
        border = ((sat >= self.saturation)
                  & (hsv[:, :, 2] >= 80)).astype(np.uint8)
        near_border = cv2.dilate(border, self._DILATE_KERNEL).astype(bool)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return (gray >= self.brightness) & near_border

    def update(self, crop):
        """Feed the next subtitle-positive crop. Returns True if its text
        differs from the previous positive frame's (i.e. a new subtitle).

        Stability rule: a change only fires when the PREVIOUS comparison was
        stable. During a fade/pop-in animation the mask morphs continuously
        (every comparison reads as change), so nothing fires; a settled
        subtitle replaced by another gives stable -> change and fires once.
        """
        mask = self._glyph_mask(crop)

        prev, self._prev = self._prev, mask
        if prev is None or prev.shape != mask.shape:
            self._prev_changed = False
            return False

        a = int(np.count_nonzero(mask))
        b = int(np.count_nonzero(prev))
        if a < self.min_pixels or b < self.min_pixels:
            self._prev_changed = False
            return False

        intersection = int(np.count_nonzero(mask & prev))
        union = a + b - intersection
        changed = (intersection / union) < self.iou_threshold
        fire = changed and not self._prev_changed
        self._prev_changed = changed
        return fire
