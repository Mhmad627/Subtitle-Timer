"""Crop the user-chosen subtitle region out of a frame.

The region is a rectangle expressed as fractions of the frame size
(x, y, w, h), so the same crop works at any video resolution. The GUI lets
the user drag/resize this box on a preview; the CLI takes it as --crop.
"""

# Bottom quarter, full width — a sensible starting box for most N subtitles.
DEFAULT_CROP = (0.0, 0.75, 1.0, 0.25)

# Starting box for the S style — placed above the N box so both are visible
# on first run; users move it to where their S subtitles actually appear.
DEFAULT_CROP_S = (0.0, 0.55, 1.0, 0.2)

# Starting box for a second, optional N region — placed at the top so it
# doesn't overlap the primary N box (bottom) by default.
DEFAULT_CROP_N2 = (0.0, 0.02, 1.0, 0.15)


def clamp_rect(rect):
    """Clamp an (x, y, w, h) fraction rect to a valid region inside the frame."""
    x, y, w, h = rect
    x = min(max(x, 0.0), 0.99)
    y = min(max(y, 0.0), 0.99)
    w = min(max(w, 0.01), 1.0 - x)
    h = min(max(h, 0.01), 1.0 - y)
    return (x, y, w, h)


def crop_rect(frame, rect=DEFAULT_CROP):
    """Return the part of `frame` covered by the fraction rect (x, y, w, h).

    Args:
        frame: BGR numpy array (H x W x C).
        rect: (x, y, w, h) as fractions of frame width/height, e.g.
            (0.0, 0.75, 1.0, 0.25) is the full-width bottom quarter.

    Returns:
        A view of the cropped frame (at least 1x1 pixels).
    """
    x, y, w, h = clamp_rect(rect)
    height, width = frame.shape[:2]

    x0 = int(width * x)
    y0 = int(height * y)
    x1 = max(x0 + 1, int(width * (x + w)))
    y1 = max(y0 + 1, int(height * (y + h)))

    return frame[y0:y1, x0:x1]
