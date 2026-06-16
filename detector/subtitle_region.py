"""Locate / crop the region of a frame where subtitles usually appear.

Phase 1 uses a simple heuristic: subtitles are burned into the bottom strip
of the frame. The crop fraction is configurable. Later phases can replace
`crop_region` with actual text-region detection (e.g. MSER/EAST).
"""


def crop_region(frame, region=0.25):
    """Return the bottom `region` fraction of the frame.

    Args:
        frame: BGR numpy array (H x W x C).
        region: Fraction of the frame height to keep, measured from the
            bottom. 0.25 keeps the bottom 25%. Clamped to (0, 1].

    Returns:
        A view of the cropped frame (bottom strip).
    """
    if region <= 0 or region > 1:
        region = 0.25

    height = frame.shape[0]
    start_row = int(height * (1.0 - region))
    return frame[start_row:height, :]
