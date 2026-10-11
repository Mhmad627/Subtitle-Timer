"""Extract frames from a video at a target sampling rate using OpenCV."""

import cv2


def _native_fps(cap):
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if not native_fps or native_fps <= 0:
        # Fallback: assume 30 fps if metadata is missing/garbage.
        native_fps = 30.0
    return native_fps


def _stride(native_fps, fps):
    """How many native frames to advance between samples (at least 1)."""
    if fps and fps > 0:
        return max(1, int(round(native_fps / fps)))
    return 1  # every frame


def estimate_sample_count(video_path, fps):
    """Roughly how many frames sample_frames will yield, from the video's
    frame-count metadata (which can be a little off); 0 if the metadata is
    missing, None if the video can't be opened at all."""
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            return None
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        stride = _stride(_native_fps(cap), fps)
    finally:
        cap.release()
    return -(-total // stride)  # ceil: frames 0, stride, 2*stride, ...


def video_size(video_path):
    """The video's (width, height) in pixels, or None if unreadable."""
    cap = cv2.VideoCapture(video_path)
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    return (width, height) if width > 0 and height > 0 else None


def sample_frames(video_path, fps=2.0):
    """Yield (timestamp_seconds, frame) tuples sampled at ~`fps` frames/sec.

    The video's native frame rate is read from OpenCV; we then keep roughly
    every (native_fps / fps)-th frame. Timestamps are derived from the frame
    index so they stay accurate regardless of the sampling stride.

    Args:
        video_path: Path to the video file.
        fps: Desired sampling rate in frames per second. Zero (or None)
            means every native frame — frame-accurate sampling.

    Yields:
        (timestamp, frame): timestamp in seconds (float), frame as a BGR
        numpy array.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    try:
        native_fps = _native_fps(cap)
        stride = _stride(native_fps, fps)

        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % stride == 0:
                timestamp = frame_index / native_fps
                yield timestamp, frame

            frame_index += 1
    finally:
        cap.release()
