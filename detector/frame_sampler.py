"""Extract frames from a video at a target sampling rate using OpenCV."""

import cv2


def sample_frames(video_path, fps=2.0):
    """Yield (timestamp_seconds, frame) tuples sampled at ~`fps` frames/sec.

    The video's native frame rate is read from OpenCV; we then keep roughly
    every (native_fps / fps)-th frame. Timestamps are derived from the frame
    index so they stay accurate regardless of the sampling stride.

    Args:
        video_path: Path to the video file.
        fps: Desired sampling rate in frames per second.

    Yields:
        (timestamp, frame): timestamp in seconds (float), frame as a BGR
        numpy array.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    try:
        native_fps = cap.get(cv2.CAP_PROP_FPS)
        if not native_fps or native_fps <= 0:
            # Fallback: assume 30 fps if metadata is missing/garbage.
            native_fps = 30.0

        # How many native frames to advance between samples (at least 1).
        stride = max(1, int(round(native_fps / fps)))

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
