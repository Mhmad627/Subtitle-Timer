"""Audio fallback transcription using OpenAI Whisper (local, free).

When Phase 1's visual OCR finds no (or very few) burned-in subtitles, the
pipeline falls back here: extract the audio track with FFmpeg, run Whisper,
and return the same `SubtitleBlock` list the OCR path produces.

Whisper segments already include start/end times and clean text, so no
deduplication is needed — we map segments straight to blocks.
"""

import os
import subprocess
import tempfile

from utils.deduplicator import SubtitleBlock


def extract_audio(video_path):
    """Extract the audio track to a temporary 16 kHz mono WAV.

    Uses FFmpeg (must be installed and on PATH). The caller is responsible
    for deleting the returned temp file.

    Args:
        video_path: Path to the input video.

    Returns:
        Path to the temporary .wav file.

    Raises:
        FileNotFoundError: If FFmpeg is not installed / not on PATH.
        RuntimeError: If FFmpeg fails to extract audio.
    """
    # Create the temp path, then close the handle so FFmpeg can write to it
    # on Windows (where an open handle would block the write). delete=False
    # means we own cleanup.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name

    cmd = [
        "ffmpeg",
        "-y",                     # overwrite the empty temp file
        "-i", video_path,
        "-vn",                    # no video
        "-acodec", "pcm_s16le",   # WAV format Whisper likes
        "-ar", "16000",           # 16 kHz, Whisper's preferred rate
        "-ac", "1",               # mono
        temp_path,
    ]

    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        os.remove(temp_path)
        raise FileNotFoundError(
            "FFmpeg not found. Install it and make sure 'ffmpeg' is on your PATH."
        )
    except subprocess.CalledProcessError as exc:
        os.remove(temp_path)
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
        raise RuntimeError(f"FFmpeg failed to extract audio:\n{stderr}")

    return temp_path


def transcribe(video_path, model_name="base"):
    """Transcribe a video's audio into timed subtitle blocks.

    Args:
        video_path: Path to the input video.
        model_name: Whisper model size — tiny / base / small / medium.

    Returns:
        List[SubtitleBlock] in time order.
    """
    # Import lazily so the OCR-only path doesn't require whisper installed.
    import whisper

    temp_audio_path = extract_audio(video_path)
    try:
        model = whisper.load_model(model_name)
        result = model.transcribe(temp_audio_path)

        blocks = []
        for segment in result.get("segments", []):
            text = segment["text"].strip()
            if text:
                blocks.append(
                    SubtitleBlock(segment["start"], segment["end"], text)
                )
        return blocks
    finally:
        # Always clean up the temp audio, even if transcription crashes.
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
