"""Assemble subtitle blocks into a properly formatted .srt file."""


def format_timestamp(seconds):
    """Convert a time in seconds to SRT format: HH:MM:SS,mmm."""
    if seconds < 0:
        seconds = 0.0

    total_ms = int(round(seconds * 1000))
    hours, total_ms = divmod(total_ms, 3600 * 1000)
    minutes, total_ms = divmod(total_ms, 60 * 1000)
    secs, millis = divmod(total_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt(blocks, output_path):
    """Write subtitle blocks to an .srt file.

    Args:
        blocks: Iterable of objects with `.start`, `.end` (seconds) and
            `.text` attributes, in time order.
        output_path: Destination path for the .srt file.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for index, block in enumerate(blocks, start=1):
            start = format_timestamp(block.start)
            end = format_timestamp(block.end)
            f.write(f"{index}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{block.text}\n")
            f.write("\n")
