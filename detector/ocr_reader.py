"""Run OCR on a (cropped) frame using EasyOCR."""


class OCRReader:
    """Thin wrapper around an EasyOCR reader.

    EasyOCR loads its models once on construction, which is slow, so create
    one `OCRReader` and reuse it across all frames.
    """

    def __init__(self, languages=None, gpu=False, min_confidence=0.3):
        # Import lazily so `python main.py --help` works without the heavy deps.
        import easyocr

        self.languages = languages or ["en"]
        self.min_confidence = min_confidence
        self._reader = easyocr.Reader(self.languages, gpu=gpu)

    def read(self, image):
        """Return the combined subtitle text found in `image`, or "" if none.

        Lines detected by EasyOCR are joined with spaces. Low-confidence
        detections (below `min_confidence`) are dropped to reduce noise.

        Args:
            image: BGR numpy array (a cropped frame strip).

        Returns:
            A single cleaned text string (may be empty).
        """
        results = self._reader.readtext(image)

        parts = []
        for _box, text, confidence in results:
            if confidence < self.min_confidence:
                continue
            cleaned = text.strip()
            if cleaned:
                parts.append(cleaned)

        return " ".join(parts).strip()
