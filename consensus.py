"""
consensus.py
────────────
Multi-frame OCR vote buffer.

Problem it solves
─────────────────
EasyOCR on a single frame is noisy. Motion blur, lighting changes, and
OCR variance mean a single read can be wrong even on a clear plate.
Running OCR on 5 frames and taking the majority result is far more
reliable than trusting the first read above a confidence threshold.

Usage
─────
    buf = VoteBuffer(window=5, threshold=3)

    # Call on every confirmed OCR result from the worker thread:
    consensus = buf.add_vote(track_id=7, text="KA05MG1909")

    # consensus is the agreed plate text if threshold is reached,
    # or None if not enough votes yet.

    # When a track is removed by the tracker, clean up its buffer:
    buf.clear(track_id=7)
"""

from collections import Counter, deque


class VoteBuffer:
    """
    Sliding-window majority-vote consensus per tracked plate.

    Parameters
    ──────────
    window    : int   Number of recent OCR reads to consider per track.
    threshold : int   Minimum votes for the leading candidate to be
                      declared the consensus. Must be <= window.

    Design notes
    ────────────
    - Uses a deque per track so old reads fall off automatically.
    - Only valid plate strings (already filtered by PLATE_RE upstream)
      should be passed in — do not pass empty strings or failed reads.
    - Thread-safe for the single-writer / single-reader pattern used
      in this project (worker writes, main thread reads via result_queue).
      If you ever add multiple OCR workers, add a threading.Lock.
    """

    def __init__(self, window: int = 5, threshold: int = 3):
        if threshold > window:
            raise ValueError("threshold cannot exceed window size")
        self.window    = window
        self.threshold = threshold
        self._buffers: dict[int, deque[str]] = {}

    def add_vote(self, track_id: int, text: str) -> str | None:
        """
        Record one OCR read for a tracked plate.

        Returns the consensus plate string if the vote threshold is
        reached, otherwise returns None.

        Parameters
        ──────────
        track_id : int   Stable ID from CentroidTracker.
        text     : str   Validated plate text (e.g. "KA05MG1909").
                         Must be non-empty.
        """
        if not text:
            return None

        buf = self._buffers.setdefault(track_id, deque(maxlen=self.window))
        buf.append(text)

        counts = Counter(buf)
        best, count = counts.most_common(1)[0]

        if count >= self.threshold:
            return best

        return None

    def clear(self, track_id: int) -> None:
        """Remove vote history for a track (call when tracker drops the ID)."""
        self._buffers.pop(track_id, None)

    def clear_all(self) -> None:
        """Reset all vote buffers (call on pipeline restart)."""
        self._buffers.clear()

    def peek(self, track_id: int) -> str | None:
        """
        Return the current leading candidate without requiring threshold.
        Useful for debug overlays showing 'tentative' plate text.
        """
        buf = self._buffers.get(track_id)
        if not buf:
            return None
        counts = Counter(buf)
        return counts.most_common(1)[0][0]

    def vote_count(self, track_id: int) -> int:
        """Total votes recorded for this track (for debug displays)."""
        return len(self._buffers.get(track_id, []))