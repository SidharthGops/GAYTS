"""
frame_buffer.py
───────────────
Thread-safe single-frame buffer shared between the ANPR pipeline
and the FastAPI MJPEG stream endpoint.

Why a single frame (not a queue)?
──────────────────────────────────
The stream always wants the *latest* frame, not old ones. A queue
would let the stream fall behind under load and serve stale footage.
Overwriting the buffer means the stream always reads the most recent
annotated frame, even if it skips some.

Usage
─────
  # In livecam.py (writer):
      from frame_buffer import shared_buffer
      shared_buffer.write(frame)

  # In FastAPI stream endpoint (reader):
      from frame_buffer import shared_buffer
      frame = shared_buffer.wait_for_frame()
"""

import threading
import numpy as np


class FrameBuffer:
    """
    Holds the single most recent frame from the ANPR camera loop.

    write()          – called by livecam.py after every cap.read()
    read()           – non-blocking, returns latest frame or None
    wait_for_frame() – blocks until a new frame arrives (used by streamer)
    """

    def __init__(self) -> None:
        self._frame    : np.ndarray | None = None
        self._condition = threading.Condition(threading.Lock())

    def write(self, frame: np.ndarray) -> None:
        """
        Store a new frame and wake all waiting readers.
        Called from the main ANPR thread on every cap.read().
        Makes a copy so the pipeline can keep drawing on the original.
        """
        with self._condition:
            self._frame = frame.copy()
            self._condition.notify_all()

    def read(self) -> np.ndarray | None:
        """
        Return the latest frame without blocking.
        Returns None if no frame has been written yet.
        """
        with self._condition:
            return self._frame.copy() if self._frame is not None else None

    def wait_for_frame(self, timeout: float = 1.0) -> np.ndarray | None:
        """
        Block until a new frame is written, then return it.
        Returns None on timeout (stream loop should just continue).

        Parameters
        ──────────
        timeout : float   Seconds to wait before giving up (default 1.0).
                          Keeps the stream generator from hanging forever
                          if the ANPR loop exits.
        """
        with self._condition:
            notified = self._condition.wait(timeout=timeout)
            if not notified or self._frame is None:
                return None
            return self._frame.copy()


# ── Singleton ─────────────────────────────────────────────────────────────────
# Both livecam.py and stream_route.py import this same object.
# Python's module system guarantees a single instance per process.

shared_buffer = FrameBuffer()