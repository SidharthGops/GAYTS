"""
pipeline.py
───────────
OCR worker thread and queue management.

Responsibilities
────────────────
  • Owns the two queues (ocr_queue, result_queue).
  • Runs EasyOCR in a daemon thread so the main display loop never blocks.
  • Applies per-plate cooldown before saving snapshots and logging.
  • Calls the DetectionLogger (file / DB — injected at startup).
  • Feeds confirmed results back to the main thread for overlay display.

What changed vs the old livecam.py worker
──────────────────────────────────────────
  • frame.copy() removed — worker receives (crop, box_key, track_id),
    not a full 2.7MB frame copy.
  • Snapshot is saved from the crop only (or a slightly expanded crop
    passed in from the main loop — see livecam.py).
  • Cooldown check happens BEFORE the snapshot write, not after.
  • VoteBuffer consensus is applied: result is only forwarded once
    VOTE_THRESHOLD reads agree on the same plate text.
  • result_queue is bounded (maxsize=10) to prevent unbounded growth.
  • API calls send snapshot as multipart file upload (data + files).
"""
import requests
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from consensus import VoteBuffer

# ── Config (import from config.py in the full project) ───────────────────────

OCR_CONF_MIN      = 0.30
PLATE_COOLDOWN_S  = 5.0
VOTE_WINDOW       = 5
VOTE_THRESHOLD    = 2
SNAPSHOT_DIR      = Path("snapshots")

LOCAL_API_URL  = "http://127.0.0.1:8000/api/authorize"
REMOTE_API_URL = "http://10.241.37.137:8000/api/authorize"

# ── Queues ────────────────────────────────────────────────────────────────────
#
# ocr_queue    main → worker   plate crops to process
#              maxsize=2: if worker falls behind, new crops are dropped
#              rather than queued forever. Correct policy for real-time.
#
# result_queue worker → main   confirmed plates for overlay
#              bounded at 10 to prevent unbounded growth on reconnect.

ocr_queue    : queue.Queue = queue.Queue(maxsize=2)
result_queue : queue.Queue = queue.Queue(maxsize=10)


# ── Worker ────────────────────────────────────────────────────────────────────

class OCRWorker:
    """
    Wraps the daemon thread that runs EasyOCR in the background.

    Parameters
    ──────────
    ocr_fn   : callable  run_ocr(crop) → (text, conf) from ocr_engine.py
    logger   : object    Any DetectionLogger (has .log(plate, conf, snap))
    debug    : bool      If True, print verbose OCR output.
    """

    def __init__(self, ocr_fn, logger, debug: bool = False):
        self._ocr_fn   = ocr_fn
        self._logger   = logger
        self._debug    = debug
        self._vote_buf = VoteBuffer(window=VOTE_WINDOW, threshold=VOTE_THRESHOLD)
        self._last_logged: dict[str, float] = {}   # plate → last log timestamp

        SNAPSHOT_DIR.mkdir(exist_ok=True)

        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="ocr-worker",
        )

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        while True:
            try:
                item = ocr_queue.get(timeout=1)
            except queue.Empty:
                continue

            try:
                self._process(item)
            except Exception as e:
                # Worker must never crash — log and continue.
                print(f"[OCR-WORKER] Unhandled error: {e}")
            finally:
                ocr_queue.task_done()

    def _process(self, item: tuple) -> None:
        """
        item: (plate_crop, context_crop, box_coords, track_id)

          plate_crop   : np.ndarray  tight plate crop for OCR
          context_crop : np.ndarray  slightly wider crop for snapshot JPEG
          box_coords   : (x1,y1,x2,y2) in full-res coords, for overlay
          track_id     : int  stable tracker ID for vote buffer
        """
        plate_crop, context_crop, box_coords, track_id = item

        text, conf = self._ocr_fn(plate_crop)

        if self._debug:
            print(f"[OCR] track={track_id}  raw={text!r}  conf={conf:.2f}")

        if not text or conf < OCR_CONF_MIN:
            return

        # ── Multi-frame consensus ─────────────────────────────────────────
        agreed_text = self._vote_buf.add_vote(track_id, text)
        if agreed_text is None:
            if self._debug:
                votes = self._vote_buf.vote_count(track_id)
                print(f"[OCR] Accumulating votes for track={track_id} ({votes}/{VOTE_THRESHOLD})")
            return

        # ── Cooldown check (BEFORE snapshot write) ────────────────────────
        now = time.time()
        if now - self._last_logged.get(agreed_text, 0) < PLATE_COOLDOWN_S:
            # Still in cooldown — update overlay but skip log + snapshot
            self._push_result(agreed_text, conf, box_coords)
            return

        self._last_logged[agreed_text] = now

        # ── Save snapshot ─────────────────────────────────────────────────
        ts        = time.strftime("%Y%m%d_%H%M%S")
        snap_name = f"{agreed_text}_{ts}.jpg"
        snap_path = str(SNAPSHOT_DIR / snap_name)
        cv2.imwrite(snap_path, context_crop)

        # Confidence as a 0–100 percentage score, e.g. 0.90 → 90.0
        conf_pct = conf

        # ── Push to local server (optional — skip if not running) ─────────
        try:
            with open(snap_path, "rb") as fh:
                response = requests.post(
                    LOCAL_API_URL,
                    data={
                        "plate_number":     agreed_text,
                        "confidence_score": conf_pct,
                    },
                    files={"snapshot": (snap_name, fh, "image/jpeg")},
                    timeout=3,
                )
            print(f"[API] Local status: {response.status_code}")
        except requests.exceptions.ConnectionError:
            pass   # local server not running — silently skip
        except Exception as e:
            print(f"[API] Local push failed: {e}")

        # ── Log + push to remote DB ───────────────────────────────────────
        print(f"[ANPR] ✓  {agreed_text}  conf={conf:.2f}  snap={snap_name}")
        try:
            with open(snap_path, "rb") as fh:
                response = requests.post(
                    REMOTE_API_URL,
                    data={
                        "plate_number":     agreed_text,
                        "confidence_score": conf_pct,
                    },
                    files={"snapshot": (snap_name, fh, "image/jpeg")},
                    timeout=3,
                )
            print(f"[API] Detection pushed to DB")
            print(response.json())
        except Exception as e:
            print(f"[API] Remote push failed: {e}")

        # ── Update overlay ────────────────────────────────────────────────
        self._push_result(agreed_text, conf, box_coords)

    def _push_result(self, text: str, conf: float, box_coords: tuple) -> None:
        """Push a confirmed plate result to result_queue for overlay display."""
        try:
            result_queue.put_nowait((text, conf, box_coords))
        except queue.Full:
            pass   # main thread is behind — drop silently

    def notify_track_removed(self, track_id: int) -> None:
        """
        Call this when CentroidTracker drops a track ID.
        Clears the vote buffer so stale votes don't bleed into a new
        vehicle that later gets the same ID.
        """
        self._vote_buf.clear(track_id)


# ── Queue helpers used by livecam.py ─────────────────────────────────────────

def enqueue_crop(
    plate_crop    : np.ndarray,
    context_crop  : np.ndarray,
    box_coords    : tuple[int, int, int, int],
    track_id      : int,
) -> bool:
    """
    Non-blocking enqueue. Returns True if accepted, False if queue was full
    (crop dropped — correct real-time behaviour).
    """
    try:
        ocr_queue.put_nowait((
            plate_crop.copy(),
            context_crop.copy(),
            box_coords,
            track_id,
        ))
        return True
    except queue.Full:
        return False


def drain_results() -> list[tuple[str, float, tuple]]:
    """
    Drain all pending OCR results from result_queue.
    Returns list of (plate_text, conf, box_coords).
    Call once per display frame.
    """
    results = []
    try:
        while True:
            results.append(result_queue.get_nowait())
    except queue.Empty:
        pass
    return results