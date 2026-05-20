"""
livecam.py
──────────
Real-time ANPR main loop.

Architecture
────────────
  Main thread   cap.read → YOLO (every N frames) → tracker → display
  Worker thread EasyOCR → consensus → snapshot → logger
                (managed by pipeline.py — main thread never blocks on OCR)
  Stream        FastAPI reads annotated frames from shared_buffer
                (frame_buffer.py) — zero extra camera opens

Visual cues
───────────
  Yellow box   YOLO detected, OCR in progress / accumulating votes
  Green box    Consensus confirmed — plate text shown above box
  Label fmt    "KA05MG1909  87%"
"""

import time

import cv2

from tracker      import CentroidTracker
from pipeline     import OCRWorker, enqueue_crop, drain_results
from frame_buffer import shared_buffer   # ← shared with FastAPI stream

# Loaded from yoloconfig or models.py — whichever name you kept
from yoloconfig import get_model, run_ocr

# ── Config ────────────────────────────────────────────────────────────────────
# Move these to config.py in the full project.

YOLO_CONF_MIN    = 0.40   # ignore YOLO detections below this
OCR_THROTTLE_S   = 1.0    # min seconds between OCR jobs per track
OVERLAY_HOLD_S   = 3.0    # seconds the confirmed plate label stays on screen
YOLO_INPUT_W     = 640    # frame width fed to YOLO
YOLO_EVERY_N     = 3      # run YOLO every Nth frame (tracker fills gaps)
CONTEXT_PAD      = 40     # extra pixels around plate for snapshot crop


# ── Logger stub (replace with your DB logger from logger.py) ─────────────────

class _FileLogger:
    def log(self, plate: str, conf: float, snap: str) -> None:
        pass   # pipeline.py already prints the line; add DB call here


# ── Main entry point ──────────────────────────────────────────────────────────

def run(
    camera_index : int  = 0,
    model_path   : str  = "models/best.pt",
    debug        : bool = False,
) -> None:
    """
    Start the live ANPR loop.

    Parameters
    ──────────
    camera_index : int   cv2.VideoCapture index (0 = default webcam)
    model_path   : str   Path to YOLOv8 .pt weights file
    debug        : bool  Print verbose per-frame OCR output
    """

    # ── Model + camera ────────────────────────────────────────────────────
    model = get_model()   # singleton — safe to call multiple times

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[livecam] ERROR: cannot open camera index {camera_index}")
        return

    # Optional: request a specific resolution. Camera may ignore this.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # ── Subsystems ────────────────────────────────────────────────────────
    tracker = CentroidTracker(max_distance=400, max_missing=10)
    worker  = OCRWorker(ocr_fn=run_ocr, logger=_FileLogger(), debug=debug)
    worker.start()

    # ── State ─────────────────────────────────────────────────────────────
    frame_count  : int              = 0
    last_ocr_sent: dict[int, float] = {}   # track_id → last enqueue timestamp

    # Overlay: the last confirmed detection to render on screen
    overlay_text    : str   = ""
    overlay_conf    : float = 0.0
    overlay_box     : tuple | None = None
    overlay_expires : float = 0.0

    # Last known boxes from tracker (drawn yellow between YOLO frames)
    active_boxes: dict[int, tuple[int,int,int,int]] = {}

    print("[livecam] Running — press Q to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[livecam] Failed to read frame — retrying...")
            time.sleep(0.05)
            continue

        h, w = frame.shape[:2]
        now  = time.monotonic()
        frame_count += 1

        # ── YOLO inference (every N frames) ───────────────────────────────
        if frame_count % YOLO_EVERY_N == 0:
            scale = YOLO_INPUT_W / w
            small = cv2.resize(frame, (YOLO_INPUT_W, int(h * scale)))

            yolo_out = model(small, verbose=False)

            raw_boxes: list[tuple[int,int,int,int]] = []
            for result in yolo_out:
                for box in result.boxes:
                    if float(box.conf[0]) < YOLO_CONF_MIN:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    # Scale coords back to original resolution
                    x1 = int(x1 / scale); x2 = int(x2 / scale)
                    y1 = int(y1 / scale); y2 = int(y2 / scale)
                    raw_boxes.append((x1, y1, x2, y2))

            active_boxes = tracker.update(raw_boxes)

            # ── Enqueue OCR jobs for each active track ─────────────────
            for track_id, (x1, y1, x2, y2) in active_boxes.items():

                # Per-track throttle using stable ID (not jittery pixels)
                if now - last_ocr_sent.get(track_id, 0) < OCR_THROTTLE_S:
                    continue

                # Tight plate crop for OCR
                pad = 8
                px1, px2 = max(0, x1 - pad), min(w, x2 + pad)
                py1, py2 = max(0, y1 - pad), min(h, y2 + pad)
                plate_crop = frame[py1:py2, px1:px2]
                if plate_crop.size == 0:
                    continue

                # Wider context crop for snapshot JPEG (~10× smaller than full frame)
                cx1 = max(0, x1 - CONTEXT_PAD)
                cy1 = max(0, y1 - CONTEXT_PAD)
                cx2 = min(w, x2 + CONTEXT_PAD)
                cy2 = min(h, y2 + CONTEXT_PAD)
                context_crop = frame[cy1:cy2, cx1:cx2]

                accepted = enqueue_crop(
                    plate_crop   = plate_crop,
                    context_crop = context_crop,
                    box_coords   = (x1, y1, x2, y2),
                    track_id     = track_id,
                )
                if accepted:
                    last_ocr_sent[track_id] = now

        # ── Drain OCR results ─────────────────────────────────────────────
        for text, conf, box_coords in drain_results():
            overlay_text    = text
            overlay_conf    = conf
            overlay_box     = box_coords
            overlay_expires = now + OVERLAY_HOLD_S

        # ── Draw yellow boxes for all active YOLO tracks ──────────────────
        for track_id, (x1, y1, x2, y2) in active_boxes.items():
            pad = 8
            px1 = max(0, x1 - pad); px2 = min(w, x2 + pad)
            py1 = max(0, y1 - pad); py2 = min(h, y2 + pad)
            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 200, 255), 2)

        # ── Draw confirmed overlay (green box + label) ────────────────────
        if overlay_box and now < overlay_expires:
            x1, y1, x2, y2 = overlay_box
            label = f"{overlay_text}  {overlay_conf:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 8, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1 + 4, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        elif now >= overlay_expires:
            overlay_text = ""
            overlay_box  = None

        # ── FPS counter (debug only) ──────────────────────────────────────
        if debug:
            fps_label = f"Frame {frame_count}"
            cv2.putText(frame, fps_label, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # ── Publish annotated frame to shared buffer ──────────────────────
        # The FastAPI MJPEG endpoint reads from here.
        # All YOLO boxes and plate labels are already drawn on `frame`
        # at this point — the stream gets the same view as the local window.
        shared_buffer.write(frame)

        cv2.imshow("Live ANPR", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n[livecam] Stopped.")


if __name__ == "__main__":
    run()