"""
stream_route.py
───────────────
FastAPI router that serves the live ANPR camera as an MJPEG stream.

The frames come from frame_buffer.shared_buffer, which livecam.py
writes to after every cap.read(). No second camera open — zero conflict.

How to add to your existing FastAPI app
────────────────────────────────────────
    # In your main FastAPI file (e.g. app.py / server.py):
    from stream_route import router as stream_router
    app.include_router(stream_router)

Then point the dashboard img tag at:
    http://<your-server-ip>:8000/api/camera/stream

IMPORTANT: livecam.py must be running in the same process (or at least
the same machine with a shared import) for frames to appear.
The recommended way is to launch livecam in a background thread from
your FastAPI startup event — see the startup example at the bottom.
"""

import threading
import cv2

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from frame_buffer import shared_buffer

router = APIRouter()

# ── MJPEG quality (0–100). 60 is a good balance of size vs clarity. ──────────
STREAM_JPEG_QUALITY = 60


def _generate_mjpeg():
    """
    Generator that yields MJPEG boundary frames indefinitely.

    Blocks on shared_buffer.wait_for_frame() so it only pushes data
    when a new frame is available — no busy-loop, no duplicate frames.
    On timeout (ANPR loop paused / not started) it loops and retries.
    """
    while True:
        frame = shared_buffer.wait_for_frame(timeout=1.0)
        if frame is None:
            # ANPR loop hasn't started yet or is paused — keep waiting
            continue

        ok, buffer = cv2.imencode(
            '.jpg', frame,
            [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY],
        )
        if not ok:
            continue

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            buffer.tobytes() +
            b'\r\n'
        )


@router.get("/api/camera/stream")
def camera_stream():
    """
    MJPEG stream of the live ANPR camera feed with annotations.

    Use as the src of an <img> tag in the dashboard:
        <img src="http://SERVER_IP:8000/api/camera/stream">

    The stream includes all YOLO bounding boxes and confirmed plate
    overlays drawn by livecam.py — no extra processing needed here.
    """
    return StreamingResponse(
        _generate_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ── Optional: launch livecam in a background thread from FastAPI startup ──────
#
# If you want everything to start with a single `uvicorn app:app` command,
# add this to your FastAPI app file instead of running main.py separately:
#
#   from contextlib import asynccontextmanager
#   from stream_route import router as stream_router, start_anpr_thread
#
#   @asynccontextmanager
#   async def lifespan(app):
#       start_anpr_thread(camera_index=0, debug=False)
#       yield
#
#   app = FastAPI(lifespan=lifespan)
#   app.include_router(stream_router)
#
# ─────────────────────────────────────────────────────────────────────────────

def start_anpr_thread(camera_index: int = 0, debug: bool = False) -> None:
    """
    Start livecam.run() in a daemon thread so it shares the process
    (and therefore the same shared_buffer instance) with FastAPI.

    Call this from your FastAPI lifespan/startup event.
    """
    import livecam

    t = threading.Thread(
        target=livecam.run,
        kwargs={"camera_index": camera_index, "debug": debug},
        daemon=True,
        name="anpr-livecam",
    )
    t.start()
    print(f"[stream] ANPR thread started (camera={camera_index})")