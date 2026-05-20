# GAYTS — Technical Documentation

> Deep-dive reference for every module, class, function, and design decision in the GAYTS ANPR gate system.

---

## Table of Contents

1. [System Boot Sequence](#1-system-boot-sequence)
2. [Module Reference](#2-module-reference)
   - [db.py](#dbpy)
   - [models.py](#modelspy)
   - [auth.py](#authpy)
   - [yoloconfig.py](#yoloconfigpy)
   - [tracker.py](#trackerpy)
   - [consensus.py](#consensuspy)
   - [frame_buffer.py](#frame_bufferpy)
   - [pipeline.py](#pipelinepy)
   - [livecam.py](#livecampy)
   - [stream_route.py](#stream_routepy)
   - [ocr.py](#ocrpy)
   - [main.py](#mainpy)
3. [Data Flow Diagrams](#3-data-flow-diagrams)
4. [Threading Model](#4-threading-model)
5. [OCR Pipeline Deep-Dive](#5-ocr-pipeline-deep-dive)
6. [Indian Plate Format & Correction](#6-indian-plate-format--correction)
7. [Database Design Decisions](#7-database-design-decisions)
8. [Configuration Reference](#8-configuration-reference)
9. [Error Handling Strategy](#9-error-handling-strategy)
10. [Frontend Architecture](#10-frontend-architecture)

---

## 1. System Boot Sequence

```
1.  FastAPI starts:  uvicorn main:app
      └─ SQLAlchemy creates tables if not exist
      └─ StaticFiles mounts /frames directory
      └─ CORS middleware registered

2.  ANPR loop starts:  python livecam.py
      └─ yoloconfig.get_model()   → loads YOLOv8 weights (once)
      └─ yoloconfig.get_reader()  → loads EasyOCR (once, ~3-5 seconds)
      └─ CentroidTracker()        → initialised empty
      └─ OCRWorker().start()      → daemon thread begins blocking on ocr_queue
      └─ cv2.VideoCapture(0)      → camera opened

3.  Per-frame loop:
      └─ cap.read()
      └─ frame_count % YOLO_EVERY_N == 0 → run YOLO
      └─ tracker.update(boxes)    → assign stable IDs
      └─ enqueue_crop()           → push to ocr_queue (non-blocking)
      └─ drain_results()          → pull confirmed plates from result_queue
      └─ draw overlays
      └─ shared_buffer.write()    → publish for MJPEG stream
      └─ cv2.imshow()

4.  FastAPI MJPEG stream (/api/camera/stream):
      └─ shared_buffer.wait_for_frame() → blocks until new frame written
      └─ cv2.imencode JPEG
      └─ yield MJPEG boundary
```

---

## 2. Module Reference

---

### `db.py`

Initialises the SQLAlchemy engine and session factory from the environment.

```python
engine       = create_engine(os.getenv("DATABASE_URL"))
SessionLocal = sessionmaker(bind=engine)
Base         = declarative_base()
```

**`get_db()`**
FastAPI dependency that yields a database session and guarantees it is closed after the request, even on exception.

```python
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

Used with `Depends(get_db)` in every endpoint that touches the database.

---

### `models.py`

Defines the two ORM models that map to PostgreSQL tables.

**`Vehicle`** — the authorised plate whitelist.

| Attribute | Column | Constraint |
|---|---|---|
| `id` | `INTEGER` | Primary key, auto-increment |
| `plate_number` | `VARCHAR(20)` | Unique, not null |
| `registered_at` | `TIMESTAMP` | Server default `NOW()` |

**`GateLog`** — immutable record of every detection event.

| Attribute | Column | Notes |
|---|---|---|
| `id` | `INTEGER` | Primary key, auto-increment |
| `plate_number` | `VARCHAR(20)` | Raw OCR text — may not be in vehicles |
| `status` | `VARCHAR(15)` | `'AUTHORIZED'` or `'UNAUTHORIZED'` |
| `confidence_score` | `FLOAT` | 0.0–1.0 from EasyOCR |
| `snapshot_path` | `TEXT` | Relative path to saved JPEG, nullable |
| `timestamp` | `TIMESTAMP` | Server default `NOW()` |

Both models are created at startup by `Base.metadata.create_all(bind=engine)` in `main.py`.

---

### `auth.py`

**`check_and_log(plate_number, confidence_score, snapshot_path) → dict | None`**

A self-contained authorisation helper that opens its own DB session. Intended for direct use in scripts, tests, and MQTT callbacks that run outside the FastAPI request lifecycle.

**Logic:**
1. Query `vehicles` table for an exact `plate_number` match.
2. Set `status = "AUTHORIZED"` if found, else `"UNAUTHORIZED"`.
3. Insert a new `GateLog` row with the status, confidence score, and optional snapshot path.
4. Return a summary dict: `{log_id, plate_number, status, confidence_score, timestamp}`.
5. On any DB exception: rollback, print error, return `None`.
6. Always: close the session in `finally`.

> **Note:** The API endpoint in `main.py` uses a similar flow but receives the session via `Depends(get_db)` and handles file upload separately. `auth.py` is the standalone version for non-HTTP callers.

---

### `yoloconfig.py`

Central utility module for all AI model interactions. Imported by both `ocr.py` (batch mode) and `livecam.py` (real-time mode). All expensive objects are module-level singletons — they are initialised once and reused across calls.

**Singletons**

| Symbol | Type | Lazy-loaded by |
|---|---|---|
| `_yolo_model` | `ultralytics.YOLO` | `get_model()` |
| `_ocr_reader` | `easyocr.Reader` | `get_reader()` |

**`detect_plate(image_path) → (crop, annotated_img, confidence)`**

Single-image plate detector. Runs YOLO on a file path, selects the largest bounding box by area, pads it by 10 px, and returns the crop alongside the annotated full image and confidence score. Returns `(None, None, 0.0)` if no plate is found.

**`preprocess_plate(plate_bgr) → np.ndarray`**

Image enhancement pipeline for an OpenCV BGR plate crop:

```
Input BGR crop
    │
    ├─ width < 200px? → bicubic upscale to 200px width
    │
    ▼
Grayscale conversion
    │
    ▼
CLAHE (clipLimit=2.0, tileGridSize=8×8)  ← contrast normalisation
    │
    ▼
Gaussian blur (3×3)  ← noise reduction
    │
    ▼
Otsu threshold  ← binary image
    │
    ▼
Output: binary np.ndarray
```

The conditional upscale avoids tripling the compute cost on already-large crops (e.g. a 300 px wide crop does not need 3× scaling).

**`smart_correct(s) → str`**

OCR character correction using knowledge of Indian plate structure. Returns the string unchanged if it already passes `PLATE_RE` or is outside the expected length range (8–11 chars).

Correction maps:

| Map | When applied | Examples |
|---|---|---|
| `L2D` (letter→digit) | Digit zones (district number, serial) | `O→0`, `I→1`, `Z→2`, `S→5`, `B→8`, `Q→0`, `D→0` |
| `D2L` (digit→letter) | Letter zones (state code, series) | `0→O`, `1→I`, `5→S`, `8→B` |

Zone assignment follows the fixed Indian plate structure: positions 0–1 are always letters (state), the next 1–2 characters are district digits, then series letters, then 4 serial digits.

**`run_ocr(plate_bgr) → (text, confidence)`**

Full OCR runner combining preprocessing, dual-pass reading, token merging, correction, and validation:

```
1. preprocess_plate()  →  binary image
2. cv2.bitwise_not()   →  inverted image  (handles dark plates)
3. For each of [binary, inverted]:
      EasyOCR readtext()
          allowlist: A-Z 0-9
          text_threshold: 0.3
          width_ths: 0.9  (merge horizontally close tokens)
      Sort tokens left-to-right by x coordinate
      Join tokens → single string
      smart_correct()
      PLATE_RE.match() → accept if valid
4. Return best (text, confidence) across both passes
```

Returns `('', 0.0)` if no valid plate is found in either pass.

---

### `tracker.py`

**`CentroidTracker`**

Nearest-centroid bounding box tracker that assigns stable integer IDs to YOLO detections across frames. Solves the problem that pixel-coordinate keys break the OCR throttle every time a box shifts by a single pixel.

**Constructor parameters**

| Parameter | Default | Description |
|---|---|---|
| `max_distance` | 60 px | Maximum centroid shift to match same vehicle |
| `max_missing` | 10 frames | Frames without detection before track is removed |

**`update(boxes) → dict[track_id, (x1, y1, x2, y2)]`**

Called once per YOLO inference frame.

```
Input:  list of (x1, y1, x2, y2) boxes from YOLO
Output: dict { stable_id: (x1, y1, x2, y2) }

Algorithm:
1. Compute centroid of each input box.
2. If no existing tracks → register all as new.
3. If no input boxes → age all tracks, remove expired.
4. Otherwise:
   a. Build distance matrix (existing centroids × input centroids).
   b. Greedy nearest-match: sort all pairs by distance, assign
      best match for each existing track that hasn't been matched.
   c. Pairs exceeding max_distance → skipped (new object).
   d. Unmatched existing tracks → age by 1; remove if > max_missing.
   e. Unmatched input centroids → register as new tracks.
```

Internal state: `centroids` (OrderedDict), `missing` (dict), `boxes` (dict), `next_id` (int counter).

**`get_box(track_id) → (x1, y1, x2, y2) | None`**

Returns the last known bounding box for a track ID, or `None` if the ID is not active.

---

### `consensus.py`

**`VoteBuffer`**

Multi-frame OCR majority vote buffer. Eliminates single-frame OCR noise by requiring a plate text to appear in multiple consecutive reads before being accepted.

**Constructor parameters**

| Parameter | Default | Description |
|---|---|---|
| `window` | 5 | Number of recent reads to keep per track (deque maxlen) |
| `threshold` | 3 | Minimum votes for the leading candidate |

**`add_vote(track_id, text) → str | None`**

Appends an OCR read to the deque for `track_id` (created on first call). Returns the consensus plate string when the most common reading reaches `threshold` votes, otherwise `None`.

**`clear(track_id)`**
Must be called when `CentroidTracker` drops a track ID to prevent stale votes from bleeding into a new vehicle that inherits the same numeric ID.

**`clear_all()`**
Resets all buffers — call on pipeline restart.

**`peek(track_id) → str | None`**
Returns the current leading candidate without requiring threshold. Useful for debug overlays showing tentative (unconfirmed) plate text.

**`vote_count(track_id) → int`**
Total votes in the buffer for a track. Used in debug logging.

---

### `frame_buffer.py`

**`FrameBuffer`**

Thread-safe single-frame buffer that decouples the ANPR camera loop from the FastAPI MJPEG stream endpoint. Holds only the most recent annotated frame — the stream always reads fresh footage and never serves stale frames from a queue backlog.

Synchronisation primitive: `threading.Condition(threading.Lock())`.

**`write(frame)`**
Called by `livecam.py` after every `cap.read()`. Stores a copy of the frame and calls `notify_all()` to wake waiting stream readers.

**`read() → np.ndarray | None`**
Non-blocking read. Returns a copy of the latest frame, or `None` if no frame has been written yet.

**`wait_for_frame(timeout=1.0) → np.ndarray | None`**
Blocks until a new frame is written (via `Condition.wait()`). Returns `None` on timeout. The 1-second timeout prevents the stream generator from hanging forever if the ANPR loop exits.

**`shared_buffer`**
Module-level singleton. Both `livecam.py` and `stream_route.py` import this same instance. Python's module system guarantees a single instance per process.

---

### `pipeline.py`

OCR worker thread and queue management. Separates the slow EasyOCR work from the real-time display loop.

**Queues**

| Queue | Direction | maxsize | Drop policy |
|---|---|---|---|
| `ocr_queue` | main → worker | 2 | New crop dropped if full (correct for real-time) |
| `result_queue` | worker → main | 10 | Result dropped silently if full (stream is behind) |

**`OCRWorker`**

Daemon thread that runs EasyOCR in the background.

Constructor: `OCRWorker(ocr_fn, logger, debug=False)`

- `ocr_fn`: callable — `run_ocr(crop) → (text, conf)` from `yoloconfig.py`.
- `logger`: any object with a `.log(plate, conf, snap)` method (currently a `_FileLogger` stub in `livecam.py`).

**`_process(item)` — core worker logic**

```
item = (plate_crop, context_crop, box_coords, track_id)

1. Call ocr_fn(plate_crop) → (text, conf)
2. Discard if conf < OCR_CONF_MIN (0.30)
3. VoteBuffer.add_vote(track_id, text)
4. If no consensus yet → return (update overlay only on cooldown skip)
5. Cooldown check: skip log + snapshot if same plate seen < PLATE_COOLDOWN_S ago
6. Save context_crop as JPEG to snapshots/
7. POST multipart to LOCAL_API_URL (skip silently on ConnectionError)
8. POST multipart to REMOTE_API_URL
9. Push (text, conf, box_coords) to result_queue for overlay display
```

**`notify_track_removed(track_id)`**
Clears the `VoteBuffer` for a dropped track. Should be called from `livecam.py` when `CentroidTracker._deregister()` fires (currently not wired — see Future Roadmap).

**`enqueue_crop(plate_crop, context_crop, box_coords, track_id) → bool`**
Non-blocking put to `ocr_queue`. Returns `True` if accepted, `False` if dropped (queue full).

**`drain_results() → list[(text, conf, box_coords)]`**
Drains all pending results from `result_queue`. Call once per display frame in `livecam.py`.

---

### `livecam.py`

**`run(camera_index=0, model_path="models/best.pt", debug=False)`**

Main real-time ANPR loop. Manages the camera, YOLO inference, tracking, OCR handoff, overlay drawing, and frame publishing.

**Per-frame flow**

```python
ret, frame = cap.read()

if frame_count % YOLO_EVERY_N == 0:
    small = resize(frame, YOLO_INPUT_W)
    raw_boxes = yolo(small)               # inference
    active_boxes = tracker.update(...)    # stable IDs

    for track_id, box in active_boxes:
        if throttle OK:
            plate_crop, context_crop = crop(frame, box)
            enqueue_crop(plate_crop, context_crop, box, track_id)

for text, conf, box in drain_results():   # non-blocking
    overlay_text = text
    overlay_expires = now + OVERLAY_HOLD_S

# Draw yellow boxes (all tracked vehicles)
# Draw green box + text label (confirmed plate, if not expired)

shared_buffer.write(frame)   # publish to MJPEG stream
cv2.imshow("Live ANPR", frame)
```

**Overlay state machine**

The overlay renders the most recently confirmed plate for `OVERLAY_HOLD_S` seconds (default 3.0), then clears. If a new consensus arrives before expiry, the overlay refreshes immediately.

**Visual cues**

| Colour | Meaning |
|---|---|
| Yellow box | YOLO detected, OCR in progress / accumulating votes |
| Green box | Multi-frame consensus confirmed |

---

### `stream_route.py`

**FastAPI router** serving the live ANPR feed as an MJPEG stream.

**`GET /api/camera/stream`**

Returns a `StreamingResponse` with `media_type="multipart/x-mixed-replace; boundary=frame"`.

**`_generate_mjpeg()` generator**

```python
while True:
    frame = shared_buffer.wait_for_frame(timeout=1.0)
    if frame is None:
        continue   # ANPR loop not started — retry

    ok, buffer = cv2.imencode('.jpg', frame, [JPEG_QUALITY, 60])
    if not ok:
        continue

    yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
```

The 1-second timeout in `wait_for_frame` keeps the generator alive even when `livecam.py` pauses, without busy-looping.

**`start_anpr_thread(camera_index=0, debug=False)`**

Convenience function to launch `livecam.run()` in a daemon thread from the FastAPI startup lifespan, so everything starts with a single `uvicorn` command.

---

### `ocr.py`

**`run(image_path, debug=False)`**

CLI entry point for single-image ANPR. Prints structured output to stdout:

```
[OCR] Input image: car.jpg
[YOLO] Detection successful
[YOLO] Confidence : 0.92
[YOLO] Crop size  : 312x96
[OCR] Preprocessing plate image...
[OCR] Running EasyOCR...
══════════════════════════════════════
[ANPR] VALID PLATE DETECTED
Plate Number   : KL07BH1234
YOLO Confidence: 0.92
OCR Confidence : 0.87
══════════════════════════════════════
```

When `--debug` is passed, three OpenCV windows open:
- `Detected Vehicle` — full image with YOLO bounding box
- `Plate Crop (Colour)` — raw plate crop
- `Plate (Binary / OCR)` — preprocessed binary image fed to EasyOCR

**CLI usage**
```bash
python ocr.py car.jpg
python ocr.py /path/to/image.png --debug
```

---

### `main.py`

FastAPI application. All route handlers use `Depends(get_db)` for session management.

**App setup order (important)**
```python
app = FastAPI(...)                          # 1. Create app first
app.mount("/frames", StaticFiles(...))      # 2. Mount static files
app.add_middleware(CORSMiddleware, ...)     # 3. Add CORS
models.Base.metadata.create_all(...)       # 4. Create DB tables
```

CORS is configured with `allow_origins=["*"]` for development. Restrict to your dashboard origin in production.

**`POST /api/authorize`** (multipart/form-data)

1. Query `vehicles` by `plate_number`.
2. Set `status`.
3. If `snapshot` file uploaded: save to `frames/<plate>_<timestamp>.<ext>`, store relative path.
4. Insert `GateLog` row.
5. Return `{log_id, status, snapshot_path}`.

**`GET /api/logs`**
Returns all `GateLog` rows ordered by `timestamp DESC` as a list of SQLAlchemy model instances (FastAPI auto-serialises via Pydantic).

**`GET /api/logs/unauthorized`**
Same as above, filtered to `status == "UNAUTHORIZED"`.

**`GET /api/vehicles`**
Returns all `Vehicle` rows.

**`POST /api/vehicles`**
Checks for existing plate first; raises `HTTP 400` if already registered. Otherwise inserts and returns the new record.

**`DELETE /api/vehicles/{plate_number}`**
Raises `HTTP 404` if plate not found. Otherwise deletes and returns a confirmation message.

---

## 3. Data Flow Diagrams

### Detection → Log flow

```
Camera frame
    │
    ▼ every 3rd frame
YOLOv8 inference
    │
    ▼
CentroidTracker.update()
    │  stable track IDs
    ▼
enqueue_crop() ──────────────► ocr_queue (maxsize=2)
                                    │
                                    ▼  OCRWorker thread
                               run_ocr(plate_crop)
                                    │
                               VoteBuffer.add_vote()
                                    │  threshold reached?
                                    ▼
                               cooldown check
                                    │  not in cooldown?
                                    ▼
                               save JPEG snapshot
                                    │
                                    ▼
                               POST /api/authorize
                                    │
                                    ▼
                         DB: vehicles lookup
                                    │
                         INSERT gate_logs
                                    │
                         ◄──────── response
                                    │
                               result_queue
                                    │
                         ◄── drain_results() (main thread)
                                    │
                               overlay display
```

### MJPEG stream flow

```
livecam.py (main thread)              stream_route.py (FastAPI)
    │                                        │
    │  shared_buffer.write(frame)            │  shared_buffer.wait_for_frame()
    │         │                              │         │
    │     Condition                          │     Condition
    │     .notify_all()  ──────────────────► │     .wait() unblocks
    │                                        │
    │                                   cv2.imencode JPEG
    │                                        │
    │                                   yield MJPEG boundary
    │                                        │
    │                                   HTTP client (img tag)
```

---

## 4. Threading Model

The system uses three threads:

| Thread | Name | Started by | Blocks on |
|---|---|---|---|
| Main | (Python main) | `python livecam.py` | `cap.read()`, `cv2.waitKey()` |
| OCR Worker | `ocr-worker` | `OCRWorker.start()` | `ocr_queue.get(timeout=1)` |
| ANPR (optional) | `anpr-livecam` | `start_anpr_thread()` | Same as Main |

**Thread safety:**
- `shared_buffer` uses `threading.Condition` for write/read synchronisation.
- `VoteBuffer` uses a plain dict — safe under the single-writer model (one OCR worker).
- `ocr_queue` and `result_queue` are `queue.Queue` instances — inherently thread-safe.
- `last_ocr_sent` dict in `livecam.py` is only accessed by the main thread.
- `_last_logged` dict in `OCRWorker` is only accessed by the worker thread.

**No locks needed** between main and worker beyond the two queues, because state is partitioned: main owns `last_ocr_sent` and `active_boxes`; worker owns `_last_logged` and `_vote_buf`.

---

## 5. OCR Pipeline Deep-Dive

### Why two OCR passes (binary + inverted)?

Indian number plates come in two colour schemes:
- **White background, black text** — standard
- **Black background, yellow text** — commercial vehicles

Otsu thresholding on a white plate produces black characters on white (correct for EasyOCR). On a yellow plate it may produce the inverse. Running both `binary` and `cv2.bitwise_not(binary)` ensures at least one pass gives EasyOCR the representation it reads best.

### Why merge EasyOCR tokens?

EasyOCR sometimes splits a plate into two tokens: `['KL07', 'BH1234']`. Merging all tokens sorted left-to-right by their bounding box x-coordinate reconstructs the full plate string before applying `PLATE_RE`.

### Confidence aggregation

EasyOCR returns per-token confidence. The merged candidate's confidence is the mean of all token confidences. This is more reliable than taking `max` (which can be inflated by a single high-confidence short token).

### `VoteBuffer` threshold tuning

| window | threshold | Behaviour |
|---|---|---|
| 5 | 3 | Balanced — confirms after 3 of 5 reads agree (~300ms at 10 FPS) |
| 5 | 5 | Very strict — needs all 5 reads to agree (rejects more false positives, slower) |
| 3 | 2 | Fast confirmation (~200ms) — good for high-speed cameras |

Increase `VOTE_WINDOW` and `VOTE_THRESHOLD` for slower cameras or noisier conditions.

---

## 6. Indian Plate Format & Correction

### Regex

```python
PLATE_RE = re.compile(r'^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$')
```

Matches: `KL07BH1234`, `MH1AZ3739`, `KA09ABC0001` — covers all standard formats.

### Structure

```
K  L  0  7  B  H  1  2  3  4
│  │  │  │  │  │  │  │  │  │
└──┘  └──┘  └──┘  └──────────┘
State District Series   Serial
code  number   letters  number
(2L)  (1-2D)   (1-3L)   (4D)
```

### `smart_correct` logic

The function iterates through the string applying zone-specific substitutions:

```
Position 0–1       → enforce letters  (D2L map: 0→O, 1→I, 5→S, 8→B)
Position n-4 to n  → enforce digits   (L2D map: O→0, I→1, Z→2, S→5, B→8, Q→0, D→0)
Middle section     → district then series:
                     • Count up to 2 digits (L2D map)
                     • Once 2 digits seen or non-digit hit → switch to letters (D2L map)
```

---

## 7. Database Design Decisions

### No foreign key from `gate_logs` to `vehicles`

**Why:** A `gate_logs` row is created for every detection, including UNAUTHORIZED ones where the `plate_number` has no corresponding row in `vehicles`. Using a FK with `ON DELETE SET NULL` would require nullable FK columns and complicate queries. The current design stores the raw OCR string in both tables; lookup is by equality match on `plate_number`.

**Consequence:** If a vehicle is removed from the whitelist, historical `gate_logs` rows still show `AUTHORIZED` for past events. This is intentional — logs are immutable audit records.

### `confidence_score` stored as FLOAT (0.0–1.0)

EasyOCR returns confidence as a proportion. The dashboard displays it as a percentage (`× 100`). This is consistent throughout the codebase — no ambiguity about whether 92 means 0.92 or 92%.

> **Caution:** `pipeline.py` currently sets `conf_pct = conf` (not `conf * 100`) before POSTing to the API. If your dashboard is showing values like 0.87 instead of 87%, check this variable.

### Server-side timestamps

`registered_at` and `timestamp` both use `server_default=func.now()`. This ensures timestamps are set by the PostgreSQL server clock, not the Python process clock — important if the Python host clock drifts or has a different timezone setting.

---

## 8. Configuration Reference

### `livecam.py`

```python
YOLO_CONF_MIN   = 0.40   # Minimum YOLO detection confidence
OCR_THROTTLE_S  = 1.0    # Seconds between OCR submissions per track
OVERLAY_HOLD_S  = 3.0    # Seconds confirmed plate stays on screen
YOLO_INPUT_W    = 640    # Width to resize frame before YOLO
YOLO_EVERY_N    = 3      # Run YOLO every N frames (tracker fills gaps)
CONTEXT_PAD     = 40     # Extra px around plate for snapshot context crop
```

### `pipeline.py`

```python
OCR_CONF_MIN     = 0.30         # Minimum OCR confidence to enter vote buffer
PLATE_COOLDOWN_S = 5.0          # Seconds between DB logs for same plate
VOTE_WINDOW      = 5            # VoteBuffer sliding window size
VOTE_THRESHOLD   = 2            # Votes needed to confirm plate
SNAPSHOT_DIR     = Path("snapshots")
LOCAL_API_URL    = "http://127.0.0.1:8000/api/authorize"
REMOTE_API_URL   = "http://10.241.37.137:8000/api/authorize"  # ← update this
```

### `yoloconfig.py`

```python
MODEL_PATH     = "models/best.pt"   # Path to YOLOv8 .pt weights
_MIN_OCR_WIDTH = 200                # Minimum plate crop width before upscale
```

### `tracker.py`

```python
max_distance = 60    # px — tune to typical plate width × 0.15
max_missing  = 10    # frames — increase for slower cameras or occlusions
```

---

## 9. Error Handling Strategy

| Location | Failure | Handling |
|---|---|---|
| `auth.py` | Any DB exception | Rollback, print, return `None` |
| `pipeline.py _process` | Any unhandled exception | Caught in `_run` loop, printed, worker continues |
| `pipeline.py LOCAL_API_URL` | `ConnectionError` | Silently skipped (`pass`) |
| `pipeline.py REMOTE_API_URL` | Any exception | Printed, execution continues |
| `livecam.py cap.read()` | Failed frame | Print warning, `time.sleep(0.05)`, retry |
| `livecam.py enqueue_crop` | Queue full | `False` returned, crop dropped (correct real-time policy) |
| `stream_route._generate_mjpeg` | No frame (timeout) | Generator loops and retries — stream stays open |
| `stream_route cv2.imencode` | Encode failure | `continue` — frame skipped |
| `main.py endpoints` | DB errors | Let FastAPI return HTTP 500 (unhandled) |
| `main.py POST /api/vehicles` | Duplicate plate | `HTTPException(400)` |
| `main.py DELETE /api/vehicles` | Plate not found | `HTTPException(404)` |

---

## 10. Frontend Architecture

`frontend/index.html` is a standalone single-file dashboard. It has no build step, no npm, no framework — only vanilla HTML, CSS, and JavaScript.

### Page structure

The dashboard uses a custom tab system. Each tab corresponds to a `<div class="page" id="page-{name}">`. `showPage(id, el)` toggles the active class and calls the appropriate data loader.

```javascript
function showPage(id, el) {
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
    document.getElementById('page-' + id).classList.add('active');
    el.classList.add('active');
    if (id === 'logs') loadLogs();
    if (id === 'alerts') renderAlerts();
    if (id === 'whitelist') renderWhitelist();
}
```

### Data polling

The dashboard polls the API on a 2-second interval:

```javascript
setInterval(loadLogs, 2000);
```

`loadLogs()` fetches `GET /api/logs`, updates the four stat cards (total, authorised, denied, registered), and re-renders the log table if the Logs page is active.

### API base URL

```javascript
const API = "http://127.0.0.1:8000";
```

Change this constant at the top of the `<script>` block if your server runs on a different host or port.

### CSS design system

The dashboard uses a Slate + Indigo light theme defined in `:root` CSS variables:

| Variable | Use |
|---|---|
| `--accent` / `--accent2` | Indigo primary (`#4f46e5` / `#6366f1`) |
| `--green` / `--green2` / `--green3` | Authorised status colours |
| `--red` / `--red2` / `--red3` | Unauthorized / alert colours |
| `--amber` | Warning / pending |
| `--mono` | `Space Mono` — plates, code, IDs |
| `--sans` | `Inter` — body text |

### Gate simulation

Gates are simulated client-side only (no backend call on toggle). The `gateStates` object tracks open/closed status locally. In production, `toggleGate()` should call `POST /api/gates/{id}/control` which publishes the MQTT command to the ESP32.

### Camera simulation

`updateCameras()` cycles through a hardcoded list of test plate strings every 2 seconds to simulate live detections. Replace this with an `<img src="/api/camera/stream">` tag pointing at the FastAPI MJPEG endpoint for real camera output.

---

*GAYTS Technical Documentation — generated from source*
