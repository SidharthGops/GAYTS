# GAYTS — Gate Automated YOLO Tracking System

> Real-time Automatic Number Plate Recognition (ANPR) gate control system for Indian vehicles.  
> Detects licence plates via YOLOv8 + EasyOCR, checks them against a whitelist, and triggers a physical gate via MQTT/ESP32.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the System](#running-the-system)
- [API Reference](#api-reference)
- [Dashboard](#dashboard)
- [Hardware Setup](#hardware-setup)
- [Module Descriptions](#module-descriptions)
- [Database Schema](#database-schema)
- [Testing](#testing)
- [Future Roadmap](#future-roadmap)

---

## Overview

GAYTS is a complete end-to-end gate security system built around computer vision. A camera feed is continuously processed by a two-stage YOLOv8 pipeline (vehicle detection → plate localisation), followed by EasyOCR for text extraction. Detected plates are checked against a PostgreSQL whitelist in under 10 ms, and an ESP32 microcontroller opens or closes the gate barrier over MQTT. Every detection — authorised or not — is logged with a confidence score and an optional JPEG snapshot.

A lightweight HTML dashboard (no framework dependencies) provides live camera preview, gate controls, log browsing, whitelist management, and API documentation.

---

## Features

| Category | Detail |
|---|---|
| **AI Pipeline** | YOLOv8 vehicle detect → YOLOv8 plate localise → CLAHE preprocessing → EasyOCR + Tesseract fallback |
| **OCR Reliability** | Multi-frame consensus voting (`VoteBuffer`) — 5-frame window, configurable threshold |
| **Authorization** | Whitelist-based (PostgreSQL `vehicles` table). All detections logged regardless of status |
| **Gate Control** | MQTT command to ESP32 → servo/relay opens boom barrier, auto-closes after timeout |
| **Live Stream** | MJPEG stream served over FastAPI — no second camera open, zero conflict |
| **Dashboard** | Single-file HTML UI: live feed, gate simulation, log table, alerts, whitelist manager, DB schema, API docs, implementation guide |
| **Snapshot Storage** | JPEG of context crop saved on every confirmed unique plate (configurable cooldown) |
| **Indian Plate Support** | Regex validation + `smart_correct()` fixes common OCR confusions (0↔O, 1↔I, etc.) |

---

## Architecture

```
Camera (USB / RTSP)
       │
       ▼
┌─────────────────────────────────────────┐
│  livecam.py  (Main Thread)              │
│  cap.read() → YOLO (every 3 frames)     │
│  → CentroidTracker (stable IDs)         │
│  → enqueue_crop() → OCR Worker Thread  │
│  → shared_buffer.write(annotated frame) │
└───────────────────┬─────────────────────┘
                    │ result_queue
                    ▼
┌─────────────────────────────────────────┐
│  pipeline.py  (Daemon Thread)           │
│  EasyOCR → VoteBuffer consensus         │
│  → snapshot save → POST /api/authorize  │
└───────────────────┬─────────────────────┘
                    │ HTTP
                    ▼
┌─────────────────────────────────────────┐
│  FastAPI  (main.py)                     │
│  /api/authorize → PostgreSQL lookup     │
│  → INSERT gate_logs                     │
│  /api/camera/stream ← shared_buffer     │
└───────────────────┬─────────────────────┘
                    │ MQTT
                    ▼
             ESP32 / Arduino
             Servo boom barrier
```

**End-to-end latency target: ~350 ms**

| Stage | Budget |
|---|---|
| Frame capture → YOLO detection | ~80 ms |
| Plate localisation | ~60 ms |
| OCR extraction | ~120 ms |
| DB authorisation lookup | ~8 ms |
| MQTT gate trigger | ~15 ms |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | Python 3.11 · FastAPI · Uvicorn |
| ORM | SQLAlchemy 2.0 |
| Database | PostgreSQL 17 |
| DB driver | psycopg2-binary |
| Vehicle detection | YOLOv8n (Ultralytics) |
| Plate localisation | YOLOv8 fine-tuned on Indian plates |
| OCR (primary) | EasyOCR |
| OCR (fallback) | Tesseract 5 |
| Computer vision | OpenCV 4 |
| Gate controller | ESP32 · MQTT · PubSubClient |
| Env config | python-dotenv |
| Frontend | Vanilla HTML + CSS + JS (zero dependencies) |

---

## Project Structure

```
GAYTS/
├── .env                  # DATABASE_URL (not committed)
├── requirements.txt
│
├── main.py               # FastAPI app — all REST endpoints
├── db.py                 # SQLAlchemy engine + session factory
├── models.py             # Vehicle + GateLog ORM models
├── auth.py               # check_and_log() — standalone auth helper
│
├── livecam.py            # Real-time ANPR main loop
├── pipeline.py           # OCR worker thread + queue management
├── yoloconfig.py         # YOLO singleton, EasyOCR singleton, preprocessing, OCR runner
├── tracker.py            # CentroidTracker — stable cross-frame IDs
├── consensus.py          # VoteBuffer — multi-frame OCR majority vote
├── frame_buffer.py       # Thread-safe single-frame buffer (livecam ↔ stream)
├── stream_route.py       # FastAPI MJPEG stream endpoint
├── ocr.py                # Single-image ANPR CLI tool
│
├── test_auth.py          # Auth function smoke tests
│
├── models/
│   └── best.pt           # YOLOv8 plate detection weights
│
├── snapshots/            # Saved plate JPEG crops (auto-created)
├── frames/               # Snapshot uploads served as static files
│
└── frontend/
    └── index.html        # Admin dashboard (standalone, no build step)
```

---

## Prerequisites

- Python 3.10+
- PostgreSQL 17 running locally (or remote, set via `DATABASE_URL`)
- Tesseract 5 installed on the OS (`sudo apt install tesseract-ocr` / `brew install tesseract`)
- A webcam (index `0`) or an IP camera with an RTSP URL
- CUDA-capable GPU (optional but recommended for real-time performance)
- Arduino IDE + ESP32 board package (for hardware gate control)

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/SidharthGops/GAYTS.git
cd GAYTS

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Create the .env file
echo DATABASE_URL=postgresql://USER:PASSWORD@localhost:5432/gayts > .env

# 5. Create the database
psql -U postgres -c "CREATE DATABASE gayts;"

# 6. Run migrations (SQLAlchemy auto-create)
python -c "from db import engine; from models import Base; Base.metadata.create_all(bind=engine)"

# 7. Download YOLOv8 base weights (first run downloads automatically)
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

**requirements.txt (minimum)**
```
fastapi
uvicorn[standard]
sqlalchemy
psycopg2-binary
python-dotenv
ultralytics
easyocr
opencv-python
pytesseract
python-multipart
requests
```

---

## Configuration

All tuneable constants live at the top of their respective modules. The most important ones:

### `livecam.py`
| Constant | Default | Description |
|---|---|---|
| `YOLO_CONF_MIN` | `0.40` | Minimum YOLO box confidence to process |
| `OCR_THROTTLE_S` | `1.0` | Seconds between OCR jobs per tracked vehicle |
| `OVERLAY_HOLD_S` | `3.0` | How long a confirmed plate label stays on screen |
| `YOLO_EVERY_N` | `3` | Run YOLO inference every N frames |
| `CONTEXT_PAD` | `40` | Extra pixel padding around plate for snapshot |

### `pipeline.py`
| Constant | Default | Description |
|---|---|---|
| `OCR_CONF_MIN` | `0.30` | Minimum EasyOCR confidence to pass to vote buffer |
| `PLATE_COOLDOWN_S` | `5.0` | Minimum seconds between DB logs for the same plate |
| `VOTE_WINDOW` | `5` | Number of recent OCR reads per track for consensus |
| `VOTE_THRESHOLD` | `2` | Votes needed for consensus |
| `LOCAL_API_URL` | `http://127.0.0.1:8000/api/authorize` | Local FastAPI server |
| `REMOTE_API_URL` | `http://10.241.37.137:8000/api/authorize` | Remote server (update to your IP) |

### `tracker.py`
| Constant | Default | Description |
|---|---|---|
| `max_distance` | `60` px | Max centroid shift to be treated as same vehicle |
| `max_missing` | `10` frames | Frames without detection before track is dropped |

---

## Running the System

### Start the API server
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Start the live ANPR camera loop (separate terminal)
```bash
python livecam.py
# Press Q in the OpenCV window to stop
```

### Open the dashboard
Open `frontend/index.html` directly in a browser. The dashboard polls `http://localhost:8000` by default. Change the `API` constant at the top of the `<script>` block if your server is on a different host.

### Single-image test (no camera required)
```bash
python ocr.py car.jpg
python ocr.py car.jpg --debug    # opens OpenCV windows showing detection + crop
```

### Run as a single Uvicorn process (API + camera thread)
Add the lifespan helper from `stream_route.py` to `main.py`:
```python
from contextlib import asynccontextmanager
from stream_route import router as stream_router, start_anpr_thread

@asynccontextmanager
async def lifespan(app):
    start_anpr_thread(camera_index=0, debug=False)
    yield

app = FastAPI(title="LPR Gate System API", lifespan=lifespan)
app.include_router(stream_router)
```

---

## API Reference

Base URL: `http://localhost:8000`

### Authorization

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/authorize` | Core pipeline endpoint — logs detection, returns AUTHORIZED / UNAUTHORIZED |

**Request** (multipart/form-data)
```
plate_number     string   required
confidence_score float    optional (default 0)
snapshot         file     optional JPEG upload
```

**Response**
```json
{
  "log_id": 145,
  "status": "AUTHORIZED",
  "snapshot_path": "frames/KL07BH1234_20250519_103422.jpg"
}
```

---

### Vehicles (Whitelist)

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/vehicles` | List all whitelisted plates |
| `POST` | `/api/vehicles` | Add a plate to the whitelist |
| `DELETE` | `/api/vehicles/{plate_number}` | Remove a plate |

**POST body (JSON)**
```json
{ "plate_number": "KL07BH1234" }
```

---

### Gate Logs

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/logs` | All gate logs, newest first |
| `GET` | `/api/logs/unauthorized` | Only UNAUTHORIZED entries |

---

### Camera Stream

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/camera/stream` | MJPEG live stream with YOLO + plate overlays |

Embed in HTML:
```html
<img src="http://SERVER_IP:8000/api/camera/stream">
```

---

## Dashboard

`frontend/index.html` is a zero-dependency single-file admin panel with nine tabs:

| Tab | Contents |
|---|---|
| **Dashboard** | Stat cards (total / authorised / denied / registered), live camera simulation, gate controls, manual plate lookup, live detection feed, hourly traffic chart |
| **Gate Logs** | Searchable/filterable table with plate, status, confidence bar, snapshot thumbnail; CSV export |
| **Alerts** | Active unauthorised detections, alert rule table, 24-hour alert history |
| **Whitelist** | Add / remove plates with instant feedback |
| **DB Schema** | Visual schema cards for both tables, relationship diagram, sample SQL queries |
| **API Docs** | Collapsible endpoint cards with request/response examples |
| **AI Pipeline** | Step-by-step pipeline diagram, performance budget bars, hardware table, ESP32 code snippet |
| **Project** | Folder tree, full tech stack table, future enhancement cards |
| **Guide** | Step-by-step implementation checklist, demo day script, camera YAML config, dataset recommendations |

---

## Hardware Setup

### Components
| Component | Role | Interface |
|---|---|---|
| Webcam / IP Camera | Video capture | USB (index 0) / RTSP URL |
| ESP32 | Gate relay control | WiFi + MQTT |
| Servo / DC Motor | Boom barrier actuator | PWM (GPIO 18) |
| IR Sensor | Vehicle presence detection | GPIO 34 (digital in) |
| Host laptop / Raspberry Pi 4 | AI inference + API server | Local network |

### ESP32 Wiring
```
ESP32 GPIO 18  →  Servo signal wire (yellow)
5V             →  Servo power (red)
GND            →  Servo ground (black)
ESP32 GPIO 34  →  IR sensor OUT (pull-up enabled)
```

### MQTT Commands
```bash
# Open gate (holds 8 seconds, then auto-closes)
mosquitto_pub -t gate/1/command -m OPEN

# Force close immediately
mosquitto_pub -t gate/1/command -m FORCE_CLOSE
```

Upload `gate_controller.ino` via Arduino IDE. Set your WiFi SSID, password, and MQTT broker IP in the sketch before flashing.

---

## Module Descriptions

| File | Purpose |
|---|---|
| `main.py` | FastAPI application. Defines all REST endpoints. Handles multipart snapshot uploads, static file serving from `frames/`, and CORS. |
| `db.py` | Creates the SQLAlchemy engine from `DATABASE_URL`, provides `SessionLocal` and the `Base` declarative class. |
| `models.py` | ORM models: `Vehicle` (whitelist) and `GateLog` (detection log). Server-side timestamps via `func.now()`. |
| `auth.py` | `check_and_log()` — standalone helper that checks a plate against the vehicles table and inserts a gate_log row. Used for testing and can be called directly from other scripts. |
| `livecam.py` | Main real-time loop. Reads frames from OpenCV, runs YOLO every N frames, tracks detections with `CentroidTracker`, enqueues crops to the OCR worker, draws overlays, writes to `shared_buffer`. |
| `pipeline.py` | Background OCR worker thread. Pulls crops from `ocr_queue`, runs EasyOCR, applies `VoteBuffer` consensus, saves snapshots, POSTs to `/api/authorize`, pushes results to `result_queue`. |
| `yoloconfig.py` | Singletons for YOLO model and EasyOCR reader. `preprocess_plate()` for image enhancement. `smart_correct()` for OCR error correction. `run_ocr()` combining binary + inverted pass. |
| `tracker.py` | `CentroidTracker` — assigns stable integer IDs to YOLO bounding boxes across frames using nearest-centroid matching. Prevents the OCR throttle from resetting on every pixel shift. |
| `consensus.py` | `VoteBuffer` — sliding-window majority vote per track ID. Eliminates single-frame OCR noise. Returns consensus only when N of M recent reads agree. |
| `frame_buffer.py` | Thread-safe single-frame buffer shared between `livecam.py` (writer) and the MJPEG stream endpoint (reader). Always serves the latest frame; never queues stale footage. |
| `stream_route.py` | FastAPI `APIRouter` serving `/api/camera/stream` as an MJPEG response. Reads frames from `shared_buffer`. Includes `start_anpr_thread()` for launching `livecam` from the FastAPI lifespan. |
| `ocr.py` | CLI tool for single-image ANPR. Runs the full detect → preprocess → OCR pipeline on a JPEG/PNG file. Supports `--debug` flag to show OpenCV windows. |
| `test_auth.py` | Smoke tests for `check_and_log()`. Runs three test plates (known, unknown, second known) and prints results. Verify in pgAdmin that rows appear in `gate_logs`. |

---

## Database Schema

### `vehicles`
| Column | Type | Notes |
|---|---|---|
| `id` | SERIAL | Primary key |
| `plate_number` | VARCHAR(20) | Unique, uppercase |
| `registered_at` | TIMESTAMP | Auto-set on insert |

### `gate_logs`
| Column | Type | Notes |
|---|---|---|
| `id` | SERIAL | Primary key |
| `plate_number` | VARCHAR(20) | Raw OCR output |
| `status` | VARCHAR(15) | `AUTHORIZED` or `UNAUTHORIZED` |
| `confidence_score` | FLOAT | EasyOCR confidence 0.0–1.0 |
| `snapshot_path` | TEXT | Relative path to saved JPEG |
| `timestamp` | TIMESTAMP | Auto-set on insert |

**Design decision:** No foreign key from `gate_logs` to `vehicles`. Unauthorised plates (not in `vehicles`) are still fully logged without null FK violations.

### Useful queries
```sql
-- All logs newest first
SELECT * FROM gate_logs ORDER BY timestamp DESC;

-- Unauthorised attempts only
SELECT * FROM gate_logs WHERE status = 'UNAUTHORIZED' ORDER BY timestamp DESC;

-- Daily summary
SELECT status, COUNT(*) FROM gate_logs
WHERE DATE(timestamp) = CURRENT_DATE
GROUP BY status;

-- Top detected plates
SELECT plate_number, COUNT(*) AS hits
FROM gate_logs GROUP BY plate_number
ORDER BY hits DESC LIMIT 10;
```

---

## Testing

```bash
# Smoke test the auth helper against your live database
python test_auth.py

# Test single-image OCR pipeline
python ocr.py test.jpg
python ocr.py test.jpg --debug

# Test the API with curl
curl -X POST http://localhost:8000/api/vehicles \
  -H "Content-Type: application/json" \
  -d '{"plate_number": "KL07BH1234"}'

curl http://localhost:8000/api/vehicles
curl http://localhost:8000/api/logs
```

---

## Future Roadmap

| Feature | Description |
|---|---|
| Face Recognition | Dual-auth: plate + driver face via InsightFace |
| Visitor QR Passes | Time-limited temporary access codes |
| Parking Management | Slot tracking and live occupancy display |
| Cloud Dashboard | Multi-site analytics via BigQuery |
| Mobile App | React Native guard patrol interface |
| Anomaly Detection | Isolation Forest on access patterns (repeated attempts, after-hours) |
| Multi-camera Support | Camera config via `cameras.yaml`, parallel pipeline instances |

---

## License

This project is developed as a student project. See `LICENSE` for details.

---

*Built with YOLOv8 · EasyOCR · FastAPI · PostgreSQL · ESP32*
