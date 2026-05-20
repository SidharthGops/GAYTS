"""
yoloconfig.py
─────────────
Shared utility module for the ANPR project.
Imported by both ocr.py (single-image) and livecam.py (live camera).

Provides
  • YOLO plate detector
  • EasyOCR reader (singleton — loaded once, reused everywhere)
  • Image preprocessing pipeline
  • OCR correction and runner
"""

import cv2
import re
import numpy as np
import easyocr
from ultralytics import YOLO
from pathlib import Path

# ── Singletons ────────────────────────────────────────────────────────────────

_yolo_model : YOLO           | None = None
_ocr_reader : easyocr.Reader | None = None
MODEL_PATH   = "models/best.pt"
SNAPSHOT_DIR = Path("snapshots")
def get_model() -> YOLO:
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO(MODEL_PATH)
    return _yolo_model

def get_reader() -> easyocr.Reader:
    global _ocr_reader
    if _ocr_reader is None:
        _ocr_reader = easyocr.Reader(['en'], gpu=False)
    return _ocr_reader

# ── Indian plate regex ────────────────────────────────────────────────────────
# Format:  SS  DD  LL  NNNN   e.g.  KL 10 AZ 3739

PLATE_RE = re.compile(r'^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$')

# ── YOLO detection (single-image mode) ───────────────────────────────────────

def detect_plate(image_path: str) -> tuple:
    """
    Run YOLO on a single image file and return the largest detected plate.

    Returns
    ───────
    (plate_crop, annotated_image, yolo_confidence)
    Returns (None, None, 0.0) if no plate is found.
    """
    model = get_model()
    img   = cv2.imread(image_path)

    if img is None:
        print(f"[YOLO] Image not found: {image_path}")
        return None, None, 0.0

    results   = model(img)
    best_crop = None
    best_conf = 0.0
    best_area = 0
    best_box  = None
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best_conf = conf
                pad = 10
                px1 = max(0, x1 - pad);  px2 = min(img.shape[1], x2 + pad)
                py1 = max(0, y1 - pad);  py2 = min(img.shape[0], y2 + pad)
                best_crop = img[py1:py2, px1:px2]
                best_box  = (px1, py1, px2, py2)
    if best_box:
        x1, y1, x2, y2 = best_box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        print(f"[YOLO] Plate detected  conf={best_conf:.2f}  area={best_area}px²")
    else:
        print("[YOLO] No plate found in image.")
    return best_crop, img, best_conf
# ── Preprocessing ─────────────────────────────────────────────────────────────
# Minimum width before we bother upscaling.
# Crops wider than this are already large enough for EasyOCR.
_MIN_OCR_WIDTH = 200
def preprocess_plate(plate_bgr: np.ndarray) -> np.ndarray:
    """
    BGR plate crop → clean binary image ready for EasyOCR.

    Pipeline: conditional upscale → CLAHE → Gaussian blur → Otsu threshold

    Upscale is conditional: only applied when the crop is narrower than
    _MIN_OCR_WIDTH pixels. A 300px crop fed at 3× becomes 900px with no
    OCR benefit and 3× the compute cost. A 60px crop needs the upscale.

    Why not sharpen + bilateral?
    At high scale factors the bilateral filter smears thin character strokes;
    the sharpening kernel then amplifies those merge artefacts.
    CLAHE + Otsu gives crisp black/white characters without distortion.
    """
    crop_w = plate_bgr.shape[1]
    if crop_w < _MIN_OCR_WIDTH:
        scale = _MIN_OCR_WIDTH / crop_w   # e.g. 60px → 3.3×, 150px → 1.3×
        img = cv2.resize(
            plate_bgr, None,
            fx=scale, fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )
    else:
        img = plate_bgr   # already large enough — no copy needed
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)
    gray  = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary
# ── OCR correction ────────────────────────────────────────────────────────────
def smart_correct(s: str) -> str:
    """
    Fix common OCR character confusions using the known structure of
    Indian plates.

    Plate structure
    ───────────────
      idx 0–1      → state code     always letters    KL
      idx 2–(n-5)  → district + series                10AZ
        • district → digits first (max 2)
        • series   → letters after
      idx (n-4)–   → serial number  always digits     3739

    Correction maps
    ───────────────
      L2D  letter lookalikes → digit  (used in digit zones)
      D2L  digit lookalikes  → letter (used in letter zones)

    Known OCR confusions covered
    ────────────────────────────
      O/Q → 0   I → 1   Z → 2   S → 5   B → 8   (letter→digit)
      0   → O   1 → I   5 → S   8 → B            (digit→letter)
    """
    # Q/D added: EasyOCR regularly reads '0' as 'Q' or 'D' in digit positions
    L2D = {"O": "0", "Q": "0", "D": "0", "I": "1", "Z": "2", "S": "5", "B": "8"}
    D2L = {"0": "O", "1": "I", "5": "S", "8": "B"}

    # Already valid — trust it, skip correction
    if PLATE_RE.match(s):
        return s

    if not (8 <= len(s) <= 11):
        return s

    n = len(s)
    c = list(s)

    # Positions 0–1: always state letters
    for i in (0, 1):
        c[i] = D2L.get(c[i], c[i])

    # Last 4: always serial digits
    for i in range(n - 4, n):
        c[i] = L2D.get(c[i], c[i])

    # Middle: district digits first (max 2), then series letters
    mid        = c[2 : n - 4]
    digit_done = False
    d_count    = 0

    for j, ch in enumerate(mid):
        if not digit_done and d_count < 2:
            as_digit = L2D.get(ch, ch)
            if as_digit.isdigit():
                mid[j]   = as_digit
                d_count += 1
                if d_count == 2:
                    digit_done = True
            else:
                digit_done = True
                mid[j]    = D2L.get(ch, ch)
        else:
            digit_done = True
            mid[j]     = D2L.get(ch, ch)

    c[2 : n - 4] = mid
    return "".join(c)


# ── OCR runner ────────────────────────────────────────────────────────────────

def run_ocr(plate_bgr: np.ndarray) -> tuple[str, float]:
    """
    Run OCR on a BGR plate crop and return the best valid result.

    Strategy
    ────────
    1. Preprocess to binary (conditional upscale + CLAHE + Otsu).
    2. Try binary AND inverted — handles white plates and old black plates.
    3. Merge EasyOCR tokens left-to-right (prevents split reads).
    4. Apply smart_correct, validate against PLATE_RE.
    5. Return the candidate with the highest average confidence.

    Returns ('', 0.0) if no valid plate is found.
    """
    reader   = get_reader()
    binary   = preprocess_plate(plate_bgr)
    inverted = cv2.bitwise_not(binary)

    best_text, best_conf = "", 0.0

    for img in (binary, inverted):
        raw = reader.readtext(
            img,
            detail=1,
            paragraph=False,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            text_threshold=0.3,
            width_ths=0.9,
        )
        if not raw:
            continue

        raw_sorted = sorted(raw, key=lambda r: r[0][0][0])
        merged     = "".join(re.sub(r'[^A-Z0-9]', '', r[1].upper()) for r in raw_sorted)
        conf       = sum(r[2] for r in raw_sorted) / len(raw_sorted)

        cleaned = smart_correct(merged)

        if PLATE_RE.match(cleaned) and conf > best_conf:
            best_text, best_conf = cleaned, conf

    return best_text, best_conf