"""
ocr.py — Single-image ANPR.
Called by:
    python main.py --image car.jpg

or directly:
    python ocr.py car.jpg
    python ocr.py car.jpg --debug
"""

import cv2
from yoloconfig import (
    detect_plate,
    run_ocr,
    preprocess_plate,
)
def run(image_path: str, debug: bool = False) -> None:
    print("\n" + "=" * 55)
    print("[OCR] Starting ANPR image pipeline")
    print("=" * 55)
    print(f"\n[OCR] Input image: {image_path}")
    # ── YOLO Detection ────────────────────────────────────────────────────────
    print("\n[YOLO] Running plate detection...")
    plate_crop, full_img, yolo_conf = detect_plate(image_path)
    if plate_crop is None:
        print("\n[YOLO] No plate detected.")
        return

    print(f"[YOLO] Detection successful")
    print(f"[YOLO] Confidence : {yolo_conf:.2f}")
    print(f"[YOLO] Crop size  : {plate_crop.shape[1]}x{plate_crop.shape[0]}")
    # ── OCR ───────────────────────────────────────────────────────────────────
    print("\n[OCR] Preprocessing plate image...")
    binary = preprocess_plate(plate_crop)   # kept for debug window only
    print("[OCR] Running EasyOCR...")
    plate_text, ocr_conf = run_ocr(plate_crop)
    print("[OCR] OCR completed")
    # ── Results ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 55)
    if plate_text:
        print("[ANPR] VALID PLATE DETECTED")
        print(f"Plate Number   : {plate_text}")
        print(f"YOLO Confidence: {yolo_conf:.2f}")
        print(f"OCR Confidence : {ocr_conf:.2f}")
    else:
        print("[ANPR] No valid plate extracted")

    print("═" * 55 + "\n")

    # ── Debug windows (only when --debug is passed) ───────────────────────────
    if debug:
        print("[DEBUG] Opening visualization windows...")
        print("[DEBUG] Press any key in any image window to exit.")
        cv2.imshow("Detected Vehicle",     full_img)
        cv2.imshow("Plate Crop (Colour)",  plate_crop)
        cv2.imshow("Plate (Binary / OCR)", binary)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("[OCR] Pipeline finished.\n")


# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Single-image ANPR")
    parser.add_argument("image", nargs="?", default="car.jpg", help="Path to image")
    parser.add_argument("--debug", action="store_true", help="Show OpenCV windows")
    args = parser.parse_args()

    run(args.image, debug=args.debug)