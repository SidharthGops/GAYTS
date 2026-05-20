"""
main.py
────────
Entry point for the ANPR project.

Usage
─────
Single image:
    python main.py --image car.jpg

Live webcam (default camera):
    python main.py --live

Live webcam (specific camera + debug output):
    python main.py --live --camera 1 --debug

Custom model:
    python main.py --live --model models/custom.pt
"""

import argparse
import sys
import ocr
import livecam
def main() -> None:
    parser = argparse.ArgumentParser(
        description="ANPR System — YOLOv8 + EasyOCR"
    )
    # ── Mode (mutually exclusive, one required) ───────────────────────────────
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--image",
        type=str,
        metavar="PATH",
        help="Path to a single image file",
    )
    mode.add_argument(
        "--live",
        action="store_true",
        help="Run live webcam ANPR",
    )
    # ── Shared options ────────────────────────────────────────────────────────
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        metavar="INDEX",
        help="Camera device index for --live mode (default: 0)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/best.pt",
        metavar="PATH",
        help="Path to YOLOv8 weights file (default: models/best.pt)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose OCR output and debug overlays",
    )
    args = parser.parse_args()
    # ── Image mode ────────────────────────────────────────────────────────────
    if args.image:
        print("\n[MAIN] Running single-image ANPR...")
        try:
            ocr.run(args.image, debug=args.debug)
        except KeyboardInterrupt:
            print("\n[MAIN] Interrupted.")
            sys.exit(0)
        except Exception as e:
            print(f"\n[MAIN] Error: {e}")
            if args.debug:
                raise
            sys.exit(1)
    # ── Live mode ─────────────────────────────────────────────────────────────
    elif args.live:
        print(
            f"\n[MAIN] Starting live ANPR  "
            f"camera={args.camera}  model={args.model}  debug={args.debug}"
        )
        try:
            livecam.run(
                camera_index = args.camera,
                model_path   = args.model,
                debug        = args.debug,
            )
        except KeyboardInterrupt:
            print("\n[MAIN] Interrupted.")
            sys.exit(0)
        except Exception as e:
            print(f"\n[MAIN] Error: {e}")
            if args.debug:
                raise
            sys.exit(1)


if __name__ == "__main__":
    main()