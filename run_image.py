#!/usr/bin/env python3
"""
Smart Road Lane Line Detection System — Single Image Processing CLI
====================================================================
Process a single image through the lane detection pipeline.

Usage:
    python run_image.py --input path/to/image.jpg
    python run_image.py --input path/to/image.jpg --output output/images/ --no-display
"""

import os
import sys
import argparse
import time

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
from core.pipeline import LaneDetectionPipeline


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Smart Road Lane Line Detection — Image Processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_image.py --input data/sample/road.jpg
  python run_image.py --input photo.png --output results/ --no-display
        """,
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the input image file (jpg, png, bmp, tiff)",
    )
    parser.add_argument(
        "--output", "-o",
        default=os.path.join("output", "images"),
        help="Directory to save the processed image (default: output/images/)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Suppress the OpenCV display window",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Validate input ─────────────────────────────────────────────
    if not os.path.isfile(args.input):
        print(f"[ERROR] Input file not found: {args.input}")
        sys.exit(1)

    image = cv2.imread(args.input)
    if image is None:
        print(f"[ERROR] Could not read image: {args.input}")
        sys.exit(1)

    # ── Ensure output directory exists ─────────────────────────────
    os.makedirs(args.output, exist_ok=True)

    # ── Initialize pipeline ────────────────────────────────────────
    pipeline = LaneDetectionPipeline()
    print("=" * 60)
    print("  Smart Road Lane Line Detection System")
    print("  Mode: Single Image Processing")
    print("=" * 60)
    print(f"  Input  : {os.path.abspath(args.input)}")
    print(f"  Size   : {image.shape[1]}x{image.shape[0]} pixels")
    print(f"  Output : {os.path.abspath(args.output)}")
    print("-" * 60)

    # ── Process image ──────────────────────────────────────────────
    start_time = time.time()
    result, metadata = pipeline.process_frame(image)
    elapsed = time.time() - start_time

    # ── Print metadata ─────────────────────────────────────────────
    print("\n[RESULTS]")
    print(f"  Processing time  : {elapsed * 1000:.1f} ms")
    print(f"  Left lane found  : {'Yes' if metadata.get('left_line') is not None else 'No'}")
    print(f"  Right lane found : {'Yes' if metadata.get('right_line') is not None else 'No'}")

    if metadata.get("curvature_radius") is not None:
        print(f"  Curvature radius : {metadata['curvature_radius']:.1f} m")

    warning_level = metadata.get("warning_level", "N/A")
    print(f"  Departure status : {warning_level.upper()}")
    print("-" * 60)

    # ── Save result ────────────────────────────────────────────────
    basename = os.path.splitext(os.path.basename(args.input))[0]
    output_path = os.path.join(args.output, f"{basename}_detected.jpg")
    cv2.imwrite(output_path, result)
    print(f"  Saved to: {os.path.abspath(output_path)}")
    print("=" * 60)

    # ── Display (optional) ─────────────────────────────────────────
    if not args.no_display:
        window_name = "Lane Detection Result"
        cv2.imshow(window_name, result)
        print("\n  Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("\nDone.")


if __name__ == "__main__":
    main()
