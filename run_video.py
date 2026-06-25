#!/usr/bin/env python3
"""
Smart Road Lane Line Detection System — Video Processing CLI
==============================================================
Process a video file through the lane detection pipeline.

Usage:
    python run_video.py --input path/to/video.mp4
    python run_video.py --input path/to/video.mp4 --output output/videos/ --no-display
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
        description="Smart Road Lane Line Detection — Video Processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_video.py --input data/sample/highway.mp4
  python run_video.py --input clip.avi --output results/ --no-display
        """,
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the input video file (mp4, avi, mov, mkv)",
    )
    parser.add_argument(
        "--output", "-o",
        default=os.path.join("output", "videos"),
        help="Directory to save the processed video (default: output/videos/)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Suppress the real-time display window",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Validate input ─────────────────────────────────────────────
    if not os.path.isfile(args.input):
        print(f"[ERROR] Input file not found: {args.input}")
        sys.exit(1)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {args.input}")
        sys.exit(1)

    # ── Video properties ───────────────────────────────────────────
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    # ── Ensure output directory exists ─────────────────────────────
    os.makedirs(args.output, exist_ok=True)

    basename = os.path.splitext(os.path.basename(args.input))[0]
    output_path = os.path.join(args.output, f"{basename}_detected.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # ── Initialize pipeline ────────────────────────────────────────
    pipeline = LaneDetectionPipeline()

    print("=" * 60)
    print("  Smart Road Lane Line Detection System")
    print("  Mode: Video Processing")
    print("=" * 60)
    print(f"  Input      : {os.path.abspath(args.input)}")
    print(f"  Resolution : {frame_width}x{frame_height}")
    print(f"  FPS        : {fps:.1f}")
    print(f"  Duration   : {duration:.1f}s ({total_frames} frames)")
    print(f"  Output     : {os.path.abspath(output_path)}")
    print("-" * 60)
    print("  Processing... (press 'q' to abort)\n")

    # ── Processing loop ────────────────────────────────────────────
    frame_count = 0
    detection_count = 0
    processing_times = []
    session_start = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            t0 = time.time()
            result, metadata = pipeline.process_frame(frame)
            t1 = time.time()

            processing_times.append(t1 - t0)

            # Track detections
            if metadata.get("detection_success"):
                detection_count += 1

            # Write output frame
            writer.write(result)

            # Progress
            if frame_count % 30 == 0 or frame_count == total_frames:
                pct = (frame_count / total_frames * 100) if total_frames > 0 else 0
                avg_ms = (sum(processing_times[-30:]) / len(processing_times[-30:])) * 1000
                sys.stdout.write(
                    f"\r  Frame {frame_count}/{total_frames} "
                    f"({pct:.0f}%) | {avg_ms:.1f} ms/frame"
                )
                sys.stdout.flush()

            # Display (optional)
            if not args.no_display:
                cv2.imshow("Lane Detection — Video", result)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\n\n  [INFO] Processing aborted by user.")
                    break

    except KeyboardInterrupt:
        print("\n\n  [INFO] Processing interrupted.")

    finally:
        cap.release()
        writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    # ── Summary statistics ─────────────────────────────────────────
    session_elapsed = time.time() - session_start
    avg_fps = frame_count / session_elapsed if session_elapsed > 0 else 0
    detection_rate = (detection_count / frame_count * 100) if frame_count > 0 else 0
    avg_processing = (
        sum(processing_times) / len(processing_times) * 1000
        if processing_times else 0
    )

    print("\n")
    print("=" * 60)
    print("  PROCESSING SUMMARY")
    print("=" * 60)
    print(f"  Frames processed   : {frame_count}/{total_frames}")
    print(f"  Total time         : {session_elapsed:.1f}s")
    print(f"  Average FPS        : {avg_fps:.1f}")
    print(f"  Avg processing/frm : {avg_processing:.1f} ms")
    print(f"  Detection rate     : {detection_rate:.1f}% ({detection_count}/{frame_count})")
    print(f"  Output saved to    : {os.path.abspath(output_path)}")
    print("=" * 60)
    print("\nDone.")


if __name__ == "__main__":
    main()
