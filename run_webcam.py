#!/usr/bin/env python3
"""
Smart Road Lane Line Detection System — Real-Time Webcam CLI
==============================================================
Capture frames from the webcam and process them through the
lane detection pipeline in real time.

Usage:
    python run_webcam.py

Controls:
    q  — Quit the application
"""

import os
import sys
import time

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
from core.pipeline import LaneDetectionPipeline


def main():
    # ── Open webcam ────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Please check your camera.")
        sys.exit(1)

    # Try to set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ── Initialize pipeline ────────────────────────────────────────
    pipeline = LaneDetectionPipeline()

    print("=" * 60)
    print("  Smart Road Lane Line Detection System")
    print("  Mode: Real-Time Webcam")
    print("=" * 60)
    print(f"  Camera resolution : {frame_width}x{frame_height}")
    print(f"  Press 'q' to quit")
    print("-" * 60)

    # ── Processing loop ────────────────────────────────────────────
    frame_count = 0
    detection_count = 0
    fps_times = []
    session_start = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Failed to read frame from webcam.")
                break

            frame_count += 1
            t0 = time.time()
            result, metadata = pipeline.process_frame(frame)
            t1 = time.time()

            fps_times.append(t1 - t0)
            # Keep only last 60 measurements for rolling FPS
            if len(fps_times) > 60:
                fps_times.pop(0)

            # Track detections
            if metadata.get("detection_success"):
                detection_count += 1

            # Calculate rolling FPS
            current_fps = 1.0 / (sum(fps_times) / len(fps_times)) if fps_times else 0

            # Draw FPS on the frame
            cv2.putText(
                result,
                f"FPS: {current_fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # Draw departure status
            departure = metadata.get("warning_level", "safe").upper()
            color = (0, 200, 0)  # Green by default
            if departure == "CAUTION":
                color = (0, 200, 255)
            elif departure == "DANGER":
                color = (0, 0, 255)

            cv2.putText(
                result,
                f"Status: {departure}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )

            # Show frame
            cv2.imshow("Lane Detection — Webcam (press 'q' to quit)", result)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n  [INFO] Interrupted by user.")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    # ── Session statistics ─────────────────────────────────────────
    session_elapsed = time.time() - session_start
    avg_fps = frame_count / session_elapsed if session_elapsed > 0 else 0
    detection_rate = (detection_count / frame_count * 100) if frame_count > 0 else 0

    print("\n")
    print("=" * 60)
    print("  SESSION STATISTICS")
    print("=" * 60)
    print(f"  Session duration   : {session_elapsed:.1f}s")
    print(f"  Frames processed   : {frame_count}")
    print(f"  Average FPS        : {avg_fps:.1f}")
    print(f"  Detection rate     : {detection_rate:.1f}% ({detection_count}/{frame_count})")
    print("=" * 60)
    print("\nSession ended. Goodbye!")


if __name__ == "__main__":
    main()
