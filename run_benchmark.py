#!/usr/bin/env python3
"""
Smart Road Lane Line Detection System — Benchmark and Report Runner
====================================================================
Generates a synthetic test image, runs the degradation benchmark suite,
and produces the visual charts and summary report.
"""

import os
import sys
import time

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import config
from core.pipeline import LaneDetectionPipeline
from evaluation.benchmark import run_benchmark
from evaluation.report_generator import generate_full_report


def create_synthetic_road_image(path):
    """Draw a synthetic road scene with clear lanes to test the pipeline."""
    h, w = 720, 1280
    # Background (green field)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (34, 139, 34)  # Forest green (BGR)

    # Road surface (gray trapezoid)
    road_pts = np.array([
        [100, h],
        [w // 2 - 120, int(h * 0.55)],
        [w // 2 + 120, int(h * 0.55)],
        [w - 100, h]
    ], np.int32)
    cv2.fillPoly(img, [road_pts], (60, 60, 60))  # Dark gray (BGR)

    # Lane boundaries (solid white lines)
    cv2.line(img, (250, h), (w // 2 - 80, int(h * 0.58)), (255, 255, 255), 12, cv2.LINE_AA)
    cv2.line(img, (w - 250, h), (w // 2 + 80, int(h * 0.58)), (255, 255, 255), 12, cv2.LINE_AA)

    # Dashboard/sky horizon line or details (optional, but keep it clean)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)
    print(f"[INFO] Created synthetic test image at: {path}")


def main():
    print("=" * 60)
    print("  Smart Road Lane Line Detection System")
    print("  Evaluation & Report Generator")
    print("=" * 60)

    # Ensure sample image exists
    sample_path = os.path.join(config.SAMPLE_DIR, "road.jpg")
    create_synthetic_road_image(sample_path)

    # Initialize pipeline
    print("\n[STEP 1] Initializing Lane Detection Pipeline...")
    pipeline = LaneDetectionPipeline()

    # Run Benchmark
    print("\n[STEP 2] Running benchmark on conditions (brightness, noise, contrast)...")
    t0 = time.time()
    results = run_benchmark(pipeline, sample_path)
    elapsed = time.time() - t0

    if "error" in results:
        print(f"[ERROR] Benchmark execution failed: {results['error']}")
        sys.exit(1)

    print(f"Benchmark finished in {elapsed:.2f}s.")
    summary = results["summary"]
    print(f"  Total conditions tested : {summary['total_conditions']}")
    print(f"  Passed (lanes detected) : {summary['passed']}")
    print(f"  Overall detection rate  : {summary['detection_rate']}%")

    # Generate Reports
    print("\n[STEP 3] Generating visual charts and summary report...")
    report = generate_full_report(results, config.REPORT_DIR)

    if report.get("success"):
        print("=" * 60)
        print("  REPORT GENERATION COMPLETED")
        print("=" * 60)
        print(f"  Summary text     : {os.path.abspath(report['summary_txt'])}")
        print(f"  Detection chart  : {os.path.abspath(report['detection_chart'])}")
        print(f"  FPS chart        : {os.path.abspath(report['fps_chart'])}")
        print(f"  Heatmap chart    : {os.path.abspath(report['condition_heatmap'])}")
        print("=" * 60)
    else:
        print(f"[ERROR] Report generation failed: {report.get('summary_error')}")


if __name__ == "__main__":
    main()
