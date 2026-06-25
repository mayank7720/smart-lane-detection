"""
Smart Lane Detection — Benchmark Runner
========================================
Stress-tests the lane detection pipeline under a range of synthetic
image degradations (brightness, noise, contrast) and collects
per-condition performance metrics.
"""

import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config


# ── Image Degradation Helpers ────────────────────────────────────────────────

def adjust_brightness(frame, factor):
    """Scale pixel values by *factor* to simulate brighter/darker scenes.

    Parameters
    ----------
    frame : np.ndarray
        BGR image.
    factor : float
        Multiplicative brightness factor.  ``<1`` darkens, ``>1``
        brightens.

    Returns
    -------
    np.ndarray or None
        Brightness-adjusted image (uint8), or ``None`` if input is
        invalid.
    """
    if frame is None or not isinstance(frame, np.ndarray):
        return None
    adjusted = np.clip(frame.astype(np.float32) * factor, 0, 255)
    return adjusted.astype(np.uint8)


def add_noise(frame, sigma):
    """Add Gaussian noise with standard deviation *sigma*.

    Parameters
    ----------
    frame : np.ndarray
        BGR image.
    sigma : float
        Standard deviation of the additive Gaussian noise.

    Returns
    -------
    np.ndarray or None
        Noisy image (uint8), or ``None`` if input is invalid.
    """
    if frame is None or not isinstance(frame, np.ndarray):
        return None
    if sigma <= 0:
        return frame.copy()
    noise = np.random.normal(0, sigma, frame.shape).astype(np.float32)
    noisy = np.clip(frame.astype(np.float32) + noise, 0, 255)
    return noisy.astype(np.uint8)


def adjust_contrast(frame, factor):
    """Adjust contrast by scaling deviation from the per-channel mean.

    Parameters
    ----------
    frame : np.ndarray
        BGR image.
    factor : float
        ``<1`` reduces contrast; ``>1`` increases it.

    Returns
    -------
    np.ndarray or None
        Contrast-adjusted image (uint8), or ``None`` if input is
        invalid.
    """
    if frame is None or not isinstance(frame, np.ndarray):
        return None
    f32 = frame.astype(np.float32)
    mean = f32.mean(axis=(0, 1), keepdims=True)
    adjusted = np.clip(mean + factor * (f32 - mean), 0, 255)
    return adjusted.astype(np.uint8)


# ── Benchmark Runner ────────────────────────────────────────────────────────

def run_benchmark(pipeline, image_path):
    """Run the pipeline on an image under various degradation conditions.

    Conditions tested:

    * **Brightness**: each factor in ``config.BENCHMARK_BRIGHTNESS_FACTORS``
    * **Noise**: each sigma in ``config.BENCHMARK_NOISE_LEVELS``
    * **Contrast**: factors ``[0.5, 0.75, 1.0, 1.5, 2.0]``

    Parameters
    ----------
    pipeline : LaneDetectionPipeline
        An initialised pipeline instance.
    image_path : str
        Path to the test image.

    Returns
    -------
    dict
        Nested structure::

            {
                "image_path": str,
                "conditions": {
                    "<label>": {
                        "detection_success": bool,
                        "processing_time": float,
                        "fps": float,
                        "warning_level": str,
                        "curvature_radius": float | None,
                        "offset": float | None,
                    },
                    ...
                },
                "summary": {
                    "total_conditions": int,
                    "passed": int,
                    "detection_rate": float,
                }
            }
    """
    if not os.path.isfile(image_path):
        return {"error": f"Image not found: {image_path}"}

    frame = cv2.imread(image_path)
    if frame is None:
        return {"error": f"Failed to read image: {image_path}"}

    conditions = {}

    def _run(label, img):
        """Process a single degraded image and record metrics."""
        # Reset pipeline state for fair comparison
        pipeline._prev_left = None
        pipeline._prev_right = None

        t0 = time.perf_counter()
        _, meta = pipeline.process_frame(img)
        elapsed = time.perf_counter() - t0

        conditions[label] = {
            "detection_success": meta.get("detection_success", False),
            "processing_time": round(elapsed, 4),
            "fps": round(1.0 / elapsed, 2) if elapsed > 0 else 0.0,
            "warning_level": meta.get("warning_level", "safe"),
            "curvature_radius": meta.get("curvature_radius"),
            "offset": meta.get("offset"),
        }

    # ── Original ─────────────────────────────────────────────────────────
    _run("original", frame)

    # ── Brightness variations ────────────────────────────────────────────
    for bf in config.BENCHMARK_BRIGHTNESS_FACTORS:
        if bf == 1.0:
            continue  # skip duplicate of original
        label = f"brightness_{bf:.2f}"
        _run(label, adjust_brightness(frame, bf))

    # ── Noise variations ─────────────────────────────────────────────────
    for sigma in config.BENCHMARK_NOISE_LEVELS:
        if sigma == 0:
            continue
        label = f"noise_sigma_{sigma}"
        _run(label, add_noise(frame, sigma))

    # ── Contrast variations ──────────────────────────────────────────────
    contrast_factors = [0.5, 0.75, 1.0, 1.5, 2.0]
    for cf in contrast_factors:
        if cf == 1.0:
            continue
        label = f"contrast_{cf:.2f}"
        _run(label, adjust_contrast(frame, cf))

    # ── Summary ──────────────────────────────────────────────────────────
    total = len(conditions)
    passed = sum(1 for v in conditions.values() if v["detection_success"])

    return {
        "image_path": image_path,
        "conditions": conditions,
        "summary": {
            "total_conditions": total,
            "passed": passed,
            "detection_rate": round((passed / total) * 100, 2) if total else 0.0,
        },
    }
