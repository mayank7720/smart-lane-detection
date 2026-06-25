"""
Smart Lane Detection — Evaluation Metrics
==========================================
Functions for computing detection rate, average FPS, lane stability,
and generating a consolidated summary from pipeline run metadata.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config


def calculate_detection_rate(metadata_list):
    """Percentage of frames where at least one lane was detected.

    Parameters
    ----------
    metadata_list : list[dict]
        Per-frame metadata produced by
        :meth:`LaneDetectionPipeline.process_frame`.

    Returns
    -------
    float
        Detection rate as a percentage (0–100), or 0.0 if the list is
        empty.
    """
    if not metadata_list:
        return 0.0
    successes = sum(
        1 for m in metadata_list if m.get("detection_success", False)
    )
    return (successes / len(metadata_list)) * 100.0


def calculate_avg_fps(processing_times):
    """Compute the average frames-per-second from processing durations.

    Parameters
    ----------
    processing_times : list[float]
        Per-frame processing durations in seconds.

    Returns
    -------
    float
        Average FPS, or 0.0 if the list is empty or the mean time is
        zero.
    """
    if not processing_times:
        return 0.0
    mean_time = np.mean(processing_times)
    if mean_time <= 0:
        return 0.0
    return 1.0 / mean_time


def calculate_stability(metadata_list):
    """Measure lane position stability across consecutive frames.

    Stability is quantified as the inverse of the standard deviation
    of the bottom x-coordinates of the detected lanes.  Higher values
    mean more stable detections.  A perfectly stable lane yields
    ``float('inf')``.

    Returns a dict with ``left_std``, ``right_std``, ``left_stability``,
    ``right_stability``, and an overall ``stability_score`` (average of
    left and right, clipped to [0, 100]).

    Parameters
    ----------
    metadata_list : list[dict]
        Per-frame metadata.

    Returns
    -------
    dict
        Stability metrics.
    """
    result = {
        "left_std": None,
        "right_std": None,
        "left_stability": 0.0,
        "right_stability": 0.0,
        "stability_score": 0.0,
    }

    if not metadata_list:
        return result

    left_xs = []
    right_xs = []

    for m in metadata_list:
        left = m.get("left_line")
        right = m.get("right_line")
        if left is not None:
            left_xs.append(left[0])  # bottom x
        if right is not None:
            right_xs.append(right[0])

    def _stability(xs):
        if len(xs) < 2:
            return None, 0.0
        std = float(np.std(xs))
        if std < 1e-6:
            return std, 100.0
        # Map std → stability: lower std = higher stability
        # Use 100 / (1 + std) as a simple mapping to [0, 100)
        return std, min(100.0, 100.0 / (1.0 + std / 10.0))

    l_std, l_stab = _stability(left_xs)
    r_std, r_stab = _stability(right_xs)

    result["left_std"] = l_std
    result["right_std"] = r_std
    result["left_stability"] = l_stab
    result["right_stability"] = r_stab

    scores = [s for s in [l_stab, r_stab] if s > 0]
    result["stability_score"] = float(np.mean(scores)) if scores else 0.0

    return result


def generate_summary(metadata_list, processing_times):
    """Produce a consolidated evaluation summary.

    Parameters
    ----------
    metadata_list : list[dict]
        Per-frame metadata.
    processing_times : list[float]
        Per-frame processing times in seconds.

    Returns
    -------
    dict
        ``{detection_rate, avg_fps, stability, total_frames,
        frames_detected, meets_target_fps}``
    """
    if not metadata_list:
        metadata_list = []
    if not processing_times:
        processing_times = []

    detection_rate = calculate_detection_rate(metadata_list)
    avg_fps = calculate_avg_fps(processing_times)
    stability = calculate_stability(metadata_list)

    frames_detected = sum(
        1 for m in metadata_list if m.get("detection_success", False)
    )

    return {
        "total_frames": len(metadata_list),
        "frames_detected": frames_detected,
        "detection_rate": round(detection_rate, 2),
        "avg_fps": round(avg_fps, 2),
        "meets_target_fps": avg_fps >= config.TARGET_FPS,
        "stability": stability,
    }
