"""
Smart Lane Detection — Lane Detector
=====================================
Detects lane lines from a masked edge image using the Probabilistic
Hough Transform, classifies them into left/right lanes by slope,
averages each group into a single representative line, and optionally
fits polynomial curves for curved-lane scenarios.
"""

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config


class LaneTuple(tuple):
    """Custom tuple subclass that behaves like a 4-tuple but stores a curve array."""
    def __new__(cls, x1, y1, x2, y2, curve=None):
        obj = super(LaneTuple, cls).__new__(cls, (x1, y1, x2, y2))
        obj.curve = curve
        return obj


# ── Hough Transform ──────────────────────────────────────────────────────────

def hough_lines(roi_edges):
    """Detect line segments via the Probabilistic Hough Transform.

    Parameters
    ----------
    roi_edges : np.ndarray
        Single-channel edge image (ROI-masked).

    Returns
    -------
    np.ndarray or None
        Array of shape ``(N, 1, 4)`` where each row is
        ``[x1, y1, x2, y2]``, or ``None`` when no lines are found.
    """
    if roi_edges is None or not isinstance(roi_edges, np.ndarray):
        return None

    theta = np.deg2rad(config.HOUGH_THETA_DEGREES)
    lines = cv2.HoughLinesP(
        roi_edges,
        rho=config.HOUGH_RHO,
        theta=theta,
        threshold=config.HOUGH_THRESHOLD,
        minLineLength=config.HOUGH_MIN_LINE_LENGTH,
        maxLineGap=config.HOUGH_MAX_LINE_GAP,
    )
    return lines  # may be None if nothing detected


# ── Classification ───────────────────────────────────────────────────────────

def classify_lines(lines, frame_width):
    """Separate detected lines into left-lane and right-lane groups.

    Classification rules
    --------------------
    * **Left lane**: negative slope *and* midpoint x < frame centre.
    * **Right lane**: positive slope *and* midpoint x > frame centre.
    * Lines whose absolute slope falls outside
      ``[MIN_SLOPE, MAX_SLOPE]`` are rejected.

    Parameters
    ----------
    lines : np.ndarray
        Hough lines array ``(N, 1, 4)``.
    frame_width : int
        Width of the source frame in pixels.

    Returns
    -------
    tuple[list, list]
        ``(left_lines, right_lines)`` — each element is a list of
        ``(x1, y1, x2, y2)`` tuples.
    """
    left_lines = []
    right_lines = []

    if lines is None or frame_width <= 0:
        return left_lines, right_lines

    mid_x = frame_width / 2.0

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue  # vertical line — skip to avoid division by zero

        slope = (y2 - y1) / (x2 - x1)
        abs_slope = abs(slope)

        # Reject near-horizontal and near-vertical lines
        if abs_slope < config.MIN_SLOPE or abs_slope > config.MAX_SLOPE:
            continue

        line_mid_x = (x1 + x2) / 2.0

        if slope < 0 and line_mid_x < mid_x:
            left_lines.append((x1, y1, x2, y2))
        elif slope > 0 and line_mid_x > mid_x:
            right_lines.append((x1, y1, x2, y2))

    return left_lines, right_lines


# ── Averaging / Extrapolation ────────────────────────────────────────────────

def average_line(lines, frame_height, roi_edges=None):
    """Average a group of line segments and extrapolate. Fits a polynomial for curves.

    Parameters
    ----------
    lines : list[tuple]
        List of ``(x1, y1, x2, y2)`` segments belonging to one lane.
    frame_height : int
        Height of the source frame.
    roi_edges : np.ndarray, optional
        Binary edge image (ROI masked) to extract raw lane pixels for curve fitting.

    Returns
    -------
    LaneTuple or None
        LaneTuple representing the averaged/extrapolated lane line and curve.
    """
    if not lines or frame_height <= 0:
        return None

    # Filter outliers using median slope and intercept
    slopes = []
    intercepts = []
    line_data = []
    for x1, y1, x2, y2 in lines:
        dx = x2 - x1
        if dx == 0:
            continue
        slope = (y2 - y1) / dx
        intercept = y1 - slope * x1
        length = np.sqrt(dx ** 2 + (y2 - y1) ** 2)
        slopes.append(slope)
        intercepts.append(intercept)
        line_data.append((x1, y1, x2, y2, slope, intercept, length))

    if not slopes:
        return None

    median_slope = np.median(slopes)
    median_intercept = np.median(intercepts)

    # Keep lines close to the median slope and intercept
    inliers = []
    for x1, y1, x2, y2, slope, intercept, length in line_data:
        if abs(slope - median_slope) <= 0.25 and abs(intercept - median_intercept) <= 120.0:
            inliers.append((x1, y1, x2, y2, slope, intercept, length))

    # Fall back if we filtered out everything
    if not inliers:
        inliers = line_data

    # Average the inliers
    slopes = [item[4] for item in inliers]
    intercepts = [item[5] for item in inliers]
    weights = [item[6] for item in inliers]

    avg_slope = np.average(slopes, weights=weights)
    avg_intercept = np.average(intercepts, weights=weights)

    if abs(avg_slope) < 1e-6:
        return None

    # Extrapolate: y_bottom and y_top from config fractions
    y_bottom = int(frame_height * config.LANE_EXTRAPOLATE_Y_BOTTOM_FRAC)
    y_top = int(frame_height * config.LANE_EXTRAPOLATE_Y_TOP_FRAC)

    x_bottom = int((y_bottom - avg_intercept) / avg_slope)
    x_top = int((y_top - avg_intercept) / avg_slope)

    curve_points = None

    # Primary: Fit a polynomial to the actual Canny edge pixels in the neighborhood of the averaged line
    if roi_edges is not None:
        try:
            y_indices, x_indices = np.nonzero(roi_edges)
            dy = y_top - y_bottom
            if dy != 0:
                slope_dx_dy = (x_top - x_bottom) / dy
                intercept_dx_dy = x_bottom - slope_dx_dy * y_bottom
                expected_x = slope_dx_dy * y_indices + intercept_dx_dy
                distances = np.abs(x_indices - expected_x)
                
                # Filter pixels within 50 pixels of the straight line laterally
                mask = (distances < 50) & (y_indices >= y_top) & (y_indices <= y_bottom)
                lane_x = x_indices[mask]
                lane_y = y_indices[mask]

                if len(lane_y) >= 20:
                    coeffs = np.polyfit(lane_y, lane_x, 2)
                    deviation = abs(coeffs[0]) * (y_bottom - y_top)**2 / 8.0
                    
                    if deviation >= 3.0:
                        y_eval = np.linspace(y_bottom, y_top, num=50)
                        x_eval = np.polyval(coeffs, y_eval)
                        curve_points = np.column_stack((x_eval, y_eval))
                        # Update endpoints to match curve ends
                        x_bottom = int(x_eval[0])
                        x_top = int(x_eval[-1])
        except Exception:
            curve_points = None

    # Secondary (fallback): If no pixel curve found, try fitting to Hough segment points
    if curve_points is None:
        pts = []
        for x1, y1, x2, y2, _, _, length in inliers:
            num_pts = max(2, int(length / 5))
            for t in np.linspace(0, 1, num_pts):
                px = x1 + t * (x2 - x1)
                py = y1 + t * (y2 - y1)
                pts.append((px, py))

        if len(pts) >= 3:
            pts = np.array(pts)
            xs = pts[:, 0]
            ys = pts[:, 1]
            try:
                coeffs = np.polyfit(ys, xs, 2)
                deviation = abs(coeffs[0]) * (y_bottom - y_top)**2 / 8.0
                
                if deviation >= 3.0:
                    y_eval = np.linspace(y_bottom, y_top, num=50)
                    x_eval = np.polyval(coeffs, y_eval)
                    curve_points = np.column_stack((x_eval, y_eval))
                    x_bottom = int(x_eval[0])
                    x_top = int(x_eval[-1])
            except Exception:
                curve_points = None

    return LaneTuple(x_bottom, y_bottom, x_top, y_top, curve=curve_points)


# ── Polynomial Fitting (curved lanes) ───────────────────────────────────────

def fit_polynomial(points, frame_height, degree=2):
    """Fit a polynomial curve through a set of lane points.

    Parameters
    ----------
    points : list[tuple] or np.ndarray
        Collection of ``(x, y)`` points along one lane.
    frame_height : int
        Height of the source frame (used to generate evenly spaced
        evaluation points from bottom to top).
    degree : int, optional
        Polynomial degree (default 2 — quadratic).

    Returns
    -------
    np.ndarray or None
        Array of shape ``(M, 2)`` with ``(x, y)`` points along the
        fitted curve, or ``None`` on failure.
    """
    if points is None or len(points) < degree + 1:
        return None

    points = np.array(points)
    if points.ndim != 2 or points.shape[1] != 2:
        return None

    xs = points[:, 0]
    ys = points[:, 1]

    try:
        coeffs = np.polyfit(ys, xs, degree)
    except (np.RankWarning, np.linalg.LinAlgError, ValueError):
        return None

    y_top = int(frame_height * config.LANE_EXTRAPOLATE_Y_TOP_FRAC)
    y_bottom = int(frame_height * config.LANE_EXTRAPOLATE_Y_BOTTOM_FRAC)
    y_eval = np.linspace(y_top, y_bottom, num=y_bottom - y_top + 1)
    x_eval = np.polyval(coeffs, y_eval)

    curve_points = np.column_stack((x_eval.astype(int), y_eval.astype(int)))
    return curve_points


# ── Unified Detection Entry-point ────────────────────────────────────────────

def detect_lanes(roi_edges, frame_shape):
    """Detect left and right lane lines from a ROI-masked edge image.

    Pipeline: Hough → classify → average/extrapolate.

    Parameters
    ----------
    roi_edges : np.ndarray
        Single-channel, ROI-masked edge map.
    frame_shape : tuple
        ``(height, width[, channels])`` of the original frame.

    Returns
    -------
    tuple
        ``(left_line, right_line)`` where each is either
        ``(x1, y1, x2, y2)`` or ``None`` if that lane was not detected.
    """
    if roi_edges is None or frame_shape is None or len(frame_shape) < 2:
        return None, None

    h, w = frame_shape[:2]

    lines = hough_lines(roi_edges)
    if lines is None:
        return None, None

    left_lines, right_lines = classify_lines(lines, w)

    left_lane = average_line(left_lines, h, roi_edges)
    right_lane = average_line(right_lines, h, roi_edges)

    return left_lane, right_lane
