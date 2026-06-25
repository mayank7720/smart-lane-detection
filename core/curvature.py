"""
Smart Lane Detection — Curvature Estimation
============================================
Estimates the radius of curvature of the detected lane from the left
and right lane boundary lines, converting pixel measurements to
real-world metres.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config


def estimate_curvature(left_line, right_line, frame_shape):
    """Estimate the lane curvature radius in metres.

    Steps
    -----
    1. Generate evenly-spaced points along each lane line.
    2. Average x-coordinates at each y level to get the lane centre line.
    3. Fit a second-degree polynomial to the centre line.
    4. Compute *R* = (1 + (2Ay + B)²)^1.5 / |2A| in the real-world
       metric space using the pixel-to-metre conversion factors from
       ``config``.

    Parameters
    ----------
    left_line : tuple or None
        ``(x1, y1, x2, y2)`` of the averaged left lane boundary.
    right_line : tuple or None
        ``(x1, y1, x2, y2)`` of the averaged right lane boundary.
    frame_shape : tuple
        ``(height, width[, channels])`` of the source frame.

    Returns
    -------
    float or None
        Curvature radius in metres.  Positive values indicate a
        right-curving lane; negative values a left curve.  Returns
        ``None`` when insufficient data is available.
    """
    if frame_shape is None or len(frame_shape) < 2:
        return None

    h, w = frame_shape[:2]

    # We need at least one lane to estimate curvature
    if left_line is None and right_line is None:
        return None

    def _line_points(line):
        """Generate (x, y) points along a lane line, using curve array if present."""
        curve = getattr(line, 'curve', None)
        if curve is not None:
            return curve[:, 0], curve[:, 1]

        x1, y1, x2, y2 = line
        # Generate y values from bottom to top
        y_vals = np.linspace(y1, y2, num=max(abs(y2 - y1), 50))
        if abs(y2 - y1) < 1:
            x_vals = np.full_like(y_vals, (x1 + x2) / 2.0)
        else:
            slope = (x2 - x1) / (y2 - y1)
            x_vals = x1 + slope * (y_vals - y1)
        return x_vals, y_vals

    if left_line is not None and right_line is not None:
        lx, ly = _line_points(left_line)
        rx, ry = _line_points(right_line)

        # Resample both to the same y range
        y_min = max(ly.min(), ry.min())
        y_max = min(ly.max(), ry.max())
        if y_max <= y_min:
            # Fall back to using only the longer line
            if abs(left_line[1] - left_line[3]) >= abs(right_line[1] - right_line[3]):
                cx, cy = lx, ly
            else:
                cx, cy = rx, ry
        else:
            num_pts = int(y_max - y_min) + 1
            common_y = np.linspace(y_min, y_max, num=num_pts)
            lx_interp = np.interp(common_y, np.sort(ly), lx[np.argsort(ly)])
            rx_interp = np.interp(common_y, np.sort(ry), rx[np.argsort(ry)])
            cx = (lx_interp + rx_interp) / 2.0
            cy = common_y
    else:
        # Only one lane available — use it directly
        lane = left_line if left_line is not None else right_line
        cx, cy = _line_points(lane)

    # Convert pixel coordinates to metres
    cy_m = cy * config.METERS_PER_PIXEL_Y
    cx_m = cx * config.METERS_PER_PIXEL_X

    # Fit second-degree polynomial x = f(y) in metres
    if len(cy_m) < 3:
        return None

    try:
        coeffs = np.polyfit(cy_m, cx_m, config.CURVATURE_POLY_DEGREE)
    except (np.RankWarning, np.linalg.LinAlgError, ValueError):
        return None

    # Evaluate curvature at the bottom of the frame (closest to car)
    A = coeffs[0]
    B = coeffs[1]
    y_eval = (h - 1) * config.METERS_PER_PIXEL_Y

    denominator = abs(2.0 * A)
    if denominator < 1e-9:
        # Essentially straight
        return 1e6  # very large radius → straight road

    numerator = (1.0 + (2.0 * A * y_eval + B) ** 2) ** 1.5
    radius = numerator / denominator

    # Sign convention: positive A → curve to the right
    if A < 0:
        radius = -radius

    return radius


def determine_direction(radius, threshold=500):
    """Human-readable driving direction from curvature radius.

    Parameters
    ----------
    radius : float or None
        Curvature radius in metres (from :func:`estimate_curvature`).
    threshold : float
        If ``|radius|`` exceeds this value the road is classified as
        straight.

    Returns
    -------
    str
        ``"Straight"``, ``"Left Curve"``, or ``"Right Curve"``.
    """
    if radius is None:
        return "Unknown"
    if abs(radius) > threshold:
        return "Straight"
    if radius < 0:
        return "Left Curve"
    return "Right Curve"
