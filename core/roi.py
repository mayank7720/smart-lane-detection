"""
Smart Lane Detection — Region of Interest (ROI)
================================================
Defines the trapezoidal region-of-interest mask that isolates the road
surface from distracting scenery (sky, trees, dashboards, etc.).
"""

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config


def get_roi_vertices(frame_shape):
    """Compute ROI polygon vertices from fractional config values.

    The vertices in ``config.ROI_VERTICES_FRAC`` are specified as
    ``(x_fraction, y_fraction)`` of frame width and height.  This function
    scales them to absolute pixel coordinates and returns a NumPy array
    suitable for ``cv2.fillPoly``.

    Parameters
    ----------
    frame_shape : tuple
        Shape of the frame ``(height, width[, channels])``.

    Returns
    -------
    np.ndarray or None
        Integer array of shape ``(1, N, 2)`` with vertex coordinates, or
        ``None`` if `frame_shape` is invalid.
    """
    if frame_shape is None or len(frame_shape) < 2:
        return None

    h, w = frame_shape[:2]
    vertices = []
    for x_frac, y_frac in config.ROI_VERTICES_FRAC:
        vertices.append([int(x_frac * w), int(y_frac * h)])

    return np.array([vertices], dtype=np.int32)


def apply_roi(edges, vertices=None):
    """Mask the edge image so that only the ROI region is retained.

    Parameters
    ----------
    edges : np.ndarray
        Single-channel edge map (e.g. Canny output).
    vertices : np.ndarray, optional
        Polygon vertices shaped ``(1, N, 2)``.  If *None*, they are
        computed automatically from the edge map's shape via
        :func:`get_roi_vertices`.

    Returns
    -------
    np.ndarray or None
        Masked edge map with the same dtype as *edges*, or ``None`` if
        the input is invalid.
    """
    if edges is None or not isinstance(edges, np.ndarray):
        return None

    if vertices is None:
        vertices = get_roi_vertices(edges.shape)
    if vertices is None:
        return None

    # Create an all-black mask the same size as the edge map
    mask = np.zeros_like(edges)

    # Determine the fill value — 255 for uint8, max value otherwise
    if len(edges.shape) == 3:
        fill_value = (255,) * edges.shape[2]
    else:
        fill_value = 255

    cv2.fillPoly(mask, vertices, fill_value)

    # Keep only edge pixels inside the ROI
    return cv2.bitwise_and(edges, mask)
