"""
Smart Lane Detection — Lane Departure Warning
===============================================
Computes the vehicle's lateral offset from the lane centre and issues
graduated warnings (safe → caution → danger) when the offset exceeds
configurable thresholds.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config


def calculate_offset(left_line, right_line, frame_width):
    """Calculate the vehicle's lateral offset from the lane centre.

    The lane centre is defined as the midpoint between the bottom
    x-coordinates of the two detected lane lines.  The vehicle is
    assumed to be centred at ``frame_width / 2``.

    The returned offset is *normalised* to the lane width so that
    ``0.0`` means perfectly centred and ``±1.0`` means one full lane
    width away (i.e. completely outside the lane).

    Parameters
    ----------
    left_line : tuple or None
        ``(x1, y1, x2, y2)`` — bottom point is ``(x1, y1)``.
    right_line : tuple or None
        ``(x1, y1, x2, y2)`` — bottom point is ``(x1, y1)``.
    frame_width : int
        Width of the source frame in pixels.

    Returns
    -------
    float or None
        Signed offset fraction (negative = shifted left, positive =
        shifted right).  ``None`` if either lane is missing.
    """
    if left_line is None or right_line is None:
        return None
    if frame_width <= 0:
        return None

    # Bottom x-coordinates (the first point of each extrapolated line)
    left_x_bottom = left_line[0]
    right_x_bottom = right_line[0]

    lane_centre = (left_x_bottom + right_x_bottom) / 2.0
    frame_centre = frame_width / 2.0
    lane_width = abs(right_x_bottom - left_x_bottom)

    if lane_width < 1:
        return None

    # Positive offset → vehicle is to the right of lane centre
    offset = (frame_centre - lane_centre) / lane_width
    return offset


def check_departure(offset, caution_thresh=None, danger_thresh=None):
    """Classify the departure severity based on the offset magnitude.

    Parameters
    ----------
    offset : float or None
        Normalised lateral offset (from :func:`calculate_offset`).
    caution_thresh : float, optional
        Offset magnitude at which to trigger a caution alert.
        Defaults to ``config.DEPARTURE_THRESHOLD_CAUTION``.
    danger_thresh : float, optional
        Offset magnitude at which to trigger a danger alert.
        Defaults to ``config.DEPARTURE_THRESHOLD_DANGER``.

    Returns
    -------
    dict
        ``{"level": str, "offset": float, "message": str}`` where
        *level* is ``"safe"``, ``"caution"``, or ``"danger"``, and
        *message* is a human-readable description.
    """
    if caution_thresh is None:
        caution_thresh = config.DEPARTURE_THRESHOLD_CAUTION
    if danger_thresh is None:
        danger_thresh = config.DEPARTURE_THRESHOLD_DANGER

    # Unknown offset — treat as safe but flag it
    if offset is None:
        return {
            "level": "safe",
            "offset": 0.0,
            "message": "Lane boundaries not fully detected — offset unavailable.",
        }

    abs_offset = abs(offset)
    direction = "left" if offset < 0 else "right"

    if abs_offset >= danger_thresh:
        return {
            "level": "danger",
            "offset": offset,
            "message": (f"DANGER: Vehicle drifting {direction} — "
                        f"offset {abs_offset:.2f} exceeds danger threshold."),
        }
    if abs_offset >= caution_thresh:
        return {
            "level": "caution",
            "offset": offset,
            "message": (f"Caution: Vehicle drifting {direction} — "
                        f"offset {abs_offset:.2f}."),
        }
    return {
        "level": "safe",
        "offset": offset,
        "message": "Vehicle centred in lane.",
    }
