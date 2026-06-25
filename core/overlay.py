"""
Smart Lane Detection — Overlay Drawing
=======================================
Draws lane lines, a semi-transparent lane fill, and a heads-up-display
(HUD) with curvature, offset, warning level, and FPS onto the video
frame.
"""

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config


# ── Helpers ──────────────────────────────────────────────────────────────────

def _warning_color(level):
    """Return the BGR colour tuple for a given warning level string."""
    level = (level or "safe").lower()
    if level == "danger":
        return config.WARNING_COLOR_DANGER
    if level == "caution":
        return config.WARNING_COLOR_CAUTION
    return config.WARNING_COLOR_SAFE


# ── Helpers for Curved Roads ──────────────────────────────────────────────────

def _generate_straight_curve(line, num_pts=50):
    """Generate (x, y) coordinates along a straight line segment."""
    x1, y1, x2, y2 = line
    y_vals = np.linspace(y1, y2, num=num_pts)
    if abs(y2 - y1) < 1:
        x_vals = np.full_like(y_vals, (x1 + x2) / 2.0)
    else:
        slope = (x2 - x1) / (y2 - y1)
        x_vals = x1 + slope * (y_vals - y1)
    return np.column_stack((x_vals, y_vals))


def get_x_at_y(line, y):
    """Interpolate the x coordinate at a specific y value for a line/curve."""
    curve = getattr(line, 'curve', None)
    if curve is not None:
        xs = curve[:, 0]
        ys = curve[:, 1]
        sort_idx = np.argsort(ys)
        return int(np.interp(y, ys[sort_idx], xs[sort_idx]))
    else:
        x1, y1, x2, y2 = line
        if y1 == y2:
            return int((x1 + x2) / 2)
        t = (y - y2) / (y1 - y2)
        return int(x2 + t * (x1 - x2))


def _clip_line(line, y_hood):
    """Clip a line or curve to keep only parts above the hood line."""
    if line is None:
        return None
    x1, y1, x2, y2 = line
    curve = getattr(line, 'curve', None)
    
    # Clip straight coordinates
    if y1 > y_hood:
        if y1 != y2:
            t = (y_hood - y2) / (y1 - y2)
            x_hood = int(x2 + t * (x1 - x2))
        else:
            x_hood = x1
        new_coords = (x_hood, y_hood, x2, y2)
    else:
        new_coords = (x1, y1, x2, y2)
        
    # Clip curve points
    new_curve = None
    if curve is not None:
        mask = curve[:, 1] <= y_hood
        filtered = curve[mask]
        
        # Prepend the hood intersection point if the line was clipped
        if y1 > y_hood and len(filtered) > 0:
            x_hood = get_x_at_y(line, y_hood)
            hood_pt = np.array([[x_hood, y_hood]])
            new_curve = np.vstack((hood_pt, filtered))
        else:
            new_curve = filtered if len(filtered) > 0 else None
            
    from core.lane_detector import LaneTuple
    return LaneTuple(*new_coords, curve=new_curve)


def draw_centerline(frame, left_line, right_line, color=(0, 255, 255), thickness=2):
    """Draw the lane centerline between the left and right lane boundaries."""
    if frame is None or not isinstance(frame, np.ndarray):
        return frame
    if left_line is None or right_line is None:
        return frame

    # Generate points along both curves
    left_curve = getattr(left_line, 'curve', None)
    if left_curve is None:
        left_curve = _generate_straight_curve(left_line)

    right_curve = getattr(right_line, 'curve', None)
    if right_curve is None:
        right_curve = _generate_straight_curve(right_line)

    # Get common y-range
    ly = left_curve[:, 1]
    ry = right_curve[:, 1]
    y_min = max(ly.min(), ry.min())
    y_max = min(ly.max(), ry.max())

    if y_max <= y_min:
        return frame

    # Resample both on a common y grid (50 points)
    common_y = np.linspace(y_min, y_max, num=50)
    
    # Sort for interpolation (np.interp requires sorted x-coordinates)
    sort_l = np.argsort(ly)
    sort_r = np.argsort(ry)
    
    lx_interp = np.interp(common_y, ly[sort_l], left_curve[:, 0][sort_l])
    rx_interp = np.interp(common_y, ry[sort_r], right_curve[:, 0][sort_r])
    
    cx = (lx_interp + rx_interp) / 2.0
    
    pts = np.column_stack((cx, common_y)).astype(np.int32).reshape((-1, 1, 2))
    
    # Draw centerline as a dashed line
    for i in range(0, len(pts) - 1, 2):
        cv2.line(frame, tuple(pts[i][0]), tuple(pts[i+1][0]), color, thickness, cv2.LINE_AA)

    return frame


# ── Lane Boundary Lines ─────────────────────────────────────────────────────

def draw_lane_lines(frame, left_line, right_line, color=None, thickness=None):
    """Draw the two detected lane boundary lines onto the frame, supporting curves.

    Parameters
    ----------
    frame : np.ndarray
        BGR image to draw on (modified in-place *and* returned).
    left_line, right_line : tuple or None
        ``(x1, y1, x2, y2)`` for each lane boundary.
    color : tuple[int, int, int], optional
        BGR colour.  Defaults to ``config.LANE_LINE_COLOR``.
    thickness : int, optional
        Line thickness.  Defaults to ``config.LANE_LINE_THICKNESS``.

    Returns
    -------
    np.ndarray
        The annotated frame.
    """
    if frame is None or not isinstance(frame, np.ndarray):
        return frame

    if color is None:
        color = config.LANE_LINE_COLOR
    if thickness is None:
        thickness = config.LANE_LINE_THICKNESS

    for line in [left_line, right_line]:
        if line is not None:
            curve = getattr(line, 'curve', None)
            if curve is not None and len(curve) >= 2:
                pts = curve.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
            else:
                x1, y1, x2, y2 = line
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness, lineType=cv2.LINE_AA)

    return frame


# ── Lane Fill Polygon ────────────────────────────────────────────────────────

def draw_lane_fill(frame, left_line, right_line, color=None, alpha=None):
    """Fill the area between the two lane lines with a semi-transparent polygon and horizontal stripes."""
    if frame is None or not isinstance(frame, np.ndarray):
        return frame
    if left_line is None or right_line is None:
        return frame

    if color is None:
        color = config.LANE_FILL_COLOR
    if alpha is None:
        alpha = config.LANE_FILL_ALPHA

    overlay = frame.copy()

    # Get curves for polygon fill, generating them if they are straight lines
    left_curve = getattr(left_line, 'curve', None)
    if left_curve is None:
        left_curve = _generate_straight_curve(left_line)

    right_curve = getattr(right_line, 'curve', None)
    if right_curve is None:
        right_curve = _generate_straight_curve(right_line)

    # Polygon: left curve (bottom to top) + right curve (top to bottom)
    pts = np.vstack((left_curve, right_curve[::-1]))
    pts = pts.astype(np.int32).reshape((-1, 1, 2))

    # 1. Solid fill with low background opacity
    cv2.fillPoly(overlay, [pts], color)

    # 2. Draw horizontal projected stripes inside the lane
    lx1, ly1, lx2, ly2 = left_line
    rx1, ry1, rx2, ry2 = right_line
    y_min = min(ly2, ry2)
    y_max = max(ly1, ry1)
    
    # We draw horizontal lines every 15 pixels
    for y in range(int(y_min), int(y_max), 15):
        x_left = get_x_at_y(left_line, y)
        x_right = get_x_at_y(right_line, y)
        # Draw horizontal stripe
        cv2.line(overlay, (x_left, y), (x_right, y), color, 4, cv2.LINE_AA)

    return cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0)


# ── HUD Info Overlay ────────────────────────────────────────────────────────

def draw_info_overlay(frame, curvature=None, offset=None,
                      warning_level="safe", fps=None,
                      left_line=None, right_line=None):
    """Draw a heads-up-display (HUD) with lane metrics onto the frame."""
    if frame is None or not isinstance(frame, np.ndarray):
        return frame

    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, w / 1600.0)
    line_thickness = max(1, int(w / 1000))
    color = _warning_color(warning_level)

    # Calculate padding and gaps dynamically to prevent text overlapping
    gap = max(28, int(42 * font_scale))
    y_offset = 35

    num_items = 4
    if fps is not None:
        num_items = 5

    # Semi-transparent background panel scaled to fit all text
    bg_width = max(260, int(w * 0.28))
    bg_height = y_offset + num_items * gap - 10
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + bg_width, 10 + bg_height),
                  (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

    # 1. Curvature
    if curvature is not None:
        if abs(curvature) > 5000:
            text = "Curvature: ~Straight"
        else:
            text = f"Curvature: {abs(curvature):.0f} m"
    else:
        text = "Curvature: N/A"
    cv2.putText(frame, text, (20, y_offset), font, font_scale,
                (255, 255, 255), line_thickness, cv2.LINE_AA)
    y_offset += gap

    # 2. Offset
    if offset is not None:
        direction = "Left" if offset < 0 else "Right"
        text = f"Offset: {abs(offset):.2f} ({direction})"
    else:
        text = "Offset: N/A"
    cv2.putText(frame, text, (20, y_offset), font, font_scale,
                (255, 255, 255), line_thickness, cv2.LINE_AA)
    y_offset += gap

    # 3. Warning level
    level_str = (warning_level or "safe").upper()
    cv2.putText(frame, f"Status: {level_str}", (20, y_offset), font,
                font_scale, color, line_thickness + 1, cv2.LINE_AA)
    y_offset += gap

    # 4. Lane Quality
    if left_line is not None and right_line is not None and (warning_level == "safe"):
        lane_text = "Lane: PERFECT"
        lane_color = (0, 200, 83) # Bright Green
    else:
        lane_text = "Lane: IMPERFECT"
        lane_color = (0, 109, 255) # Warning Orange/Red
        if warning_level == "danger":
            lane_color = (0, 0, 255) # Red
    cv2.putText(frame, lane_text, (20, y_offset), font,
                font_scale, lane_color, line_thickness + 1, cv2.LINE_AA)
    y_offset += gap

    # 5. FPS
    if fps is not None:
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, y_offset), font,
                    font_scale, (255, 255, 255), line_thickness, cv2.LINE_AA)

    return frame


# ── Combined Overlay ────────────────────────────────────────────────────────

def draw_lane_overlay(frame, left_line, right_line,
                      curvature=None, offset=None,
                      warning_level="safe", fps=None,
                      detected_cars=None):
    """Draw full lane overlay: fill + boundary lines + HUD + vehicle boxes."""
    if frame is None or not isinstance(frame, np.ndarray):
        return frame

    result = frame.copy()
    h, w = frame.shape[:2]
    y_hood = int(h * config.DASHBOARD_HOOD_Y_FRAC)

    # Clip lane lines to dashboard hood level for drawing
    clipped_left = _clip_line(left_line, y_hood)
    clipped_right = _clip_line(right_line, y_hood)

    # 1. Lane fill (needs both lines) using clipped coordinates
    result = draw_lane_fill(result, clipped_left, clipped_right)

    # 1.5. Draw lane centerline (yellow/cyan dashed line)
    result = draw_centerline(result, clipped_left, clipped_right, color=(0, 255, 255), thickness=2)

    # 2. Lane boundary lines using clipped coordinates
    line_color = _warning_color(warning_level)
    result = draw_lane_lines(result, clipped_left, clipped_right, color=line_color)

    # 3. Draw Detected Cars (Bounding Boxes with Red color)
    if detected_cars:
        for (x, y, w_box, h_box) in detected_cars:
            # Draw standard red rectangle around the vehicle
            cv2.rectangle(result, (x, y), (x + w_box, y + h_box), (0, 0, 255), 2)
            # Draw a semi-transparent red fill inside the box
            box_overlay = result.copy()
            cv2.rectangle(box_overlay, (x, y), (x + w_box, y + h_box), (0, 0, 255), -1)
            cv2.addWeighted(box_overlay, 0.15, result, 0.85, 0, result)
            # Draw label
            cv2.putText(result, "CAR", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

    # 4. HUD using clipped coordinates for quality determination
    result = draw_info_overlay(result, curvature=curvature, offset=offset,
                               warning_level=warning_level, fps=fps,
                               left_line=clipped_left, right_line=clipped_right)

    return result
