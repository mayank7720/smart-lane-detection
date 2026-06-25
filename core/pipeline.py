"""
Smart Lane Detection — Full Pipeline
=====================================
Orchestrates every processing stage — from raw frame through
preprocessing, ROI masking, lane detection, curvature estimation,
departure warning, and overlay drawing — into a single reusable class.
Supports single images, video files, and live webcam feeds.
"""

import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config

from core.preprocessor import preprocess
from core.roi import apply_roi
from core.lane_detector import detect_lanes
from core.curvature import estimate_curvature, determine_direction
from core.departure_warning import calculate_offset, check_departure
from core.overlay import draw_lane_overlay, get_x_at_y


class LaneDetectionPipeline:
    """End-to-end lane detection pipeline with temporal smoothing.

    Attributes
    ----------
    _prev_left : tuple or None
        Smoothed left-lane line from the previous frame.
    _prev_right : tuple or None
        Smoothed right-lane line from the previous frame.
    _alpha : float
        Exponential-moving-average smoothing factor.
    """

    def __init__(self):
        self._prev_left = None
        self._prev_right = None
        self._alpha = config.LANE_SMOOTHING_FACTOR

        # Load vehicle classifier
        cascade_path = os.path.join(os.path.dirname(__file__), "cars.xml")
        if os.path.exists(cascade_path):
            self.car_cascade = cv2.CascadeClassifier(cascade_path)
        else:
            self.car_cascade = None

    # ── Temporal Smoothing ───────────────────────────────────────────────

    def _generate_straight_curve(self, line, num_pts=50):
        """Generate (x, y) coordinates along a straight line segment."""
        x1, y1, x2, y2 = line
        y_vals = np.linspace(y1, y2, num=num_pts)
        if abs(y2 - y1) < 1:
            x_vals = np.full_like(y_vals, (x1 + x2) / 2.0)
        else:
            slope = (x2 - x1) / (y2 - y1)
            x_vals = x1 + slope * (y_vals - y1)
        return np.column_stack((x_vals, y_vals))

    def _smooth_line(self, new_line, prev_line, alpha):
        """Smooth a lane line using exponential moving average, handling curves.

        Parameters
        ----------
        new_line : tuple or None
            Newly detected ``(x1, y1, x2, y2)``.
        prev_line : tuple or None
            Previous smoothed line.
        alpha : float
            Weight for the *previous* line (``0`` = no smoothing,
            ``1`` = ignore new detection entirely).

        Returns
        -------
        LaneTuple or None
            Smoothed LaneTuple or ``None``.
        """
        if new_line is None:
            return prev_line  # keep the old estimate
        if prev_line is None:
            return new_line   # first detection — nothing to smooth with

        # Unpack coords
        x1_n, y1_n, x2_n, y2_n = new_line
        x1_p, y1_p, x2_p, y2_p = prev_line

        # Smooth the 4-tuple coords
        x1 = int(alpha * x1_p + (1 - alpha) * x1_n)
        y1 = int(alpha * y1_p + (1 - alpha) * y1_n)
        x2 = int(alpha * x2_p + (1 - alpha) * x2_n)
        y2 = int(alpha * y2_p + (1 - alpha) * y2_n)

        # Handle curve smoothing
        new_curve = getattr(new_line, 'curve', None)
        prev_curve = getattr(prev_line, 'curve', None)

        if new_curve is None and prev_curve is None:
            smoothed_curve = None
        else:
            # If one is None, generate straight curve for it
            if new_curve is None:
                new_curve = self._generate_straight_curve(new_line)
            if prev_curve is None:
                prev_curve = self._generate_straight_curve(prev_line)

            # Resample prev_curve to match new_curve's y values if they differ in length
            if len(new_curve) != len(prev_curve):
                ys_new = new_curve[:, 1]
                ys_prev = prev_curve[:, 1]
                xs_prev = prev_curve[:, 0]
                sort_idx = np.argsort(ys_prev)
                xs_prev_interp = np.interp(ys_new, ys_prev[sort_idx], xs_prev[sort_idx])
                prev_curve = np.column_stack((xs_prev_interp, ys_new))

            smoothed_curve = alpha * prev_curve + (1 - alpha) * new_curve

            # Check if the smoothed curve has significant deviation
            try:
                xs = smoothed_curve[:, 0]
                ys = smoothed_curve[:, 1]
                coeffs = np.polyfit(ys, xs, 2)
                y_bottom = max(y1, y2)
                y_top = min(y1, y2)
                deviation = abs(coeffs[0]) * (y_bottom - y_top)**2 / 8.0
                if deviation < 3.0:
                    smoothed_curve = None
            except Exception:
                pass

        from core.lane_detector import LaneTuple
        return LaneTuple(x1, y1, x2, y2, curve=smoothed_curve)

    def get_intersection_y(self, left_line, right_line, y_top, y_bottom):
        """Find the y-coordinate where left and right lines intersect, if any."""
        # Check from bottom to top
        for y in np.linspace(y_bottom, y_top, num=100):
            x_left = get_x_at_y(left_line, y)
            x_right = get_x_at_y(right_line, y)
            if x_left >= x_right:
                return int(y)
        return None

    def clip_line_top(self, line, y_top_new):
        """Clip the top of a line/curve to a new y_top coordinate."""
        if line is None:
            return None
        x1, y1, x2, y2 = line
        curve = getattr(line, 'curve', None)
        
        # Calculate new top x-coordinate
        x_top_new = get_x_at_y(line, y_top_new)
        new_coords = (x1, y1, x_top_new, y_top_new)
        
        new_curve = None
        if curve is not None:
            # Keep points with y >= y_top_new (y decreases as we go up in image coordinates)
            mask = curve[:, 1] >= y_top_new
            filtered = curve[mask]
            
            # Append the exact intersection point at the top
            if len(filtered) > 0:
                if filtered[-1, 1] != y_top_new:
                    top_pt = np.array([[x_top_new, y_top_new]])
                    new_curve = np.vstack((filtered, top_pt))
                else:
                    new_curve = filtered
            else:
                new_curve = None
                
        from core.lane_detector import LaneTuple
        return LaneTuple(*new_coords, curve=new_curve)

    def validate_detection(self, left_line, right_line, w, h):
        """Validate left and right lane boundaries to prevent intersections, crossing,
        and ensure realistic lane width constraints.
        """
        if left_line is None or right_line is None:
            # Accept single line detections as valid
            return True

        y_bottom = int(h * config.LANE_EXTRAPOLATE_Y_BOTTOM_FRAC)
        y_top = int(h * config.LANE_EXTRAPOLATE_Y_TOP_FRAC)

        # Check for intersection. If they intersect close to the bottom, it's invalid!
        y_intersect = self.get_intersection_y(left_line, right_line, y_top, y_bottom)
        if y_intersect is not None:
            if y_intersect > y_bottom - 120:
                return False

        # Validate lane width at bottom (drivable roadway width)
        x_left_b = get_x_at_y(left_line, y_bottom)
        x_right_b = get_x_at_y(right_line, y_bottom)
        width_b = x_right_b - x_left_b
        if width_b < 0.15 * w or width_b > 1.15 * w:
            return False

        # If they don't intersect, validate width at the top
        if y_intersect is None:
            x_left_t = get_x_at_y(left_line, y_top)
            x_right_t = get_x_at_y(right_line, y_top)
            width_t = x_right_t - x_left_t
            if width_t < 0.01 * w or width_t > 0.45 * w:
                return False

        return True

    def apply_nms(self, boxes, overlap_thresh=0.3):
        """Apply non-maximum suppression (NMS) to eliminate overlapping and nested boxes."""
        if not boxes:
            return []

        boxes = np.array(boxes)
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]
        x2 = x1 + w
        y2 = y1 + h

        areas = w * h
        order = np.argsort(y2)  # sort by bottom coordinate to prioritize closer vehicles

        keep = []
        while len(order) > 0:
            i = order[-1]
            keep.append(i)

            if len(order) == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[:-1]])
            yy1 = np.maximum(y1[i], y1[order[:-1]])
            xx2 = np.minimum(x2[i], x2[order[:-1]])
            yy2 = np.minimum(y2[i], y2[order[:-1]])

            ww = np.maximum(0.0, xx2 - xx1)
            hh = np.maximum(0.0, yy2 - yy1)
            inter = ww * hh

            # Intersection over Union (IoU)
            union = areas[i] + areas[order[:-1]] - inter
            iou = inter / union

            # Keep boxes that do not overlap significantly with box i
            inds = np.where(iou <= overlap_thresh)[0]
            order = order[inds]

        return boxes[keep].tolist()

    # ── Single-Frame Processing ──────────────────────────────────────────

    def process_frame(self, frame):
        """Run the full pipeline on a single BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            Input BGR image.

        Returns
        -------
        tuple[np.ndarray, dict]
            ``(annotated_frame, metadata)`` where *metadata* contains:

            - ``left_line`` / ``right_line``: smoothed lane tuples
            - ``curvature_radius``: float or None
            - ``direction``: str
            - ``offset``: float or None
            - ``warning_level``, ``warning_message``: str
            - ``detection_success``: bool
        """
        metadata = {
            "left_line": None,
            "right_line": None,
            "curvature_radius": None,
            "direction": "Unknown",
            "offset": None,
            "warning_level": "safe",
            "warning_message": "",
            "detection_success": False,
        }

        if frame is None or not isinstance(frame, np.ndarray):
            return frame, metadata

        # 1. Preprocess
        edges, _ = preprocess(frame)
        if edges is None:
            return frame.copy(), metadata

        # 2. Region of Interest
        roi_edges = apply_roi(edges)
        if roi_edges is None:
            return frame.copy(), metadata

        # 3. Lane detection
        left_line, right_line = detect_lanes(roi_edges, frame.shape)

        # Validate raw detection
        h, w = frame.shape[:2]
        is_valid = self.validate_detection(left_line, right_line, w, h)
        if not is_valid:
            # Fallback to previous valid frame prediction
            left_line = self._prev_left
            right_line = self._prev_right
        else:
            # If they intersect near the horizon, adjust y_top to the intersection point
            if left_line is not None and right_line is not None:
                y_bottom = int(h * config.LANE_EXTRAPOLATE_Y_BOTTOM_FRAC)
                y_top = int(h * config.LANE_EXTRAPOLATE_Y_TOP_FRAC)
                y_intersect = self.get_intersection_y(left_line, right_line, y_top, y_bottom)
                if y_intersect is not None:
                    left_line = self.clip_line_top(left_line, y_intersect)
                    right_line = self.clip_line_top(right_line, y_intersect)

        # 4. Temporal smoothing
        left_line = self._smooth_line(left_line, self._prev_left, self._alpha)
        right_line = self._smooth_line(right_line, self._prev_right, self._alpha)
        self._prev_left = left_line
        self._prev_right = right_line

        detection_success = left_line is not None or right_line is not None
        metadata["left_line"] = left_line
        metadata["right_line"] = right_line
        metadata["detection_success"] = detection_success

        # 5. Curvature estimation
        curvature = estimate_curvature(left_line, right_line, frame.shape)
        direction = determine_direction(curvature)
        metadata["curvature_radius"] = curvature
        metadata["direction"] = direction

        # 6. Lane departure warning
        offset = calculate_offset(left_line, right_line, w)
        departure = check_departure(offset)
        metadata["offset"] = departure["offset"]
        metadata["warning_level"] = departure["level"]
        metadata["warning_message"] = departure["message"]

        # 6.5. Vehicle detection (with performance downscaling, spatial road ROI filtering, and NMS)
        detected_cars = []
        if self.car_cascade is not None:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Downscale to 360px height to speed up detection significantly
                scale = 360.0 / h
                gray_small = cv2.resize(gray, (int(w * scale), 360))
                # Detect cars
                cars = self.car_cascade.detectMultiScale(gray_small, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20))
                
                y_hood = int(h * config.DASHBOARD_HOOD_Y_FRAC)
                raw_cars = []
                
                for (x_box, y_box, w_box, h_box) in cars:
                    x_orig = int(x_box / scale)
                    y_orig = int(y_box / scale)
                    w_orig = int(w_box / scale)
                    h_orig = int(h_box / scale)
                    
                    # Calculate bottom-center coordinate of the bounding box
                    y_bottom_box = y_orig + h_orig
                    x_center_box = x_orig + w_orig / 2.0
                    
                    # 1. Ignore detections below the dashboard hood level (windshield devices/reflections)
                    if y_bottom_box > y_hood:
                        continue
                        
                    # 2. Restrict to the drivable road region plus a buffer
                    # Calculate road width limits at y_bottom_box using lane coordinates
                    if left_line is not None and right_line is not None:
                        x_l = get_x_at_y(left_line, y_bottom_box)
                        x_r = get_x_at_y(right_line, y_bottom_box)
                        lane_w = x_r - x_l
                        # Allow cars to be in current and adjacent lanes (1.2 * lane_width buffer)
                        road_l = x_l - 1.2 * lane_w
                        road_r = x_r + 1.2 * lane_w
                    else:
                        # Fallback road region
                        road_l = 0.1 * w
                        road_r = 0.9 * w
                        
                    # 3. Reject if outside road region or in the sky (above 0.40*h)
                    if (road_l <= x_center_box <= road_r) and (y_orig >= 0.40 * h):
                        raw_cars.append((x_orig, y_orig, w_orig, h_orig))
                        
                # 4. Apply Non-Maximum Suppression (NMS) to remove overlapping detections
                detected_cars = self.apply_nms(raw_cars, overlap_thresh=0.3)
            except Exception:
                pass

        # 7. Overlay
        annotated = draw_lane_overlay(
            frame, left_line, right_line,
            curvature=curvature,
            offset=offset,
            warning_level=departure["level"],
            detected_cars=detected_cars,
        )

        return annotated, metadata

    # ── Image Processing ─────────────────────────────────────────────────

    def process_image(self, input_path, output_path=None):
        """Load a single image, run the pipeline, and optionally save.

        Parameters
        ----------
        input_path : str
            Path to the source image.
        output_path : str or None
            If given, the annotated image is saved here.

        Returns
        -------
        tuple[np.ndarray, dict] or (None, dict)
            ``(result_frame, metadata)``.
        """
        if not os.path.isfile(input_path):
            return None, {"error": f"File not found: {input_path}",
                          "detection_success": False}

        frame = cv2.imread(input_path)
        if frame is None:
            return None, {"error": f"Failed to read image: {input_path}",
                          "detection_success": False}

        # Reset smoothing state for a single image
        self._prev_left = None
        self._prev_right = None

        result, metadata = self.process_frame(frame)

        if output_path is not None and result is not None:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            cv2.imwrite(output_path, result)

        return result, metadata

    # ── Video Processing ─────────────────────────────────────────────────

    def process_video(self, input_path, output_path=None):
        """Process every frame of a video file.

        Parameters
        ----------
        input_path : str
            Path to the source video.
        output_path : str or None
            If given, the annotated video is written here.

        Returns
        -------
        list[dict]
            Per-frame metadata list.
        """
        if not os.path.isfile(input_path):
            return [{"error": f"File not found: {input_path}",
                     "detection_success": False}]

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return [{"error": f"Cannot open video: {input_path}",
                     "detection_success": False}]

        # Reset smoothing for a new video
        self._prev_left = None
        self._prev_right = None

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = None
        if output_path is not None:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        all_metadata = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.perf_counter()
            annotated, meta = self.process_frame(frame)
            elapsed = time.perf_counter() - t0

            meta["frame_index"] = frame_idx
            meta["processing_time"] = elapsed
            meta["fps"] = 1.0 / elapsed if elapsed > 0 else 0.0
            all_metadata.append(meta)

            if writer is not None and annotated is not None:
                writer.write(annotated)

            frame_idx += 1

        cap.release()
        if writer is not None:
            writer.release()

        return all_metadata

    # ── Live Webcam ──────────────────────────────────────────────────────

    def run_webcam(self):
        """Run real-time lane detection on the default webcam.

        Press **q** to quit.
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[LaneDetectionPipeline] Cannot open webcam.")
            return

        self._prev_left = None
        self._prev_right = None

        print("[LaneDetectionPipeline] Webcam started. Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.perf_counter()
            annotated, meta = self.process_frame(frame)
            elapsed = time.perf_counter() - t0
            fps = 1.0 / elapsed if elapsed > 0 else 0.0

            # Re-draw overlay with live FPS
            annotated = draw_lane_overlay(
                frame, meta["left_line"], meta["right_line"],
                curvature=meta["curvature_radius"],
                offset=meta["offset"],
                warning_level=meta["warning_level"],
                fps=fps,
            )

            cv2.imshow("Lane Detection — Press Q to quit", annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
