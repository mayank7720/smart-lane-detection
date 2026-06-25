"""
Smart Lane Detection — Image Preprocessor
==========================================
Functions for converting frames to grayscale, applying Gaussian blur,
CLAHE adaptive histogram equalization, and Canny edge detection.
The `preprocess` function chains all steps into a single pipeline call.
"""

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config


def to_grayscale(frame):
    """Convert a BGR frame to grayscale.

    Parameters
    ----------
    frame : np.ndarray
        Input BGR image (H×W×3).

    Returns
    -------
    np.ndarray or None
        Single-channel grayscale image (H×W), or None if input is invalid.
    """
    if frame is None or not isinstance(frame, np.ndarray):
        return None
    # If already single-channel, return as-is
    if len(frame.shape) == 2:
        return frame.copy()
    if len(frame.shape) == 3 and frame.shape[2] == 1:
        return frame[:, :, 0].copy()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def apply_blur(frame, kernel_size=None):
    """Apply Gaussian blur to reduce noise.

    Parameters
    ----------
    frame : np.ndarray
        Input image (grayscale or color).
    kernel_size : int, optional
        Size of the Gaussian kernel (must be odd). Falls back to
        ``config.GAUSSIAN_KERNEL_SIZE``.

    Returns
    -------
    np.ndarray or None
        Blurred image, or None if input is invalid.
    """
    if frame is None or not isinstance(frame, np.ndarray):
        return None
    if kernel_size is None:
        kernel_size = config.GAUSSIAN_KERNEL_SIZE
    # Ensure kernel size is odd and positive
    kernel_size = max(1, kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)


def detect_edges(frame, low=None, high=None):
    """Detect edges using the Canny algorithm.

    Parameters
    ----------
    frame : np.ndarray
        Input single-channel image (typically blurred grayscale).
    low : int, optional
        Lower hysteresis threshold. Defaults to ``config.CANNY_LOW_THRESHOLD``.
    high : int, optional
        Upper hysteresis threshold. Defaults to ``config.CANNY_HIGH_THRESHOLD``.

    Returns
    -------
    np.ndarray or None
        Binary edge map, or None if input is invalid.
    """
    if frame is None or not isinstance(frame, np.ndarray):
        return None
    if low is None:
        low = config.CANNY_LOW_THRESHOLD
    if high is None:
        high = config.CANNY_HIGH_THRESHOLD
    return cv2.Canny(frame, low, high)


def apply_clahe(frame):
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).

    CLAHE improves contrast in images with varying lighting conditions by
    operating on small tiles rather than the whole image.

    Parameters
    ----------
    frame : np.ndarray
        Single-channel grayscale image.

    Returns
    -------
    np.ndarray or None
        Enhanced grayscale image, or None if input is invalid.
    """
    if frame is None or not isinstance(frame, np.ndarray):
        return None
    # Ensure single-channel input
    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(
        clipLimit=config.CLAHE_CLIP_LIMIT,
        tileGridSize=config.CLAHE_TILE_SIZE,
    )
    return clahe.apply(frame)


def select_yellow_white(frame):
    """Create a mask to select white and yellow lane lines based on color thresholds.

    Parameters
    ----------
    frame : np.ndarray
        Input BGR image.

    Returns
    -------
    np.ndarray
        Binary mask selecting yellow and white pixels.
    """
    # Classify night driving: if average brightness is low, disable color mask to avoid headlight beam edges
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    if mean_brightness < 55.0:
        return np.zeros(gray.shape, dtype=np.uint8)

    # Convert BGR to HSV for yellow selection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([15, 50, 80], dtype=np.uint8)
    upper_yellow = np.array([35, 255, 255], dtype=np.uint8)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Convert BGR to HLS for white selection (L channel represents lightness)
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    lower_white = np.array([0, 190, 0], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)
    mask_white = cv2.inRange(hls, lower_white, upper_white)
    
    return cv2.bitwise_or(mask_yellow, mask_white)


def preprocess(frame):
    """Full preprocessing pipeline: grayscale → Adaptive CLAHE → blur → Adaptive Canny + Color Mask edge boosting.

    Parameters
    ----------
    frame : np.ndarray
        Input BGR image (H×W×3).

    Returns
    -------
    tuple[np.ndarray, np.ndarray] or (None, None)
        ``(edges, enhanced_gray)`` where *edges* is the boosted Canny edge map
        and *enhanced_gray* is the CLAHE-enhanced grayscale image.
    """
    if frame is None or not isinstance(frame, np.ndarray):
        return None, None

    gray = to_grayscale(frame)
    if gray is None:
        return None, None

    # Calculate average brightness to adaptively tune preprocessor parameters for day vs night
    mean_brightness = np.mean(gray)
    is_night = mean_brightness < 55.0

    # 1. Contrast limited adaptive histogram equalization (CLAHE)
    if is_night:
        # Boost low-contrast details under headlight shadow regions at night
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=config.CLAHE_TILE_SIZE)
        enhanced = clahe.apply(gray)
    else:
        enhanced = apply_clahe(gray)

    if enhanced is None:
        return None, None

    # 2. Gaussian Blur
    blurred = apply_blur(enhanced)
    if blurred is None:
        return None, None

    # 3. Canny Edge Detection
    if is_night:
        # Lower thresholds to pick up faint lane boundaries under shadows
        edges = detect_edges(blurred, low=40, high=100)
    else:
        edges = detect_edges(blurred)

    if edges is None:
        return None, None

    # 4. White & Yellow Color Edge Boosting (Daytime only)
    if not is_night:
        try:
            color_mask = select_yellow_white(frame)
            color_mask_blurred = cv2.GaussianBlur(color_mask, (5, 5), 0)
            color_edges = cv2.Canny(color_mask_blurred, 50, 150)
            edges = cv2.bitwise_or(edges, color_edges)
        except Exception:
            pass

    return edges, enhanced
