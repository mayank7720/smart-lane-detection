"""
Smart Road Lane Line Detection System — Global Configuration
=============================================================
All tunable parameters and constants for the CV pipeline, 
Flask app, and evaluation modules.
"""

import os

# ─── Project Paths ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
UPLOAD_DIR = os.path.join(BASE_DIR, "app", "static", "uploads")
PROCESSED_DIR = os.path.join(BASE_DIR, "app", "static", "processed")
SAMPLE_DIR = os.path.join(DATA_DIR, "sample")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")

# Ensure directories exist
for d in [DATA_DIR, OUTPUT_DIR, UPLOAD_DIR, PROCESSED_DIR, SAMPLE_DIR,
          REPORT_DIR, os.path.join(OUTPUT_DIR, "images"),
          os.path.join(OUTPUT_DIR, "videos")]:
    os.makedirs(d, exist_ok=True)


# ─── Image Preprocessing ──────────────────────────────────────────────────────
GAUSSIAN_KERNEL_SIZE = 5          # Must be odd
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = (8, 8)


# ─── Region of Interest ───────────────────────────────────────────────────────
# ROI vertices as fractions of (width, height) — trapezoidal mask
# Format: [(x_frac, y_frac), ...] counter-clockwise from bottom-left
ROI_VERTICES_FRAC = [
    (0.05, 0.72),    # Bottom-left (at hood level to exclude dashboard)
    (0.40, 0.60),    # Top-left
    (0.60, 0.60),    # Top-right
    (0.95, 0.72),    # Bottom-right (at hood level to exclude dashboard)
]


# ─── Hough Line Transform ─────────────────────────────────────────────────────
HOUGH_RHO = 2                    # Distance resolution in pixels
HOUGH_THETA_DEGREES = 1          # Angle resolution in degrees
HOUGH_THRESHOLD = 50             # Minimum votes
HOUGH_MIN_LINE_LENGTH = 40       # Minimum line length in pixels
HOUGH_MAX_LINE_GAP = 150         # Maximum gap between segments


# ─── Lane Classification ──────────────────────────────────────────────────────
MIN_SLOPE = 0.3                  # Reject near-horizontal lines
MAX_SLOPE = 10.0                 # Reject near-vertical lines
LANE_EXTRAPOLATE_Y_TOP_FRAC = 0.60   # Extrapolate lanes up to this y-fraction
LANE_EXTRAPOLATE_Y_BOTTOM_FRAC = 0.72 # Extrapolate lanes down to the hood level
DASHBOARD_HOOD_Y_FRAC = 0.72          # Visual cutoff fraction for drawing overlays to exclude the hood
LANE_SMOOTHING_FACTOR = 0.7     # Exponential moving average alpha (0 = no smoothing, 1 = full history)


# ─── Lane Overlay Drawing ─────────────────────────────────────────────────────
LANE_LINE_COLOR = (0, 255, 0)    # Green (BGR)
LANE_LINE_THICKNESS = 8
LANE_FILL_COLOR = (180, 0, 180)  # Purple/Violet fill (BGR)
LANE_FILL_ALPHA = 0.35           # Transparency for lane fill polygon
WARNING_COLOR_SAFE = (0, 200, 0)       # Green
WARNING_COLOR_CAUTION = (0, 200, 255)  # Yellow/Orange
WARNING_COLOR_DANGER = (0, 0, 255)     # Red


# ─── Lane Departure Warning ───────────────────────────────────────────────────
DEPARTURE_THRESHOLD_CAUTION = 0.15   # Offset fraction for caution
DEPARTURE_THRESHOLD_DANGER = 0.30    # Offset fraction for danger


# ─── Curvature Estimation ─────────────────────────────────────────────────────
CURVATURE_POLY_DEGREE = 2         # Degree of polynomial for curve fitting
# Pixel-to-meter conversion factors (approximate for highway driving)
METERS_PER_PIXEL_Y = 30.0 / 720  # ~30 meters visible in 720px height
METERS_PER_PIXEL_X = 3.7 / 700   # ~3.7m lane width in ~700px


# ─── Flask App ─────────────────────────────────────────────────────────────────
FLASK_SECRET_KEY = "smart-lane-detection-2026"
FLASK_DEBUG = True
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
MAX_CONTENT_LENGTH = 100 * 1024 * 1024   # 100 MB max upload
ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff"}
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}


# ─── Evaluation ────────────────────────────────────────────────────────────────
BENCHMARK_BRIGHTNESS_FACTORS = [0.5, 0.75, 1.0, 1.25, 1.5]
BENCHMARK_NOISE_LEVELS = [0, 10, 25, 50]
TARGET_FPS = 15  # Minimum acceptable FPS


# ─── Color Palette (for UI reference) ─────────────────────────────────────────
COLORS = {
    "primary_green": "#00C853",
    "primary_dark": "#009624",
    "accent_green": "#69F0AE",
    "white": "#FFFFFF",
    "off_white": "#F5F7FA",
    "dark_bg": "#0A1628",
    "dark_card": "#122240",
    "dark_surface": "#1A2D50",
    "text_primary": "#FFFFFF",
    "text_secondary": "#94A3B8",
    "success": "#00E676",
    "warning": "#FFD600",
    "danger": "#FF1744",
    "info": "#00B0FF",
}
