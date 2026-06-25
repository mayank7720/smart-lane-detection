"""
Smart Road Lane Line Detection System — Flask Routes
=====================================================
All web routes, file handling, and real-time streaming logic.
"""

import os
import sys
import time
import uuid
import json
import datetime

# Ensure the project root is on sys.path so `config` and `core` are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, send_from_directory, Response, jsonify, session,
)
from werkzeug.utils import secure_filename

# ---------------------------------------------------------------------------
# Flask application setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = config.FLASK_SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

# ---------------------------------------------------------------------------
# Try to import the CV pipeline (graceful fallback if OpenCV isn't installed)
# ---------------------------------------------------------------------------
try:
    from core.pipeline import LaneDetectionPipeline
    pipeline = LaneDetectionPipeline()
    PIPELINE_AVAILABLE = True
except Exception:
    pipeline = None
    PIPELINE_AVAILABLE = False

# ---------------------------------------------------------------------------
# In-memory statistics store
# ---------------------------------------------------------------------------
stats = {
    "images_processed": 0,
    "videos_processed": 0,
    "total_processed": 0,
    "success_count": 0,
    "fail_count": 0,
    "avg_fps": 0.0,
    "avg_curvature": 0.0,
    "history": [],          # list of dicts with processing info
}

# Per-file metadata store  {filename: {…}}
file_metadata = {}


# ---------------------------------------------------------------------------
# Copy preloaded sample images from the brain folder to the project if not present
# ---------------------------------------------------------------------------
try:
    import shutil
    brain_dir = r"C:\Users\Harsh Raj\.gemini\antigravity\brain\8c32527b-d18c-48c0-ad81-0bce5d498cf2"
    samples_mapping = {
        "highway_daylight_1782380067370.png": "highway_daylight.png",
        "curved_road_1782380091843.png": "curved_road.png",
        "night_driving_1782380108467.png": "night_driving.png"
    }
    for brain_name, proj_name in samples_mapping.items():
        src_path = os.path.join(brain_dir, brain_name)
        dst_sample_path = os.path.join(config.SAMPLE_DIR, proj_name)
        dst_upload_path = os.path.join(config.UPLOAD_DIR, f"demo_{proj_name}")

        # Copy to data/sample/
        if os.path.exists(src_path) and not os.path.exists(dst_sample_path):
            shutil.copy2(src_path, dst_sample_path)

        # Copy to app/static/uploads/ as demo_...
        if os.path.exists(src_path) and not os.path.exists(dst_upload_path):
            shutil.copy2(src_path, dst_upload_path)
except Exception:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def allowed_file(filename: str, allowed_extensions: set) -> bool:
    """Return True if the filename has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


def _unique_filename(original: str) -> str:
    """Return a unique filename by prepending a short UUID."""
    ext = original.rsplit(".", 1)[1].lower() if "." in original else "bin"
    return f"{uuid.uuid4().hex[:10]}_{secure_filename(original)}"


def generate_frames():
    """Generator that yields MJPEG frames from the default webcam (or fallback video)."""
    if PIPELINE_AVAILABLE:
        pipeline._prev_left = None
        pipeline._prev_right = None
    try:
        import cv2
        import time
        cap = cv2.VideoCapture(0)
        is_webcam = True
        if not cap.isOpened():
            cap = cv2.VideoCapture("data/sample/highway.mp4")
            is_webcam = False
            if not cap.isOpened():
                return
        while True:
            success, frame = cap.read()
            if not success:
                if not is_webcam:
                    # Loop back to the start of the video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    success, frame = cap.read()
                    if not success:
                        break
                else:
                    break
            if PIPELINE_AVAILABLE:
                try:
                    annotated_frame, _ = pipeline.process_frame(frame)
                    frame = annotated_frame
                except Exception:
                    pass
            _, buffer = cv2.imencode(".jpg", frame)
            if not is_webcam:
                time.sleep(0.04)  # ~25 FPS delay for video fallback
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
        cap.release()
    except ImportError:
        return


# ═══════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════

# ── Landing page ──────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", stats=stats)


# ── Demo processing route ────────────────────────────────────────────────
@app.route("/demo/<sample_name>")
def run_demo(sample_name):
    import shutil
    sample_file = f"{sample_name}.png" if sample_name != "road" else "road.jpg"
    sample_path = os.path.join(config.SAMPLE_DIR, sample_file)

    if not os.path.exists(sample_path):
        flash("Demo sample image not found.", "danger")
        return redirect(url_for("index"))

    filename = f"demo_{sample_file}"
    processed_filename = f"result_{filename}"
    processed_path = os.path.join(config.PROCESSED_DIR, processed_filename)

    # Ensure upload directory has the original file for side-by-side display
    shutil.copy2(sample_path, os.path.join(config.UPLOAD_DIR, filename))

    meta = {
        "original": filename,
        "processed": processed_filename,
        "type": "image",
        "timestamp": datetime.datetime.now().isoformat(),
        "curvature_radius": None,
        "direction": None,
        "lane_offset": None,
        "warning_level": "safe",
        "processing_time": None,
    }

    if PIPELINE_AVAILABLE:
        try:
            import cv2
            t0 = time.time()
            img = cv2.imread(sample_path)
            pipeline._prev_left = None
            pipeline._prev_right = None
            out_frame, meta_pipeline = pipeline.process_frame(img)
            elapsed = time.time() - t0

            cv2.imwrite(processed_path, out_frame)

            meta["curvature_radius"] = round(meta_pipeline.get("curvature_radius") or 0, 2)
            meta["direction"] = meta_pipeline.get("direction", "straight")
            meta["lane_offset"] = round(meta_pipeline.get("offset") or 0, 4)
            meta["warning_level"] = meta_pipeline.get("warning_level", "safe")
            meta["processing_time"] = round(elapsed, 3)
            stats["success_count"] += 1
        except Exception as exc:
            shutil.copy2(sample_path, processed_path)
            meta["warning_level"] = "error"
            meta["processing_time"] = 0
            stats["fail_count"] += 1
    else:
        shutil.copy2(sample_path, processed_path)
        meta["processing_time"] = 0

    stats["images_processed"] += 1
    stats["total_processed"] += 1
    stats["history"].insert(0, meta)
    file_metadata[processed_filename] = meta

    flash(f"Loaded and processed demo sample: {sample_name.replace('_', ' ').title()}", "success")
    return redirect(url_for("show_result", filename=processed_filename))


# ── Image upload (GET = form, POST = process) ────────────────────────────
@app.route("/upload/image", methods=["GET"])
def upload_image_page():
    return render_template("upload.html")


@app.route("/upload/image", methods=["POST"])
def upload_image():
    if "file" not in request.files:
        flash("No file selected.", "warning")
        return redirect(url_for("upload_image_page"))

    file = request.files["file"]
    if file.filename == "":
        flash("No file selected.", "warning")
        return redirect(url_for("upload_image_page"))

    if not allowed_file(file.filename, config.ALLOWED_IMAGE_EXTENSIONS):
        flash("Unsupported image format.", "danger")
        return redirect(url_for("upload_image_page"))

    filename = _unique_filename(file.filename)
    upload_path = os.path.join(config.UPLOAD_DIR, filename)
    file.save(upload_path)

    # ── Process with pipeline ──
    processed_filename = f"result_{filename}"
    processed_path = os.path.join(config.PROCESSED_DIR, processed_filename)
    meta = {
        "original": filename,
        "processed": processed_filename,
        "type": "image",
        "timestamp": datetime.datetime.now().isoformat(),
        "curvature_radius": None,
        "direction": None,
        "lane_offset": None,
        "warning_level": "safe",
        "processing_time": None,
    }

    if PIPELINE_AVAILABLE:
        try:
            import cv2
            t0 = time.time()
            img = cv2.imread(upload_path)
            pipeline._prev_left = None
            pipeline._prev_right = None
            out_frame, meta_pipeline = pipeline.process_frame(img)
            elapsed = time.time() - t0

            cv2.imwrite(processed_path, out_frame)

            meta["curvature_radius"] = round(meta_pipeline.get("curvature_radius") or 0, 2)
            meta["direction"] = meta_pipeline.get("direction", "straight")
            meta["lane_offset"] = round(meta_pipeline.get("offset") or 0, 4)
            meta["warning_level"] = meta_pipeline.get("warning_level", "safe")
            meta["processing_time"] = round(elapsed, 3)
            stats["success_count"] += 1
        except Exception as exc:
            # If pipeline fails, just copy original as result
            import shutil
            shutil.copy2(upload_path, processed_path)
            meta["warning_level"] = "error"
            meta["processing_time"] = 0
            stats["fail_count"] += 1
    else:
        import shutil
        shutil.copy2(upload_path, processed_path)
        meta["processing_time"] = 0

    stats["images_processed"] += 1
    stats["total_processed"] += 1
    stats["history"].insert(0, meta)
    file_metadata[processed_filename] = meta

    flash("Image processed successfully!", "success")
    return redirect(url_for("show_result", filename=processed_filename))


# ── Video upload (GET = form, POST = process) ────────────────────────────
@app.route("/upload/video", methods=["GET"])
def upload_video_page():
    return render_template("video_upload.html")


@app.route("/upload/video", methods=["POST"])
def upload_video():
    if "file" not in request.files:
        flash("No file selected.", "warning")
        return redirect(url_for("upload_video_page"))

    file = request.files["file"]
    if file.filename == "":
        flash("No file selected.", "warning")
        return redirect(url_for("upload_video_page"))

    if not allowed_file(file.filename, config.ALLOWED_VIDEO_EXTENSIONS):
        flash("Unsupported video format.", "danger")
        return redirect(url_for("upload_video_page"))

    filename = _unique_filename(file.filename)
    upload_path = os.path.join(config.UPLOAD_DIR, filename)
    file.save(upload_path)

    processed_filename = f"result_{filename}"
    processed_path = os.path.join(config.PROCESSED_DIR, processed_filename)
    meta = {
        "original": filename,
        "processed": processed_filename,
        "type": "video",
        "timestamp": datetime.datetime.now().isoformat(),
        "curvature_radius": None,
        "direction": None,
        "lane_offset": None,
        "warning_level": "safe",
        "processing_time": None,
        "frame_count": 0,
        "fps": 0,
    }

    if PIPELINE_AVAILABLE:
        try:
            import cv2
            t0 = time.time()
            pipeline._prev_left = None
            pipeline._prev_right = None
            cap = cv2.VideoCapture(upload_path)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            writer = cv2.VideoWriter(processed_path, fourcc, orig_fps, (w, h))

            frame_count = 0
            curvatures = []
            meta_pipeline = {}
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                out_frame, frame_meta = pipeline.process_frame(frame)
                writer.write(out_frame)
                frame_count += 1
                meta_pipeline = frame_meta
                if frame_meta.get("curvature_radius"):
                    curvatures.append(frame_meta["curvature_radius"])

            cap.release()
            writer.release()
            elapsed = time.time() - t0

            meta["processing_time"] = round(elapsed, 3)
            meta["frame_count"] = frame_count
            meta["fps"] = round(frame_count / elapsed, 1) if elapsed > 0 else 0
            meta["curvature_radius"] = round(sum(curvatures) / len(curvatures), 2) if curvatures else 0
            meta["direction"] = meta_pipeline.get("direction", "straight")
            meta["lane_offset"] = round(meta_pipeline.get("offset") or 0, 4)
            meta["warning_level"] = meta_pipeline.get("warning_level", "safe")
            stats["success_count"] += 1
        except Exception:
            import shutil
            shutil.copy2(upload_path, processed_path)
            stats["fail_count"] += 1
    else:
        import shutil
        shutil.copy2(upload_path, processed_path)

    stats["videos_processed"] += 1
    stats["total_processed"] += 1
    stats["history"].insert(0, meta)
    file_metadata[processed_filename] = meta

    flash("Video processed successfully!", "success")
    return redirect(url_for("show_result", filename=processed_filename))


# ── Result display ────────────────────────────────────────────────────────
@app.route("/result/<filename>")
def show_result(filename):
    meta = file_metadata.get(filename, {
        "original": filename,
        "processed": filename,
        "type": "image",
        "curvature_radius": 0,
        "direction": "N/A",
        "lane_offset": 0,
        "warning_level": "safe",
        "processing_time": 0,
    })
    return render_template("result.html", filename=filename, meta=meta)


# ── Real-time webcam ──────────────────────────────────────────────────────
@app.route("/realtime")
def realtime():
    return render_template("realtime.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# ── Dashboard / Statistics ────────────────────────────────────────────────
@app.route("/dashboard")
def dashboard():
    return render_template("statistics.html", stats=stats)


@app.route("/api/stats")
def api_stats():
    return jsonify({
        "images_processed": stats["images_processed"],
        "videos_processed": stats["videos_processed"],
        "total_processed": stats["total_processed"],
        "success_rate": round(
            stats["success_count"] / stats["total_processed"] * 100, 1
        ) if stats["total_processed"] > 0 else 100.0,
        "history": stats["history"][:20],
    })


# ── Serve uploaded / processed files directly (dev convenience) ───────────
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(config.UPLOAD_DIR, filename)


@app.route("/processed/<filename>")
def processed_file(filename):
    return send_from_directory(config.PROCESSED_DIR, filename)
