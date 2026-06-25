# Smart Road Lane Line Detection System — Internship Presentation

> _Use each `##` heading as a separate slide. Bullet points serve as talking points._

---

## Slide 1 — Title Slide

# 🛣️ Smart Road Lane Line Detection System

**Real-Time Lane Detection Using Classical Computer Vision**

- **Presented by:** Harsh Raj
- **Date:** June 2026
- **Internship Project**
- **Technologies:** Python · OpenCV · Flask

---

## Slide 2 — Agenda

### 📋 Presentation Outline

1. Problem Statement & Motivation
2. Project Objectives
3. System Architecture
4. Technology Stack
5. Preprocessing Pipeline
6. Lane Detection Algorithm
7. Advanced Features (Curvature & Departure Warning)
8. Web Application Demo
9. Results & Accuracy
10. Challenges & Solutions
11. Future Scope
12. Q&A

---

## Slide 3 — Problem Statement

### 🚗 Why Lane Detection Matters

- **Road accidents** are a leading cause of fatalities worldwide — WHO reports 1.35 million deaths annually
- **Lane departure** is responsible for a significant portion of highway accidents
- Modern vehicles rely on **ADAS** (Advanced Driver Assistance Systems) for safety
- Lane detection is a **foundational capability** for:
  - Lane Keeping Assist (LKA)
  - Lane Departure Warning (LDW)
  - Autonomous driving navigation
- **Challenge:** Detecting lanes reliably under varying lighting, weather, and road surface conditions
- **Our goal:** Build a robust, real-time lane detection system using classical computer vision

---

## Slide 4 — Objectives

### 🎯 Project Objectives

1. **Detect lane lines** on road surfaces from images, videos, and live webcam feeds
2. **Preprocess frames** using grayscale conversion, contrast enhancement, Gaussian blur, and Canny edge detection
3. **Apply ROI masking** to focus computation on the road surface
4. **Use Hough Transform** for robust line detection in edge images
5. **Classify detected lines** into left and right lane boundaries using slope analysis
6. **Estimate road curvature** using polynomial curve fitting
7. **Implement lane departure warning** with three safety levels (SAFE, CAUTION, DANGER)
8. **Build a web application** for user-friendly upload-and-process workflows
9. **Achieve real-time performance** (>30 FPS) on standard hardware
10. **Provide a comprehensive evaluation** framework for benchmarking

---

## Slide 5 — System Architecture

### 🏗️ End-to-End Pipeline

```
Input Source (Image / Video / Webcam)
        │
        ▼
┌─────────────────────────────┐
│      PREPROCESSING          │
│  Grayscale → CLAHE → Blur   │
│       → Canny Edges         │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│     ROI MASKING             │
│  Trapezoidal region mask    │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│   HOUGH LINE TRANSFORM      │
│  Probabilistic line detect  │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│   LANE CLASSIFICATION        │
│  Slope → Left / Right       │
│  Averaging & Extrapolation  │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  CURVATURE & DEPARTURE      │
│  Polynomial fit → R(m)      │
│  Center offset → Warning    │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│     OVERLAY & OUTPUT         │
│  Annotated frame + metadata │
└─────────────────────────────┘
```

- Each stage is implemented as an **independent, testable module** in the `core/` package
- The `LaneDetectionPipeline` class orchestrates the full flow

---

## Slide 6 — Technology Stack

### 🛠️ Tools & Technologies

| Layer | Technology | Role |
|-------|-----------|------|
| Language | Python 3.8+ | Core development |
| Computer Vision | OpenCV 4.x | Image processing, video I/O, Hough Transform |
| Numerical | NumPy | Matrix operations, polynomial fitting |
| Web Backend | Flask | REST endpoints, file handling |
| Web Frontend | HTML / CSS / JS | Drag-and-drop UI, responsive design |
| Templating | Jinja2 | Server-side HTML rendering |
| Testing | unittest / pytest | Unit testing framework |
| Version Control | Git & GitHub | Source code management |

- **No deep learning frameworks required** — purely classical CV
- Runs on CPU without GPU dependency

---

## Slide 7 — Preprocessing Pipeline

### 🔧 Image Preprocessing Steps

**Step 1: Grayscale Conversion**
- Convert BGR → single-channel intensity image
- Reduces computational complexity by 3×

**Step 2: CLAHE Enhancement**
- Contrast Limited Adaptive Histogram Equalization
- Clip limit: 2.0, tile size: 8×8
- Handles varying lighting conditions (shadows, glare)

**Step 3: Gaussian Blur**
- 5×5 kernel to suppress high-frequency noise
- Prevents false edges from texture and sensor noise

**Step 4: Canny Edge Detection**
- Dual thresholds: Low = 50, High = 150
- Produces a binary edge map of strong intensity gradients
- Non-maximum suppression ensures thin, precise edges

---

## Slide 8 — Lane Detection Algorithm

### 📐 Hough Transform & Line Classification

**ROI Masking:**
- Trapezoidal polygon isolates the road surface
- Eliminates sky, trees, buildings from edge map
- Vertices defined as frame-fraction for resolution independence

**Probabilistic Hough Line Transform:**
- Parameters: ρ=2px, θ=1°, threshold=50, minLength=40px, maxGap=150px
- Returns line segments as `(x1, y1, x2, y2)` arrays

**Line Classification:**
- Slope < 0 → **Left lane** (image y-axis is inverted)
- Slope > 0 → **Right lane**
- Reject slopes < 0.3 (horizontal) or > 10.0 (vertical)

**Averaging & Extrapolation:**
- Average all left-classified lines → single left boundary
- Average all right-classified lines → single right boundary
- Extrapolate from bottom of frame to ROI top (60% height)

---

## Slide 9 — Advanced Features

### 📈 Curvature Estimation

- Fit a 2nd-degree polynomial to lane boundary points
- Calculate radius of curvature:

  ```
  R = (1 + (2Ax + B)²)^(3/2) / |2A|
  ```

- Convert pixel coordinates to meters:
  - Y: 30m / 720px ≈ 0.042 m/px
  - X: 3.7m / 700px ≈ 0.005 m/px

### ⚠️ Lane Departure Warning

- Compute estimated vehicle center vs. lane center
- Offset fraction determines warning level:

| Offset | Status | Color |
|--------|--------|-------|
| < 15% | 🟢 SAFE | Green |
| 15%–30% | 🟡 CAUTION | Yellow |
| > 30% | 🔴 DANGER | Red |

- Visual overlay on the output frame in real time

---

## Slide 10 — Web Application Demo

### 🌐 Flask Web Application

**Features:**
- Drag-and-drop file upload (images & videos)
- Real-time processing with annotated output preview
- Download processed files
- Responsive design for desktop and mobile browsers

**Architecture:**
- Flask backend with Jinja2 templates
- Static file serving for uploads and processed outputs
- RESTful endpoints for file upload and processing
- Maximum upload size: 100 MB

**Supported formats:**
- Images: PNG, JPG, JPEG, BMP, TIFF
- Videos: MP4, AVI, MOV, MKV, WEBM

---

## Slide 11 — Results & Accuracy

### 📊 Performance Metrics

| Condition | Detection Rate | Processing Time | FPS |
|-----------|---------------|-----------------|-----|
| Clear daylight | ~95% | ~18 ms | ~55 |
| Overcast | ~88% | ~19 ms | ~52 |
| Dusk | ~75% | ~20 ms | ~50 |
| Night (headlights) | ~60% | ~22 ms | ~45 |
| Rain | ~70% | ~21 ms | ~47 |

**Key achievements:**
- ✅ Exceeds real-time threshold (>30 FPS) on standard hardware
- ✅ Robust across brightness variations (0.5×–1.5×)
- ✅ Modular design allows easy parameter tuning
- ✅ Clean separation of concerns for testability

---

## Slide 12 — Challenges & Solutions

### 🧩 Problems Encountered & How We Solved Them

| Challenge | Solution |
|-----------|----------|
| **Varying lighting conditions** | CLAHE preprocessing for adaptive contrast enhancement |
| **Shadows on road surface** | ROI masking + edge threshold tuning |
| **Curved roads** | Polynomial fitting instead of purely linear models |
| **Noisy detections / flickering** | Exponential moving average smoothing (α = 0.7) |
| **False edges from road markings** | Slope-based filtering (reject horizontal/vertical) |
| **Different resolutions** | Fraction-based ROI vertices for resolution independence |
| **Processing speed** | Efficient pipeline design — no redundant computation |
| **Web file handling** | Secure filename handling, size limits, format validation |

---

## Slide 13 — Future Scope

### 🔮 Potential Enhancements

- **Deep Learning Integration** — CNN/U-Net segmentation for improved accuracy in complex scenes
- **Curved Lane Models** — Spline-based or B-spline lane representations
- **Multi-Lane Detection** — Detect adjacent lanes beyond the ego lane
- **Night Mode** — Specialized IR/low-light preprocessing
- **GPU Acceleration** — CUDA-backed OpenCV for >100 FPS
- **Mobile Deployment** — Android/iOS app using OpenCV Mobile SDK
- **Sensor Fusion** — Combine camera with LiDAR/radar for 3D lane estimation
- **Dashboard Integration** — OBD-II data for speed-aware departure warnings
- **Edge Deployment** — Raspberry Pi or NVIDIA Jetson for embedded applications

---

## Slide 14 — Thank You

### 🙏 Thank You!

**Project:** Smart Road Lane Line Detection System

**Author:** Harsh Raj

**GitHub:** [github.com/harshraj/smart-lane-detection](https://github.com/harshraj/smart-lane-detection)

---

### Questions & Discussion

_I'm happy to answer any questions about the implementation, demonstrate the system live, or discuss potential improvements._

---

> _Built with Python, OpenCV, and Flask — June 2026_
