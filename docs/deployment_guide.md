# Smart Road Lane Line Detection System — Deployment Guide

This guide provides comprehensive instructions for deploying and running the **Smart Road Lane Line Detection System** in various environments, ranging from local developer setups and command-line interfaces to production-ready web servers and containerised deployments.

---

## Table of Contents
1. [Prerequisites & System Requirements](#1-prerequisites--system-requirements)
2. [Local Environment Setup](#2-local-environment-setup)
3. [Running Command Line Interfaces (CLI)](#3-running-command-line-interfaces-cli)
4. [Deploying the Flask Web Application Locally](#4-deploying-the-flask-web-application-locally)
5. [Production Deployment (WSGI & Production Servers)](#5-production-deployment-wsgi--production-servers)
6. [Containerised Deployment (Docker)](#6-containerised-deployment-docker)
7. [Performance Optimisation & Troubleshooting](#7-performance-optimisation--troubleshooting)

---

## 1. Prerequisites & System Requirements

Before deploying the system, ensure your hardware and software meet the following requirements:

### Hardware Requirements
*   **CPU**: Dual-core 2.0 GHz or higher (Quad-core recommended for video processing).
*   **RAM**: 4 GB minimum (8 GB recommended for video and real-time processing).
*   **Webcam**: USB webcam or integrated camera (required *only* for real-time webcam mode).

### Software Requirements
*   **Operating System**: Windows 10/11, macOS, or Linux (Ubuntu 20.04+ recommended).
*   **Python**: Version `3.8` to `3.13` (tested and verified on Python `3.13`).
*   **Dependencies**: OpenCV, NumPy, Matplotlib, SciPy, Flask, Pillow, Werkzeug.

---

## 2. Local Environment Setup

Follow these steps to set up a clean virtual environment and install the required packages:

### Step 1: Clone or Copy the Repository
Navigate to your desired project directory:
```bash
cd C:\Users\Harsh Raj\.gemini\antigravity\scratch\smart-lane-detection
```

### Step 2: Create a Virtual Environment
Using Python's built-in `venv` module, create an isolated environment to prevent dependency conflicts:

**On Windows:**
```powershell
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Packages
Install all project requirements using pip:
```bash
pip install -r requirements.txt
```

> [!NOTE]
> The dependencies listed in `requirements.txt` are lightweight classical computer vision libraries. No heavy deep learning frameworks (like PyTorch or TensorFlow) or GPU runtimes (like CUDA) are required.

---

## 3. Running Command Line Interfaces (CLI)

The system includes three CLI entry points for testing different CV pipelines directly from the terminal.

### A. Single Image Processing (`run_image.py`)
Processes a static road image, applies the detection overlay, prints metadata metrics, and saves the output.
```bash
python run_image.py --input data/sample/road.jpg --output output/images/ --no-display
```
*   `--input` (Required): Path to input road image.
*   `--output` (Optional): Target folder (default: `output/images`).
*   `--no-display` (Optional): Use this flag to suppress opening an interactive OpenCV window (ideal for head-less scripts or servers).

### B. Video Processing (`run_video.py`)
Processes video file frame-by-frame, performs temporal smoothing, writes the output stream to a new video file, and prints progress.
```bash
python run_video.py --input path/to/video.mp4 --output output/videos/ --no-display
```
*   `--no-display` (Optional): Hides the visual frame-by-frame window for faster processing throughput.

### C. Real-Time Webcam Detection (`run_webcam.py`)
Captures feed from a connected USB/built-in camera, applies real-time lane boundaries and HUD indicators, and renders the live output.
```bash
python run_webcam.py
```
*   Press **`q`** inside the display window to exit webcam mode safely.

---

## 4. Deploying the Flask Web Application Locally

The Flask web application provides an intuitive interactive dashboard to upload media and view real-time statistics.

### Run in Development Mode
Start the Flask application using:
```bash
python run_app.py
```
By default, the server starts on `http://127.0.0.1:5000`. 

> [!WARNING]
> Do not use `run_app.py` directly for production deployment. The Flask development server is not designed for security, stability, or concurrency under high traffic.

---

## 5. Production Deployment (WSGI & Production Servers)

For production deployments, wrap the Flask application in a WSGI container.

### A. Deploying on Linux (Gunicorn + Nginx)
[Gunicorn](https://gunicorn.org/) is a production-ready WSGI HTTP server for UNIX-like systems.

1.  **Install Gunicorn**:
    ```bash
    pip install gunicorn
    ```
2.  **Start Gunicorn**:
    Run from the project root directory:
    ```bash
    gunicorn --workers 4 --bind 0.0.0.0:8000 run_app:app
    ```
    *   `--workers 4`: Spawns 4 worker processes to handle concurrent requests. A common rule is `(2 * CPU_CORES) + 1`.
3.  **Reverse Proxy via Nginx**:
    Configure Nginx to proxy incoming traffic to the Gunicorn socket:
    ```nginx
    server {
        listen 80;
        server_name your_domain.com;

        location / {
            proxy_pass http://127.0.0.1:8000;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        }

        location /static/ {
            alias /path/to/smart-lane-detection/app/static/;
        }
    }
    ```

### B. Deploying on Windows (Waitress)
[Waitress](https://docs.pylonsproject.org/projects/waitress/) is a production-quality pure-Python WSGI server that runs efficiently on Windows.

1.  **Install Waitress**:
    ```powershell
    pip install waitress
    ```
2.  **Create a Production Launcher** (`wsgi_production.py`):
    ```python
    from waitress import serve
    from run_app import app

    if __name__ == "__main__":
        print("[INFO] Starting production Waitress server on port 80...")
        serve(app, host="0.0.0.0", port=80, threads=8)
    ```
3.  **Start the Server**:
    ```powershell
    python wsgi_production.py
    ```

### C. Deploying on Streamlit Cloud (Recommended)
Streamlit Cloud offers a completely free, fast hosting environment for Python applications.

1.  **Run Streamlit Locally**:
    ```bash
    pip install -r requirements.txt
    streamlit run streamlit_app.py
    ```
2.  **Deploy on Streamlit Cloud**:
    - Push your latest changes to GitHub (already pushed).
    - Log into [Streamlit Community Cloud](https://share.streamlit.io/).
    - Click **"New app"** and select this GitHub repository (`mayank7720/smart-lane-detection`).
    - Set the main file path to `streamlit_app.py`.
    - Click **"Deploy"**. Streamlit will automatically install the system and python packages and provision your dashboard online!

---

## 6. Containerised Deployment (Docker)


Docker allows containerising the entire application so that it runs reliably on any cloud provider (AWS, Azure, GCP) or server.

### A. Dockerfile
Create a file named `Dockerfile` in the root of the project:
```dockerfile
# Use a lightweight official Python image
FROM python:3.11-slim

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy dependency definition
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application code
COPY . .

# Expose the application port
EXPOSE 5000

# Set environment variables
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Run the app using waitress
RUN pip install waitress
CMD ["python", "-c", "from waitress import serve; from run_app import app; serve(app, host='0.0.0.0', port=5000, threads=4)"]
```

### B. Building and Running the Container
1.  **Build the Docker Image**:
    ```bash
    docker build -t smart-lane-detector:latest .
    ```
2.  **Run the Container**:
    ```bash
    docker run -d -p 80:5000 --name lane-detector smart-lane-detector:latest
    ```
    Now, the application dashboard will be accessible at `http://localhost`.

---

## 7. Performance Optimisation & Troubleshooting

### Optimization Tips
*   **Disable UI Video Display**: When running video batch processing or benchmarks, always use `--no-display` to bypass GUI rendering bottlenecks.
*   **Resize Input Video Stream**: Processing high-resolution feeds (4K or 1080p) is CPU intensive. Downscale frames to `640x360` or `1280x720` in `config.py` to boost FPS drastically.
*   **Adjust Smoothing Factor**: If you notice jittery lane detection overlays on bumpy roads, increase `LANE_SMOOTHING_FACTOR` in `config.py` (e.g., set to `0.85` or `0.90`).

### Troubleshooting
*   **Port Collisions (`Address already in use`)**: If Flask fails to start because port `5000` is occupied, change the port in `config.py` (`FLASK_PORT = 5001`) or run with `--port` dynamically.
*   **`cv2.error: C++ Exception` / `libGL.so.1 missing`**: This happens on Linux servers lacking graphical dependencies. Ensure you run `apt-get install libgl1-mesa-glx libglib2.0-0` or install `opencv-python-headless` instead of `opencv-python` in your environment.
*   **Headless Matplotlib Error**: If running report generation fails with `TclError: no display name`, ensure `matplotlib.use('Agg')` is declared *before* importing `pyplot` (this is configured by default in `report_generator.py`).
