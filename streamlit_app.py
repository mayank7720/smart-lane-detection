import os
import sys
import time
import tempfile
import cv2
import numpy as np
import streamlit as st

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from core.pipeline import LaneDetectionPipeline

# Initialize the pipeline (cached to avoid reloading model cascade on every rerun)
@st.cache_resource
def get_pipeline():
    return LaneDetectionPipeline()

pipeline = get_pipeline()

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LaneSight ADAS Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS for Premium Dark Theme ─────────────────────────────────────────
st.markdown("""
<style>
    /* Dark Slate Background & Fonts */
    .stApp {
        background-color: #0A1628;
        color: #F5F7FA;
    }
    .stSidebar {
        background-color: #122240 !important;
    }
    
    /* Headers & Badges */
    .hero-title {
        color: #00C853;
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 3rem;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .hud-card {
        background-color: #1A2D50;
        border: 1px solid #00B0FF;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .hud-val {
        font-size: 2rem;
        font-weight: bold;
        color: #FFFFFF;
    }
    .hud-label {
        font-size: 0.9rem;
        color: #94A3B8;
        margin-top: 0.4rem;
        text-transform: uppercase;
    }
    
    /* Warning Badges */
    .badge-safe {
        background-color: rgba(0, 200, 83, 0.2);
        color: #00E676;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        border: 1px solid #00E676;
    }
    .badge-caution {
        background-color: rgba(255, 214, 0, 0.2);
        color: #FFD600;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        border: 1px solid #FFD600;
    }
    .badge-danger {
        background-color: rgba(255, 23, 68, 0.2);
        color: #FF1744;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        border: 1px solid #FF1744;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar Navigation ────────────────────────────────────────────────────────
st.sidebar.markdown("<h2 style='color:#00C853; text-align:center;'>🚗 LaneSight ADAS</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='text-align:center; color:#94A3B8;'>AI-Powered Lane & Object Tracker</p>", unsafe_allow_html=True)
st.sidebar.write("---")

app_mode = st.sidebar.radio(
    "Select Operation Mode",
    ["🏠 Home & ADAS Simulator", "📸 Analyze Custom Image", "🎥 Analyze Custom Video", "📊 System Analytics"]
)

# ── Session State for Analytics History ───────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

def add_history(item_type, meta, elapsed):
    st.session_state.history.append({
        "type": item_type,
        "curvature": meta.get("curvature_radius"),
        "offset": meta.get("offset"),
        "warning": meta.get("warning_level", "safe"),
        "latency": elapsed,
        "timestamp": time.strftime("%H:%M:%S")
    })

# ══════════════════════════════════════════════════════════════════════════════
# MODE 1: Home & ADAS Simulator
# ═══════════════════════════════════════════════════════════════════════════
if app_mode == "🏠 Home & ADAS Simulator":
    st.markdown("<h1 class='hero-title'>AI-Powered Lane Detection</h1>", unsafe_allow_html=True)
    st.markdown("<p class='hero-subtitle'>Powering next-generation Advanced Driver Assistance Systems (ADAS) using classical computer vision.</p>", unsafe_allow_html=True)
    
    st.write("---")
    st.subheader("💡 Interactive Simulator")
    st.write("Choose one of our preloaded highway scenarios below to run the ADAS lane fitting and vehicle detection pipeline:")
    
    col1, col2, col3 = st.columns(3)
    
    demo_choice = None
    with col1:
        if st.button("☀️ Daylight Highway", use_container_width=True):
            demo_choice = "highway_daylight.png"
    with col2:
        if st.button("🏔️ Winding Curved Road", use_container_width=True):
            demo_choice = "curved_road.png"
    with col3:
        if st.button("🌙 Night Driving", use_container_width=True):
            demo_choice = "night_driving.png"
            
    if demo_choice:
        sample_path = os.path.join(config.SAMPLE_DIR, demo_choice)
        if os.path.exists(sample_path):
            img = cv2.imread(sample_path)
            t0 = time.time()
            # Clear pipeline temporal queue since it's a new single image run
            pipeline._prev_left = None
            pipeline._prev_right = None
            
            out_img, meta = pipeline.process_frame(img)
            elapsed = time.time() - t0
            add_history("image", meta, elapsed)
            
            # Display metrics
            st.write("---")
            st.subheader(f"📊 ADAS Simulator Telemetry: {demo_choice.replace('_', ' ').replace('.png', '').title()}")
            
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                curv = meta.get("curvature_radius")
                curv_str = "Straight" if not curv or curv > 5000 else f"{curv:.0f} m"
                st.markdown(f"<div class='hud-card'><div class='hud-val'>{curv_str}</div><div class='hud-label'>Road Curvature</div></div>", unsafe_allow_html=True)
            with m_col2:
                off = meta.get("offset")
                off_str = f"{abs(off):.2f} m ({'Left' if off < 0 else 'Right'})" if off is not None else "Centred"
                st.markdown(f"<div class='hud-card'><div class='hud-val'>{off_str}</div><div class='hud-label'>Lane Offset</div></div>", unsafe_allow_html=True)
            with m_col3:
                wl = meta.get("warning_level", "safe").upper()
                badge_class = f"badge-{wl.lower()}"
                st.markdown(f"<div class='hud-card'><div class='hud-val'><span class='{badge_class}'>{wl}</span></div><div class='hud-label'>Warning Level</div></div>", unsafe_allow_html=True)
            with m_col4:
                st.markdown(f"<div class='hud-card'><div class='hud-val'>{elapsed*1000:.1f} ms</div><div class='hud-label'>Processing Latency</div></div>", unsafe_allow_html=True)
            
            # Display original vs processed side-by-side
            img_col1, img_col2 = st.columns(2)
            with img_col1:
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original Dashcam Input", use_container_width=True)
            with img_col2:
                st.image(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB), caption="Processed ADAS HUD Overlay", use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# MODE 2: Analyze Custom Image
# ═══════════════════════════════════════════════════════════════════════════
elif app_mode == "📸 Analyze Custom Image":
    st.subheader("📸 Analyze Custom Image")
    st.write("Upload a custom road image from your vehicle dashcam to extract lane outlines and vehicle boxes:")
    
    uploaded_file = st.file_uploader("Upload Image File", type=list(config.ALLOWED_IMAGE_EXTENSIONS))
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        t0 = time.time()
        pipeline._prev_left = None
        pipeline._prev_right = None
        out_img, meta = pipeline.process_frame(img)
        elapsed = time.time() - t0
        add_history("image", meta, elapsed)
        
        # Telemetry
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            curv = meta.get("curvature_radius")
            curv_str = "Straight" if not curv or curv > 5000 else f"{curv:.0f} m"
            st.markdown(f"<div class='hud-card'><div class='hud-val'>{curv_str}</div><div class='hud-label'>Road Curvature</div></div>", unsafe_allow_html=True)
        with m_col2:
            off = meta.get("offset")
            off_str = f"{abs(off):.2f} m ({'Left' if off < 0 else 'Right'})" if off is not None else "Centred"
            st.markdown(f"<div class='hud-card'><div class='hud-val'>{off_str}</div><div class='hud-label'>Lane Offset</div></div>", unsafe_allow_html=True)
        with m_col3:
            wl = meta.get("warning_level", "safe").upper()
            badge_class = f"badge-{wl.lower()}"
            st.markdown(f"<div class='hud-card'><div class='hud-val'><span class='{badge_class}'>{wl}</span></div><div class='hud-label'>Warning Level</div></div>", unsafe_allow_html=True)
        with m_col4:
            st.markdown(f"<div class='hud-card'><div class='hud-val'>{elapsed*1000:.1f} ms</div><div class='hud-label'>Latency</div></div>", unsafe_allow_html=True)
            
        # Comparison images
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original Input", use_container_width=True)
        with img_col2:
            st.image(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB), caption="LaneSight ADAS Overlay", use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# MODE 3: Analyze Custom Video
# ═══════════════════════════════════════════════════════════════════════════
elif app_mode == "🎥 Analyze Custom Video":
    st.subheader("🎥 Batch Video Processing")
    st.write("Upload a road video clip. The application will process it frame-by-frame using temporal queue smoothing and save a browser-compatible H.264 MP4:")
    
    uploaded_video = st.file_uploader("Upload Video File", type=list(config.ALLOWED_VIDEO_EXTENSIONS))
    
    if uploaded_video is not None:
        # Save uploaded video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        tfile.close()
        
        cap = cv2.VideoCapture(tfile.name)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        st.write(f"🎞️ Video Details: `{w}x{h}` | `{fps} FPS` | `{total_frames} frames`")
        
        # Temporary output path
        out_tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        out_tfile.close()
        
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(out_tfile.name, fourcc, fps, (w, h))
        
        # Reset pipeline state
        pipeline._prev_left = None
        pipeline._prev_right = None
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        t0 = time.time()
        frame_count = 0
        curvatures = []
        latencies = []
        
        # Frame-by-frame loop
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_t0 = time.time()
            out_frame, meta = pipeline.process_frame(frame)
            frame_elapsed = time.time() - frame_t0
            
            writer.write(out_frame)
            
            frame_count += 1
            latencies.append(frame_elapsed)
            if meta.get("curvature_radius"):
                curvatures.append(meta["curvature_radius"])
                
            # Update progress
            pct = int(frame_count / total_frames * 100)
            progress_bar.progress(pct)
            status_text.text(f"Processing frame {frame_count}/{total_frames} ({pct}%) | Latency: {frame_elapsed*1000:.1f} ms/frame")
            
        cap.release()
        writer.release()
        total_elapsed = time.time() - t0
        
        # Clean up input temporary file
        os.unlink(tfile.name)
        
        st.success(f"Processing completed in {total_elapsed:.1f} seconds!")
        
        # Add to analytics history
        avg_curv = sum(curvatures)/len(curvatures) if curvatures else None
        meta_dummy = {"curvature_radius": avg_curv, "warning_level": "safe", "offset": 0.0}
        add_history("video", meta_dummy, total_elapsed / frame_count)
        
        # Display output video
        st.write("### 📺 Output Video comparison")
        v_col1, v_col2 = st.columns(2)
        with v_col1:
            st.subheader("Original Input Video")
            st.video(uploaded_video)
        with v_col2:
            st.subheader("Processed ADAS Video")
            # Read output video bytes and render
            with open(out_tfile.name, "rb") as f:
                processed_bytes = f.read()
            st.video(processed_bytes)
            
        # Clean up output temporary file
        os.unlink(out_tfile.name)

# ══════════════════════════════════════════════════════════════════════════════
# MODE 4: System Analytics
# ═══════════════════════════════════════════════════════════════════════════
elif app_mode == "📊 System Analytics":
    st.subheader("📊 System Telemetry & Historical Database")
    
    if not st.session_state.history:
        st.info("No runs logged in this session yet. Go to Home or Image Analyzer and process some frames!")
    else:
        hist_df = st.session_state.history
        
        # Key Aggregates
        total_runs = len(hist_df)
        avg_latency = sum(x["latency"] for x in hist_df) / total_runs * 1000
        safes = sum(1 for x in hist_df if x["warning"] == "safe")
        success_rate = (safes / total_runs) * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Frames Processed", total_runs)
        with col2:
            st.metric("Average Latency (ms)", f"{avg_latency:.1f}")
        with col3:
            st.metric("ADAS Safety Success Rate", f"{success_rate:.1f}%")
            
        # Plot Charts
        st.write("---")
        c_col1, c_col2 = st.columns(2)
        with c_col1:
            st.write("#### ⚡ Latency Profile over Time (ms)")
            latencies = [x["latency"]*1000 for x in hist_df]
            st.line_chart(latencies)
        with c_col2:
            st.write("#### 🗺️ Road Curvature radius over Time (m)")
            curvs = [x["curvature"] if x["curvature"] and x["curvature"] < 5000 else 1000 for x in hist_df]
            st.area_chart(curvs)
            
        # Log Table
        st.write("#### 📜 Processing History Log")
        st.write(hist_df)
