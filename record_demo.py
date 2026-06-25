import os
import sys
import time
import glob
import cv2
import numpy as np
from playwright.sync_api import sync_playwright

def record():
    print("Starting Playwright automation...")
    with sync_playwright() as p:
        # Launch chromium headlessly
        browser = p.chromium.launch(headless=True)
        
        # Create a context with video recording enabled
        video_dir = os.path.join(os.getcwd(), "temp_recording")
        os.makedirs(video_dir, exist_ok=True)
        
        context = browser.new_context(
            record_video_dir=video_dir,
            record_video_size={"width": 1280, "height": 720},
            viewport={"width": 1280, "height": 720}
        )
        
        page = context.new_page()
        
        # 1. Navigation to Homepage
        print("1. Navigating to Home Page...")
        page.goto("http://127.0.0.1:5000/")
        page.wait_for_timeout(3500) # Let the HUD canvas load and render
        
        # Scroll down to show grid and driving car
        print("Scrolling page to hero section...")
        for i in range(1, 15):
            page.evaluate("window.scrollBy(0, 15)")
            page.wait_for_timeout(60)
        page.wait_for_timeout(2500)
        
        # Scroll to interactive simulator section
        print("Scrolling to Interactive ADAS Simulator...")
        page.locator("#demo-simulator").scroll_into_view_if_needed()
        page.wait_for_timeout(2500)
        
        # 2. Run Daylight Highway demo
        print("2. Clicking Daylight Highway Analysis...")
        daylight_card = page.locator(".demo-card").nth(0)
        daylight_card.hover()
        page.wait_for_timeout(1000)
        
        daylight_link = daylight_card.locator("a.btn")
        daylight_link.click()
        page.wait_for_load_state("load")
        page.wait_for_timeout(3000)
        
        # Scroll down to comparison images
        print("Scrolling to result comparison...")
        page.locator(".result-grid").scroll_into_view_if_needed()
        page.wait_for_timeout(2500)
        
        # Scroll down to metadata cards
        print("Scrolling to metadata grid...")
        page.locator(".meta-grid").scroll_into_view_if_needed()
        page.wait_for_timeout(4000)
        
        # Scroll to bottom
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(2000)
        
        # Go back home
        page.locator("nav.navbar a.nav-link:has-text('Home')").first.click()
        page.wait_for_load_state("load")
        page.wait_for_timeout(2000)
        
        # 3. Run Curved Road demo
        page.locator("#demo-simulator").scroll_into_view_if_needed()
        page.wait_for_timeout(1500)
        
        print("3. Clicking Curved Road Analysis...")
        curved_card = page.locator(".demo-card").nth(1)
        curved_card.hover()
        page.wait_for_timeout(1000)
        
        curved_link = curved_card.locator("a.btn")
        curved_link.click()
        page.wait_for_load_state("load")
        page.wait_for_timeout(3000)
        
        # Scroll to see curvature results and warning badge
        print("Scrolling to curvature results...")
        page.locator(".result-grid").scroll_into_view_if_needed()
        page.wait_for_timeout(2500)
        page.locator(".meta-grid").scroll_into_view_if_needed()
        page.wait_for_timeout(4000)
        
        # 4. Upload Image page
        print("4. Navigating to Image Upload page...")
        page.locator("nav.navbar a.nav-link:has-text('Image')").first.click()
        page.wait_for_load_state("load")
        page.wait_for_timeout(2000)
        
        # Upload file
        print("Uploading night_driving.png...")
        file_input = page.locator("input[type='file']")
        file_input.set_input_files("data/sample/night_driving.png")
        page.wait_for_timeout(1500)
        
        page.locator("button[type='submit']").click()
        page.wait_for_load_state("load")
        page.wait_for_timeout(4000)
        
        # Scroll down in result
        print("Scrolling to upload results...")
        page.locator(".result-grid").scroll_into_view_if_needed()
        page.wait_for_timeout(2500)
        page.locator(".meta-grid").scroll_into_view_if_needed()
        page.wait_for_timeout(4000)
        
        # 5. Upload Video page
        print("5. Navigating to Video Upload page...")
        page.locator("nav.navbar a.nav-link:has-text('Video')").first.click()
        page.wait_for_load_state("load")
        page.wait_for_timeout(2000)
        
        # Upload video file
        print("Uploading highway.mp4...")
        file_input = page.locator("input[type='file']")
        file_input.set_input_files("data/sample/highway.mp4")
        page.wait_for_timeout(2000)
        
        page.locator("button[type='submit']").click()
        # It takes time to process the video, wait for it
        print("Waiting for video processing to complete...")
        page.wait_for_timeout(9000) # This allows the processing screen spinner to show
        page.wait_for_load_state("load")
        page.wait_for_timeout(4000)
        
        # Scroll down in result
        print("Scrolling down to video comparison...")
        page.locator(".result-grid").scroll_into_view_if_needed()
        page.wait_for_timeout(5000) # Wait to show video comparison
        
        # 6. Real-Time simulation page
        print("6. Navigating to Real-Time Detection page...")
        page.locator("nav.navbar a.nav-link:has-text('Real-Time')").first.click()
        page.wait_for_load_state("load")
        page.wait_for_timeout(2000)
        
        # Click Start Feed
        print("Starting live feed...")
        page.locator("#startFeed").click()
        page.wait_for_timeout(8000) # Wait to show running feed
        
        # Click Stop Feed
        print("Stopping live feed...")
        page.locator("#stopFeed").click()
        page.wait_for_timeout(2000)
        
        # 7. Dashboard page
        print("7. Navigating to Dashboard page...")
        page.locator("nav.navbar a.nav-link:has-text('Dashboard')").first.click()
        try:
            page.wait_for_load_state("load", timeout=5000)
        except Exception:
            pass
        page.wait_for_timeout(5000) # Show statistics charts
        
        print("Finishing recording...")
        context.close()
        browser.close()

    print("Playwright automation finished!")


def draw_caption_bar(frame, text):
    h, w, c = frame.shape
    bar_h = 75
    
    # Create overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (18, 18, 24), -1) # Dark navy-black color
    cv2.line(overlay, (0, h - bar_h), (w, h - bar_h), (0, 229, 255), 2) # Neon cyan border line
    
    # Blend
    alpha = 0.85
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # Font
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (255, 255, 255)
    thickness = 2
    
    # Centering text
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (w - text_size[0]) // 2
    text_y = h - (bar_h - text_size[1]) // 2 - 5
    
    # Draw text with shadow
    cv2.putText(frame, text, (text_x + 1, text_y + 1), font, font_scale, (0, 0, 0), thickness)
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)


def get_caption_for_time(seconds):
    if seconds < 6.0:
        return "LaneSight ADAS Dashboard - Modern Autonomous Vehicle Telemetry HUD"
    elif seconds < 16.0:
        return "Interactive Simulator - Testing Daylight Highway Driving Scenario"
    elif seconds < 25.0:
        return "Advanced Analytics - Road Curvature Radius & Pipeline Latency Profiles"
    elif seconds < 35.0:
        return "Interactive Simulator - Testing Mountain Curves & Lateral Departure Alerts"
    elif seconds < 41.0:
        return "Curvature Alerts - Activates CAUTION warning under high-drift curves"
    elif seconds < 51.0:
        return "Static Analysis Upload - Processed Night Driving under adaptive contrast (CLAHE)"
    elif seconds < 62.0:
        return "Automated Video Pipeline - Uploading dashcam footage clip"
    elif seconds < 75.0:
        return "Video Processing Results - Frame-by-frame lane fitting & tracking comparison"
    elif seconds < 87.0:
        return "Real-Time Detection Feed - Simulating live processing on autonomous camera input"
    else:
        return "Integrated System Analytics - Total processing statistics & historical database"


def post_process_video(input_video_path, output_mp4_path, output_webm_path):
    print(f"Post-processing video: {input_video_path}...")
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error opening recorded video: {input_video_path}")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Original video properties: {width}x{height} @ {fps} fps, {total_frames} total frames")
    
    # Set up writers
    fourcc_mp4 = cv2.VideoWriter_fourcc(*"avc1")
    writer_mp4 = cv2.VideoWriter(output_mp4_path, fourcc_mp4, fps, (width, height))
    
    fourcc_webm = cv2.VideoWriter_fourcc(*"VP80")
    writer_webm = cv2.VideoWriter(output_webm_path, fourcc_webm, fps, (width, height))
    
    print("MP4 Writer opened:", writer_mp4.isOpened())
    print("WebM Writer opened:", writer_webm.isOpened())
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Draw HUD style captions
        seconds = frame_idx / fps
        caption = get_caption_for_time(seconds)
        draw_caption_bar(frame, caption)
        
        # Write to outputs
        if writer_mp4.isOpened():
            writer_mp4.write(frame)
        if writer_webm.isOpened():
            writer_webm.write(frame)
            
        frame_idx += 1
        if frame_idx % 250 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames...")
            
    cap.release()
    writer_mp4.release()
    writer_webm.release()
    print("Post-processing complete!")


if __name__ == "__main__":
    record()
    
    # Locate recorded video
    recordings = glob.glob(os.path.join("temp_recording", "*.webm"))
    if not recordings:
        print("No recordings found!")
        sys.exit(1)
        
    recorded_video = recordings[0]
    print(f"Recorded video: {recorded_video}")
    
    # Outputs in brain artifacts directory
    brain_dir = r"C:\Users\Harsh Raj\.gemini\antigravity\brain\8c32527b-d18c-48c0-ad81-0bce5d498cf2"
    output_mp4 = os.path.join(brain_dir, "lanesight_demo.mp4")
    output_webm = os.path.join(brain_dir, "lanesight_demo.webm")
    
    post_process_video(recorded_video, output_mp4, output_webm)
    print("=" * 60)
    print(f"Demo MP4 saved: {output_mp4}")
    print(f"Demo WebM saved: {output_webm}")
    print("=" * 60)
