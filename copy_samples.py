import shutil
import os

src_dir = r"C:\Users\Harsh Raj\.gemini\antigravity\brain\8c32527b-d18c-48c0-ad81-0bce5d498cf2"
dst_dir = r"C:\Users\Harsh Raj\.gemini\antigravity\scratch\smart-lane-detection\data\sample"

files = {
    "highway_daylight_1782380067370.png": "highway_daylight.png",
    "curved_road_1782380091843.png": "curved_road.png",
    "night_driving_1782380108467.png": "night_driving.png"
}

os.makedirs(dst_dir, exist_ok=True)

for src_name, dst_name in files.items():
    src_path = os.path.join(src_dir, src_name)
    dst_path = os.path.join(dst_dir, dst_name)
    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
        print(f"Copied {src_name} to {dst_name}")
    else:
        print(f"Source file not found: {src_path}")
