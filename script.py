import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
from glob import glob

# === CONFIG ===
model_path = r"runs/segment/train8/weights/best.pt"
source_dir = r"datasets/images/val"  # Test images
output_overlay_dir = r"climate_output/overlays"
output_csv_path = r"climate_output/melt_report.csv"
os.makedirs(output_overlay_dir, exist_ok=True)

# === PARAMETERS ===
DANGER_THRESHOLD = 60.0  # % coverage considered safe

# === LOAD MODEL ===
model = YOLO(model_path)

# === RUN INFERENCE ===
results = model.predict(source=source_dir, save=False, imgsz=416, stream=True)

report = []

for result in results:
    image_path = result.path
    filename = os.path.basename(image_path)
    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    # Combine masks (assume binary segmentation of ice)
    masks = result.masks
    if masks is None:
        print(f"⚠️ No mask detected in: {filename}")
        continue

    combined_mask = np.zeros((h, w), dtype=np.uint8)
    for m in masks.data:
        m_resized = cv2.resize(m.cpu().numpy(), (w, h))
        combined_mask = np.logical_or(combined_mask, m_resized > 0.5)

    # Calculate coverage
    ice_pixels = np.sum(combined_mask)
    total_pixels = h * w
    coverage_pct = 100.0 * ice_pixels / total_pixels
    danger_flag = int(coverage_pct < DANGER_THRESHOLD)

    # Extract date from filename (e.g. P0-2016042417.jpg)
    date_str = filename.split('-')[-1].split('.')[0]  # e.g. 2016042417
    try:
        date_obj = datetime.strptime(date_str, "%Y%m%d%H")
        formatted_date = date_obj.strftime("%Y-%m-%d %H:%M")
    except:
        formatted_date = "Unknown"

    # Overlay mask in color (red if danger, green if safe)
    overlay_color = (0, 0, 255) if danger_flag else (0, 255, 0)
    color_mask = np.zeros_like(image)
    color_mask[combined_mask] = overlay_color
    overlay = cv2.addWeighted(image, 1.0, color_mask, 0.4, 0)

    # Save overlay
    overlay_path = os.path.join(output_overlay_dir, filename)
    cv2.imwrite(overlay_path, overlay)

    # Add to report
    report.append({
        "filename": filename,
        "date": formatted_date,
        "coverage_percent": round(coverage_pct, 2),
        "danger": danger_flag
    })

    print(f"✅ {filename} - Coverage: {coverage_pct:.2f}% {'🚨' if danger_flag else ''}")

# Save CSV
df = pd.DataFrame(report)
os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
df.to_csv(output_csv_path, index=False)
print(f"\n📄 Melt report saved to: {output_csv_path}")
