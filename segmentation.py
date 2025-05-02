import os
import cv2
import numpy as np
from glob import glob

# === CONFIG ===
images_dir = r'C:\Users\Dinesh\Desktop\Polar_segmentation\ArcticIceData\images'
masks_dir = r'C:\Users\Dinesh\Desktop\Polar_segmentation\ArcticIceData\masks'
labels_dir = r'C:\Users\Dinesh\Desktop\Polar_segmentation\ArcticIceData\labels'
os.makedirs(labels_dir, exist_ok=True)

# Process masks
mask_files = sorted(glob(os.path.join(masks_dir, '*-mask.png')))

for mask_path in mask_files:
    base = os.path.basename(mask_path).replace('-mask.png', '')
    image_path = os.path.join(images_dir, base + '.jpg')
    label_path = os.path.join(labels_dir, base + '.txt')

    if not os.path.exists(image_path):
        print(f"❌ Missing image: {base}")
        continue

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"❌ Error reading: {mask_path}")
        continue

    h, w = mask.shape
    _, binary = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    label_lines = []
    for contour in contours:
        if cv2.contourArea(contour) < 100:
            continue

        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True).squeeze()

        if approx.ndim != 2 or approx.shape[0] < 3:
            continue

        # Normalize & deduplicate
        seen = set()
        points = []
        for x, y in approx:
            nx, ny = round(x / w, 6), round(y / h, 6)
            key = (nx, ny)
            if key not in seen:
                seen.add(key)
                points.append(key)

        if len(points) < 3:
            continue

        # Reject flat/degenerate shapes
        x_vals, y_vals = zip(*points)
        if max(x_vals) - min(x_vals) < 0.01 or max(y_vals) - min(y_vals) < 0.01:
            continue

        if len(points) > 100:
            points = points[:100]

        flat = [coord for pt in points for coord in pt]
        label_lines.append(f"0 " + " ".join([f"{c:.6f}" for c in flat]))

    with open(label_path, 'w') as f:
        f.write("\n".join(label_lines))

    print(f"✅ {base}.txt: {len(label_lines)} object(s)")

print("🎉 Label generation completed.")
