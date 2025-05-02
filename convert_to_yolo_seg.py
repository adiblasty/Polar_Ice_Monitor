import os
import cv2
import numpy as np
from glob import glob

# Paths (adjust if needed)
input_images = r'C:\Users\Dinesh\Desktop\Polar_segmentation\ArcticIceData\images'
input_masks = r'C:\Users\Dinesh\Desktop\Polar_segmentation\ArcticIceData\masks'
output_labels = r'C:\Users\Dinesh\Desktop\Polar_segmentation\ArcticIceData\labels'

os.makedirs(output_labels, exist_ok=True)

# Collect .jpg and .JPG files
image_files = glob(os.path.join(input_images, '*.jpg')) + glob(os.path.join(input_images, '*.JPG'))

print(f"🧪 Found {len(image_files)} image files.")

for image_path in image_files:
    filename = os.path.basename(image_path)
    image_name = os.path.splitext(filename)[0]

    # Adjust to match the mask filename pattern
    mask_path = os.path.join(input_masks, image_name + '-mask.png')

    if not os.path.exists(mask_path):
        print(f"⚠️ Mask not found for {filename}")
        continue

    # Read mask
    mask = cv2.imread(mask_path, 0)
    if mask is None:
        print(f"❌ Failed to read mask: {mask_path}")
        continue

    # Threshold to binary
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Read image to get dimensions
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    label_file_path = os.path.join(output_labels, image_name + '.txt')

    with open(label_file_path, 'w') as f:
        for contour in contours:
            if len(contour) < 6:  # Skip tiny or malformed shapes
                continue

            # Flatten and normalize coordinates
            normalized_points = []
            for point in contour.squeeze():
                x = point[0] / width
                y = point[1] / height
                normalized_points.extend([x, y])

            # YOLO segmentation format: class_id x1 y1 x2 y2 ...
            label_line = '0 ' + ' '.join([f'{pt:.6f}' for pt in normalized_points]) + '\n'
            f.write(label_line)

    print(f"✅ Saved label for {filename}")
