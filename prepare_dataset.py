import os
import shutil
from glob import glob
from sklearn.model_selection import train_test_split

# Input dirs
images_dir = r'C:\Users\Dinesh\Desktop\Polar_segmentation\ArcticIceData\images'
labels_dir = r'C:\Users\Dinesh\Desktop\Polar_segmentation\ArcticIceData\labels'

# Output base
output_dir = 'PolarSeg'
os.makedirs(output_dir, exist_ok=True)

# Create YOLO folders
for folder in ['images/train', 'images/val', 'labels/train', 'labels/val']:
    os.makedirs(os.path.join(output_dir, folder), exist_ok=True)

# Get image files
image_paths = glob(os.path.join(images_dir, '*.jpg')) + glob(os.path.join(images_dir, '*.JPG'))

# Split 80/20
train_imgs, val_imgs = train_test_split(image_paths, test_size=0.2, random_state=42)

# Move files to train/val
def move_files(image_list, split):
    for img_path in image_list:
        filename = os.path.basename(img_path)
        label_name = os.path.splitext(filename)[0] + '.txt'
        label_file = os.path.join(labels_dir, label_name)

        # Copy image
        shutil.copy(img_path, os.path.join(output_dir, f'images/{split}', filename))

        # Copy label if exists
        if os.path.exists(label_file):
            shutil.copy(label_file, os.path.join(output_dir, f'labels/{split}', label_name))

move_files(train_imgs, 'train')
move_files(val_imgs, 'val')

print("✅ YOLOv11 folder structure created at ./PolarSeg")
