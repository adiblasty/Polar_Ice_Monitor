import os

labels_dir = r'C:\Users\Dinesh\Desktop\Polar_segmentation\ArcticIceData\labels'

deleted = 0
for file in os.listdir(labels_dir):
    if not file.endswith('.txt'):
        continue

    path = os.path.join(labels_dir, file)
    with open(path, 'r') as f:
        lines = f.readlines()

    # Check if label is empty or has < 6 coordinates
    keep = False
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 7:  # class + at least 3 (x, y) pairs
            keep = True
            break

    if not keep:
        os.remove(path)
        deleted += 1
        print(f"🗑️ Deleted: {file}")

print(f"✅ Cleanup complete. Removed {deleted} invalid/empty label files.")
