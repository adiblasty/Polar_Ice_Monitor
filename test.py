from ultralytics import YOLO
import os

# === CONFIG ===
model_path = r'runs/segment/train8/weights/best.pt'  # Path to your trained model
source_dir = r'datasets/images/val'                 # Folder with images to test
output_dir = r'runs/segment/inference_output'       # Where predictions will be saved
os.makedirs(output_dir, exist_ok=True)

# Load model
model = YOLO(model_path)

# Run inference
results = model.predict(
    source=source_dir,
    save=True,            # Save predictions to output directory
    project='runs/segment',
    name='inference_output',
    exist_ok=True,        # Overwrite if folder exists
    show=False,           # Set True if you want to display results in a popup
    imgsz=416             # Match your training resolution
)

print(f"✅ Inference complete. Results saved to: {output_dir}")
