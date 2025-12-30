import os
import torch
from ultralytics import YOLO

# --- FIX FOR OMP ERROR ---
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# -------------------------

def train_model():
    # Load the model
    model = YOLO('yolov8n.pt')

    # Start training
    results = model.train(
        data='NEU-defects.yaml',
        epochs=50,
        imgsz=640,
        batch=16,            # Increased from 8 to utilize your 3080 Ti better
        name='NEU_defect_model_v1',
        device=0,            # <--- This forces the use of your NVIDIA GPU
        workers=8            # Increases data loading speed (optional, adjust if CPU struggles)
    )
    print("Training complete! Results are saved in the 'runs/detect/' directory.")

if __name__ == '__main__':
    # Double check GPU is visible before starting
    if torch.cuda.is_available():
        print(f"🚀 Starting training on {torch.cuda.get_device_name(0)}")
        train_model()
    else:
        print("❌ Error: GPU not detected by PyTorch, despite previous checks.")