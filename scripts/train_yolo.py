"""
YOLOv11 Training Script
Note: YOLO training requires labeled data. This is a template.
For actual use, you'll need to create YOLO format labels for your images.
"""
from ultralytics import YOLO
from pathlib import Path
import os

# Configuration
DATA_YAML = "data/detection/data.yaml"
MODEL_SIZE = "yolo11n.pt"  # nano, small, medium, large, xlarge
EPOCHS = 30
BATCH_SIZE = 16
IMG_SIZE = 640
DEVICE = 0  # Use GPU 0 (RTX 3050)

def train_yolo():
    """Train YOLOv11 model."""
    print("Starting YOLOv11 training...")
    print(f"Base model: {MODEL_SIZE}")
    print(f"Config: {DATA_YAML}")
    print(f"Epochs: {EPOCHS}")
    
    # Check GPU availability
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU Detected: {gpu_name}")
        device = 0
    else:
        print("WARNING: CUDA not available! Falling back to CPU.")
        device = 'cpu'
        
    print("="*60)
    
    # Path to last checkpoint for resuming
    # Note: Using Run 8 as it is the most complete run (35 epochs)
    LAST_CHECKPOINT = "runs/detect/backend/models/yolo_plant_detection11/weights/last.pt"
    
    if Path(LAST_CHECKPOINT).exists():
        print(f"Resuming from checkpoint: {LAST_CHECKPOINT}")
        model = YOLO(LAST_CHECKPOINT)
        resume_training = True
    else:
        # Load pretrained model
        model = YOLO(MODEL_SIZE)
        resume_training = False
    
    # Train the model
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        optimizer="AdamW",
        lr0=1e-3,
        lrf=1e-2,
        momentum=0.9,
        weight_decay=5e-4,
        warmup_epochs=5,
        cos_lr=True,
        patience=30,
        device=device,
        project="backend/models",
        name="yolo_plant_detection",
        save=True,
        save_period=1,
        plots=True,
        verbose=True,
        resume=False,
        amp=False     # Disable AMP to avoid numerical overflows
    )
    
    # Validate
    metrics = model.val()
    
    print("\n" + "="*60)
    print("Training Results:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print("="*60)
    
    # Export to ONNX
    print("Exporting model to ONNX...")
    model.export(format='onnx')
    
    print(f"\nModel saved to: backend/models/yolo_plant_detection/weights/best.pt")

if __name__ == "__main__":
    # Ensure data directory exists
    if not Path(DATA_YAML).exists():
        print(f"Error: {DATA_YAML} not found!")
        print("Please ensure your dataset is properly organized with labels.")
        exit(1)
    
    train_yolo()
