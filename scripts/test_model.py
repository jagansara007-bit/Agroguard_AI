import time
from ultralytics import YOLO
import os

# Configuration
MODEL_PATH = "runs/detect/backend/models/yolo_plant_detection18/weights/best.pt"
DATA_YAML = "data/detection/data.yaml"

def test_accuracy_and_efficiency():
    """Run validation and measure efficiency."""
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # Run validation
    print("Running validation on the validation set...")
    metrics = model.val(data=DATA_YAML)

    print("\n" + "="*40)
    print("ACCURACY RESULTS")
    print("="*40)
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    print("="*40)
    
    # Efficiency Testing
    print("\nMeasuring Inference Efficiency...")
    sample_images = [os.path.join("data/detection/images/val", f) for f in os.listdir("data/detection/images/val") if f.endswith(('.jpg', '.png', '.jpeg'))][:20]
    
    if not sample_images:
        print("No images found in data/detection/images/val for efficiency test.")
        return

    latencies = []
    # Warmup
    model.predict(source=sample_images[0], imgsz=640, verbose=False)
    
    for img_path in sample_images:
        start_time = time.time()
        model.predict(source=img_path, imgsz=640, verbose=False)
        end_time = time.time()
        latencies.append((end_time - start_time) * 1000) # ms

    avg_latency = sum(latencies) / len(latencies)
    print("\n" + "="*40)
    print("EFFICIENCY RESULTS (Inference Only)")
    print("="*40)
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"Images Processed: {len(latencies)}")
    print("="*40)

    # Run a test prediction on a few images for visual check
    print("\nRunning sample predictions for visual check...")
    results = model.predict(source="data/detection/images/val", save=True, imgsz=640, conf=0.25, max_det=10)
    
    print(f"\nSample prediction results saved to: {results[0].save_dir}")

if __name__ == "__main__":
    test_accuracy_and_efficiency()
