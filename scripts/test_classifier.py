import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from pathlib import Path
import time
import os

# Configuration
DATA_DIR = Path("data/classification/val")
MODEL_PATH = Path("backend/models/classifier_efficientnet_b0_best.pth")
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(num_classes, model_type="efficientnet_b0"):
    """Load the trained model."""
    if model_type == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
    elif model_type == "resnet50":
        model = models.resnet50(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    return model, checkpoint['classes']

def test_classifier():
    if not MODEL_PATH.exists():
        print(f"Error: Classifier model not found at {MODEL_PATH}")
        return

    if not DATA_DIR.exists():
        print(f"Error: Validation data directory not found at {DATA_DIR}")
        return

    # Data transforms
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load data
    val_dataset = datasets.ImageFolder(DATA_DIR, transform=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Load model
    model, classes = load_model(len(val_dataset.classes))
    print(f"Loaded model from {MODEL_PATH}")
    print(f"Testing on {len(val_dataset)} images from {len(classes)} classes: {classes}")

    # Accuracy Measurement
    correct = 0
    total = 0
    
    # Efficiency Measurement
    latencies = []
    
    print("\nRunning evaluation...")
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            # Measure time for the batch (to get per-image average)
            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()
            
            batch_latency = (end_time - start_time) * 1000 # ms
            latencies.append(batch_latency / inputs.size(0))
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    avg_latency = sum(latencies) / len(latencies)

    print("\n" + "="*40)
    print("CLASSIFIER RESULTS")
    print("="*40)
    print(f"Accuracy:        {accuracy:.2f}%")
    print(f"Avg Latency:     {avg_latency:.2f} ms per image")
    print(f"Total Images:    {total}")
    print("="*40)

if __name__ == "__main__":
    test_classifier()
