# Training Scripts

This directory contains scripts for training the AgroGuard AI models.

## Scripts

### 1. `organize_data.py`
Organizes the PlantVillage dataset into our classification structure.

**Usage:**
```bash
python scripts/organize_data.py
```

This will:
- Read from `extracted_data/New Plant Diseases Dataset(Augmented)/`
- Categorize images as "healthy" or "diseased"
- Copy to `data/classification/train/` and `data/classification/val/`

### 2. `train_classifier.py`
Trains a ResNet50 or EfficientNet model for disease classification.

**Usage:**
```bash
python scripts/train_classifier.py
```

**Configuration** (edit in script):
- `MODEL_TYPE`: "resnet50" or "efficientnet_b0"
- `BATCH_SIZE`: 32 (adjust based on GPU memory)
- `EPOCHS`: 20
- `LEARNING_RATE`: 0.001

**Output:**
- Trained model saved to `backend/models/classifier_<model>_best.pth`

### 3. `train_yolo.py`
Trains YOLOv11 for plant detection (requires labeled data).

**Usage:**
```bash
python scripts/train_yolo.py
```

**Note:** YOLO training requires YOLO-format labels. The PlantVillage dataset doesn't include bounding boxes, so this is a template for when you have labeled detection data.

## Workflow

1. **Extract and organize data:**
   ```bash
   # Archive extraction happens automatically
   python scripts/organize_data.py
   ```

2. **Train classifier:**
   ```bash
   python scripts/train_classifier.py
   ```

3. **Update backend to use trained model:**
   - Edit `backend/main.py`
   - Load your trained model instead of the pretrained ResNet50

## Requirements

All dependencies are in `backend/requirements.txt`:
```bash
pip install -r backend/requirements.txt
```
