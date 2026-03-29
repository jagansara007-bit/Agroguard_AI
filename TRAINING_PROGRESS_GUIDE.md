# Training Progress Monitoring

The YOLOv11 training process provides several ways to monitor progress.

## 1. Terminal Output
The terminal where you ran `python scripts/train_yolo.py` shows a real-time progress bar for each epoch, including:
- **GPU Memory Usage**
- **Losses** (Box, Class, DFL)
- **Instances**
- **Image Size**

## 2. Using `results.csv` (Currently Open)
You are currently viewing `results.csv`. This file updates after every epoch.
- **Current Status**: Epoch **46 of 200**.
- **Location**: `runs/detect/backend/models/yolo_plant_detection11/results.csv`

## 3. Visual Progress Plots
YOLO automatically generates plots in the run directory. You can open these files to see trends:
- **`results.png`**: Graphs for all metrics (mAP, Precision, Recall, Losses).
- **`train_batch*.jpg`**: Shows the images used in training with their labels.
- **`labels.jpg`**: Histogram of class distribution.

## 4. TensorBoard (Advanced)
If you have TensorBoard installed, you can visualize the training in your browser:
```powershell
tensorboard --logdir runs/detect/backend/models/yolo_plant_detection11
```
Then open `http://localhost:6006` in your browser.
