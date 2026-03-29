"""
Generate YOLO Pseudo-Labels
---------------------------
Since the PlantVillage dataset is for classification (no bounding boxes),
this script generates pseudo-labels for YOLO training.

It assumes the leaf is approximately in the center of the image and 
creates a bounding box covering 80% of the image area.

Output Format (YOLO .txt):
<class_id> <x_center> <y_center> <width> <height>
"""
import os
from pathlib import Path
from tqdm import tqdm
import yaml

# Configuration
SOURCE_DIR = Path("data/classification")
DEST_DIR = Path("data/detection")
# Map folder names to class IDs
# YOLO class IDs are 0-indexed.
CLASS_MAP = {
    'healthy': 1,   # healthy_leaf
    'diseased': 2   # diseased_leaf
}

def create_yolo_structure():
    """Create YOLO directory structure and labels."""
    
    # 1. Setup Directories
    for split in ['train', 'val']:
        (DEST_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (DEST_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
    print("Generating YOLO labels from classification data...")
    
    # 2. Process Images
    total_images = 0
    
    for split in ['train', 'val']:
        print(f"Processing {split} split...")
        source_split = SOURCE_DIR / split
        
        # We process healthy and diseased folders
        for category in ['healthy', 'diseased']:
            source_cat = source_split / category
            if not source_cat.exists():
                continue
                
            class_id = CLASS_MAP[category]
            
            # Using list(glob) to get a count for tqdm
            files = list(source_cat.glob("*.JPG"))
            for img_path in tqdm(files, desc=f"{split}/{category}"):
                # Copy image (symlinks might be tricky on basic Windows installs without dev mode)
                dest_img_path = DEST_DIR / 'images' / split / img_path.name
                
                if not dest_img_path.exists():
                    import shutil
                    shutil.copy2(img_path, dest_img_path)
                
                # Create Pseudo-Label
                # Class ID, Center X, Center Y, Width, Height
                # Box covers 80% of image, centered
                label_content = f"{class_id} 0.5 0.5 0.8 0.8\n"
                
                label_path = DEST_DIR / 'labels' / split / (img_path.stem + ".txt")
                with open(label_path, "w") as f:
                    f.write(label_content)
                    
                total_images += 1

    print(f"\nCreated pseudo-labels for {total_images} images.")
    print(f"Data ready in {DEST_DIR}")

if __name__ == "__main__":
    create_yolo_structure()
