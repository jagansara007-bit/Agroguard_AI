import os
import shutil
from pathlib import Path

def fix_dataset():
    root = Path(".")
    detection_img_root = root / "data" / "detection" / "images"
    classification_root = root / "data" / "classification"
    
    # Map filenames to their actual locations in classification folder
    filename_map = {}
    print("Mapping classification images...")
    for img_path in classification_root.rglob("*"):
        if img_path.is_file() and img_path.suffix.lower() in ('.jpg', '.png', '.jpeg'):
            filename_map[img_path.name] = img_path
            
    print(f"Found {len(filename_map)} source images in classification folder.")
    
    for split in ["train", "val"]:
        split_dir = detection_img_root / split
        if not split_dir.exists():
            print(f"Skipping {split_dir} as it does not exist.")
            continue
            
        print(f"Fixing images in {split_dir}...")
        count = 0
        fixed = 0
        skipped = 0
        
        for p in split_dir.glob("*.JPG"): # The placeholders have .JPG extension
            if p.stat().st_size < 1000: # It's a placeholder if it's very small
                filename = p.name
                if filename in filename_map:
                    shutil.copy2(filename_map[filename], p)
                    fixed += 1
                else:
                    # Try to see if it's a different case or slight mismatch
                    found = False
                    for k in filename_map:
                        if k.lower() == filename.lower():
                            shutil.copy2(filename_map[k], p)
                            fixed += 1
                            found = True
                            break
                    if not found:
                        skipped += 1
            count += 1
        
        print(f"Finished {split}: {fixed} fixed, {skipped} skipped, out of {count} files.")

if __name__ == "__main__":
    fix_dataset()
