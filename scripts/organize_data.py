"""
Data Organization Script
Organizes the extracted PlantVillage dataset into our classification structure.
"""
import os
import shutil
from pathlib import Path
from collections import defaultdict

# Paths
SOURCE_DIR = Path("data/raw/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)")
DEST_DIR = Path("data/classification")

def organize_dataset():
    """Organize images into healthy/diseased categories."""
    
    # Create counters
    stats = defaultdict(int)
    
    for split in ['train', 'valid']:
        source_split = SOURCE_DIR / split
        dest_split = DEST_DIR / ('train' if split == 'train' else 'val')
        
        if not source_split.exists():
            print(f"Warning: Source directory not found: {source_split}")
            continue
            
        print(f"\nProcessing {split} split...")
        
        # Iterate through all disease folders
        for disease_folder in source_split.iterdir():
            if not disease_folder.is_dir():
                continue
                
            disease_name = disease_folder.name
            
            dest_category = dest_split / disease_name
            dest_category.mkdir(parents=True, exist_ok=True)
            
            # Copy images
            for img_file in disease_folder.glob('*.JPG'):
                # Create unique filename with disease prefix
                new_name = f"{disease_name}_{img_file.name}"
                dest_path = dest_category / new_name
                
                shutil.copy2(img_file, dest_path)
                stats[f"{split}_{disease_name}"] += 1
                
                if stats[f"{split}_{disease_name}"] % 500 == 0:
                    print(f"  {disease_name}: {stats[f'{split}_{disease_name}']} images")
    
    # Print summary
    print("\n" + "="*50)
    print("ORGANIZATION COMPLETE")
    print("="*50)
    for key, count in sorted(stats.items()):
        print(f"{key}: {count:,} images")
    print("="*50)

if __name__ == "__main__":
    print("Starting dataset organization...")
    organize_dataset()
    print("\nDataset ready for training!")
