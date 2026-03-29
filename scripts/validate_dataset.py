import os
from pathlib import Path

def validate_labels(labels_dir):
    labels_dir = Path(labels_dir)
    if not labels_dir.exists():
        print(f"Directory {labels_dir} does not exist.")
        return

    count = 0
    bad_boxes = 0
    bad_coords = 0
    empty_files = 0
    
    for p in labels_dir.glob('*.txt'):
        count += 1
        has_content = False
        with open(p, 'r') as f:
            for line_idx, line in enumerate(f, 1):
                has_content = True
                parts = line.strip().split()
                if len(parts) < 5:
                    print(f"Malformed line in {p}:{line_idx} -> '{line.strip()}'")
                    continue
                
                try:
                    cls, x, y, w, h = map(float, parts)
                    if w <= 0 or h <= 0:
                        print(f"Zero dimension in {p}:{line_idx} -> {line.strip()}")
                        bad_boxes += 1
                    if x < 0 or y < 0 or x > 1 or y > 1 or w > 1 or h > 1:
                        print(f"Out of bounds in {p}:{line_idx} -> {line.strip()}")
                        bad_coords += 1
                except ValueError:
                    print(f"Non-float values in {p}:{line_idx} -> {line.strip()}")
        
        if not has_content:
            # Empty files are technically fine in YOLO (background images), but let's note them
            # empty_files += 1
            pass

    print(f"\nSummary for {labels_dir}:")
    print(f"  Total files: {count}")
    print(f"  Zero dimension boxes: {bad_boxes}")
    print(f"  Out of bounds coords: {bad_coords}")

if __name__ == "__main__":
    print("Validating train labels...")
    validate_labels('data/detection/labels/train')
    print("\nValidating val labels...")
    validate_labels('data/detection/labels/val')
