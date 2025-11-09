"""
Resize all images in train_image folder to 250x250
"""

import cv2
import numpy as np
from pathlib import Path

TRAIN_IMAGE_DIR = Path("./data/train_image")
TARGET_SIZE = (250, 250)

def resize_and_save(img_path, target_size=(250, 250)):
    """
    Read image, resize to target size, and save
    
    Args:
        img_path: Image path
        target_size: Target size (width, height)
    """
    # Read image
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  Warning: Unable to read {img_path}")
        return False
    
    # Resize (maintain aspect ratio, pad with black borders)
    h, w = img.shape[:2]
    scale = min(target_size[0] / w, target_size[1] / h)
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create black background
    result = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    
    # Calculate center position
    x_offset = (target_size[0] - new_w) // 2
    y_offset = (target_size[1] - new_h) // 2
    
    # Place resized image in center
    result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    # Save image
    cv2.imwrite(str(img_path), result)
    return True

def process_all_images():
    """Process all images"""
    print("="*80)
    print(f"Processing all images in train_image folder -> {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")
    print("="*80)
    
    # Statistics
    total_processed = 0
    total_failed = 0
    
    # Iterate through all class folders
    for class_dir in sorted(TRAIN_IMAGE_DIR.iterdir()):
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        print(f"\nProcessing class: {class_name}")
        
        # Find all jpg images (including subfolders)
        image_files = list(class_dir.rglob("*.jpg"))
        
        if len(image_files) == 0:
            print(f"  No images found")
            continue
        
        print(f"  Found {len(image_files)} images")
        
        # Process each image
        success_count = 0
        for idx, img_path in enumerate(image_files, 1):
            if resize_and_save(img_path, TARGET_SIZE):
                success_count += 1
                total_processed += 1
            else:
                total_failed += 1
            
            # Display progress every 10 images
            if idx % 10 == 0 or idx == len(image_files):
                print(f"  Progress: {idx}/{len(image_files)}", end='\r')
        
        print(f"  Successfully processed {success_count} images")
    
    print("\n" + "="*80)
    print("Processing Complete Statistics")
    print("="*80)
    print(f"Successfully processed: {total_processed} images")
    if total_failed > 0:
        print(f"Failed: {total_failed} images")
    print(f"Target size: {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")
    print("="*80)

if __name__ == "__main__":
    process_all_images()

