import os
import json
from data_utils import organize_dataset, load_annotations
from enhancement import ImageEnhancer
import cv2
import numpy as np

def verify_dataset():
    # Check if data directory exists
    data_dir = 'data/classification_frames'
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory not found at {data_dir}")
    
    # Check annotation files
    print("Checking annotation files...")
    annotation_files = ['annotations_train.json', 'annotations_val.json', 'annotations_test.json']
    for file in annotation_files:
        file_path = os.path.join(data_dir, file)
        if not os.path.exists(file_path):
            print(f"Warning: Annotation file not found: {file_path}")
            continue
        try:
            with open(file_path, 'r') as f:
                annotations = json.load(f)
            print(f"✓ {file}: {len(annotations)} video sequences found")
        except Exception as e:
            print(f"Error reading {file}: {str(e)}")
    
    # Try to organize dataset
    print("\nOrganizing dataset...")
    data_splits = organize_dataset(data_dir)
    
    # Verify a few images from each split
    print("\nVerifying images...")
    for split_name, (images, labels) in data_splits.items():
        print(f"\nChecking {split_name} split:")
        if len(images) == 0:
            print(f"Warning: No images found in {split_name} split")
            continue
            
        for i in range(min(3, len(images))):
            img_path = images[i]
            if not os.path.exists(img_path):
                print(f"Warning: Image not found: {img_path}")
                continue
            
            # Try to read image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image: {img_path}")
                continue
            
            print(f"✓ Image {i+1}: {img_path} (shape: {img.shape}, label: {labels[i]})")
    
    # Only test enhancement if we have images
    if len(images) > 0:
        print("\nTesting enhancement pipeline...")
        enhancer = ImageEnhancer()
        test_img = cv2.imread(images[0])
        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        enhanced = enhancer.enhance(test_img)
        metrics = enhancer.calculate_metrics(test_img, enhanced)
        print(f"Enhancement metrics: PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.2f}")

if __name__ == "__main__":
    verify_dataset() 