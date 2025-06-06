import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np

class FL3DDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, enhanced=False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.enhanced = enhanced
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not read image at {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
            
        label = self.labels[idx]
        return image, label

def load_annotations(annotation_file):
    """Load annotations from JSON file"""
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    return annotations

def process_annotations(annotations, data_dir):
    image_paths = []
    labels = []
    for rel_img_path, ann in annotations.items():
        # Remove leading './' and 'classification_frames/' if present
        rel_img_path_clean = rel_img_path.lstrip('./')
        if rel_img_path_clean.startswith('classification_frames/'):
            rel_img_path_clean = rel_img_path_clean[len('classification_frames/'):]
        img_path = os.path.join(data_dir, rel_img_path_clean)
        if not os.path.exists(img_path):
            continue
        driver_state = ann.get('driver_state', None)
        if driver_state is None:
            continue
        # You may want to filter/convert driver_state to binary (drowsy/not drowsy)
        # For now, treat 'drowsy' as 1, everything else as 0
        label = 1 if driver_state == 'drowsy' else 0
        image_paths.append(img_path)
        labels.append(label)
    return image_paths, labels

def organize_dataset(data_dir, test_size=0.15, val_size=0.15, random_state=42):
    """
    Organize the FL3D dataset into train/val/test splits
    """
    # Load annotations
    train_annotations = load_annotations(os.path.join(data_dir, 'annotations_train.json'))
    val_annotations = load_annotations(os.path.join(data_dir, 'annotations_val.json'))
    test_annotations = load_annotations(os.path.join(data_dir, 'annotations_test.json'))

    train_images, train_labels = process_annotations(train_annotations, data_dir)
    val_images, val_labels = process_annotations(val_annotations, data_dir)
    test_images, test_labels = process_annotations(test_annotations, data_dir)

    print(f"\nDataset sizes:")
    print(f"Train: {len(train_images)} images")
    print(f"Val: {len(val_images)} images")
    print(f"Test: {len(test_images)} images")
    
    return {
        'train': (train_images, train_labels),
        'val': (val_images, val_labels),
        'test': (test_images, test_labels)
    } 