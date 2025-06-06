import os
import torch
import numpy as np
from torchvision import transforms
from model import DrowsinessDetector
from data_utils import FL3DDataset, organize_dataset


def print_label_distribution(labels, split_name):
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\nLabel distribution for {split_name}:")
    for u, c in zip(unique, counts):
        print(f"  Label {u}: {c} samples")
    print(f"  Total: {len(labels)} samples")

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                         "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
    data_dir = 'data/classification_frames'
    data_splits = organize_dataset(data_dir)

    # Print label distributions
    for split in ['train', 'val', 'test']:
        labels = np.array(data_splits[split][1])
        print_label_distribution(labels, split)

    # Check model predictions on a batch
    print("\nLoading model and checking predictions on a batch...")
    model = DrowsinessDetector(pretrained=True).to(device)
    model.load_state_dict(torch.load('checkpoints/best_model.pth', map_location=device))
    model.eval()

    # Use train split for quick check
    dataset = FL3DDataset(
        data_splits['train'][0],
        data_splits['train'][1],
        transform=transform,
        enhanced=False
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, dtype=torch.float32)
            labels = labels.numpy()
            outputs = model(images)
            preds = (outputs > 0.5).float().cpu().numpy().squeeze()
            print("\nSample batch predictions and labels:")
            print(f"Predictions: {preds[:10]}")
            print(f"Labels:      {labels[:10]}")
            print(f"Unique predictions: {np.unique(preds)}")
            print(f"Unique labels:      {np.unique(labels)}")
            break  # Only check one batch

if __name__ == '__main__':
    main() 