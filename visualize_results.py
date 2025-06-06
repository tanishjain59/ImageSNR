import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torchvision import transforms
from PIL import Image
import cv2
from enhancement import ImageEnhancer
from model import DrowsinessDetector
from data_utils import FL3DDataset, organize_dataset

def visualize_enhancement(image_paths, save_dir='visualization_results'):
    """Visualize original and enhanced images side by side."""
    os.makedirs(save_dir, exist_ok=True)
    enhancer = ImageEnhancer()
    
    plt.figure(figsize=(15, 10))
    for idx, img_path in enumerate(image_paths[:3]):  # Show first 3 images
        # Read and enhance image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        enhanced = enhancer.enhance(img)
        
        # Calculate metrics
        metrics = enhancer.calculate_metrics(img, enhanced)
        
        # Plot original and enhanced images
        plt.subplot(3, 2, idx*2 + 1)
        plt.imshow(img)
        plt.title(f'Original Image\nPSNR: {metrics["psnr"]:.2f}, SSIM: {metrics["ssim"]:.2f}')
        plt.axis('off')
        
        plt.subplot(3, 2, idx*2 + 2)
        plt.imshow(enhanced)
        plt.title('Enhanced Image')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'enhancement_comparison.png'))
    plt.close()

def evaluate_model(model, dataloader, device):
    """Evaluate model and return classification metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            preds = (outputs > 0.5).float().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds),
        'recall': recall_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds)
    }
    
    return metrics

def plot_metrics(metrics_raw, metrics_enhanced, save_dir='visualization_results'):
    """Plot comparison of classification metrics."""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    raw_values = [metrics_raw[m] for m in ['accuracy', 'precision', 'recall', 'f1']]
    enhanced_values = [metrics_enhanced[m] for m in ['accuracy', 'precision', 'recall', 'f1']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, raw_values, width, label='Raw Images')
    plt.bar(x + width/2, enhanced_values, width, label='Enhanced Images')
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Classification Metrics Comparison')
    plt.xticks(x, metrics)
    plt.legend()
    
    # Add value labels on top of bars
    for i, v in enumerate(raw_values):
        plt.text(i - width/2, v, f'{v:.3f}', ha='center', va='bottom')
    for i, v in enumerate(enhanced_values):
        plt.text(i + width/2, v, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_comparison.png'))
    plt.close()

def main():
    # Set device
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
    
    # Create datasets
    train_dataset = FL3DDataset(
        data_splits['train'][0],
        data_splits['train'][1],
        transform=transform,
        enhanced=False
    )
    
    train_dataset_enhanced = FL3DDataset(
        data_splits['train'][0],
        data_splits['train'][1],
        transform=transform,
        enhanced=True
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=False, num_workers=2)
    train_loader_enhanced = torch.utils.data.DataLoader(
        train_dataset_enhanced, batch_size=32, shuffle=False, num_workers=2)
    
    # Load model
    model = DrowsinessDetector(pretrained=True).to(device)
    model.load_state_dict(torch.load('checkpoints/best_model.pth'))
    
    # Visualize enhancement results
    print("Generating enhancement visualizations...")
    visualize_enhancement(data_splits['train'][0])
    
    # Evaluate and compare metrics
    print("Evaluating model performance...")
    metrics_raw = evaluate_model(model, train_loader, device)
    metrics_enhanced = evaluate_model(model, train_loader_enhanced, device)
    
    # Plot metrics comparison
    print("Generating metrics comparison...")
    plot_metrics(metrics_raw, metrics_enhanced)
    
    print("\nResults saved in 'visualization_results' directory:")
    print("1. enhancement_comparison.png - Visual comparison of original vs enhanced images")
    print("2. metrics_comparison.png - Comparison of classification metrics")
    
    # Print metrics
    print("\nClassification Metrics:")
    print("\nRaw Images:")
    for metric, value in metrics_raw.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    
    print("\nEnhanced Images:")
    for metric, value in metrics_enhanced.items():
        print(f"{metric.capitalize()}: {value:.4f}")

if __name__ == '__main__':
    main() 