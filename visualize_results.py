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
import argparse
from tqdm import tqdm

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

def visualize_enhancement_with_predictions(image_paths, model, transform, device, labels, save_dir='visualization_results'):
    """Visualize original and enhanced images side by side with predictions and ground truth, using color coding."""
    import matplotlib.patches as patches
    os.makedirs(save_dir, exist_ok=True)
    enhancer = ImageEnhancer()
    plt.figure(figsize=(15, 12))
    legend_handles = []
    
    for idx, img_path in enumerate(image_paths[:3]):  # Show first 3 images
        # Read and enhance image
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        enhanced = enhancer.enhance(img_rgb)
        gt_label = labels[idx]
        
        # Prepare images for model
        raw_tensor = transform(img_rgb).unsqueeze(0).to(device)
        enh_tensor = transform(enhanced).unsqueeze(0).to(device)
        
        # Get predictions
        model.eval()
        with torch.no_grad():
            raw_pred = (torch.sigmoid(model(raw_tensor)).cpu().item() > 0.5)
            enh_pred = (torch.sigmoid(model(enh_tensor)).cpu().item() > 0.5)
        
        # Determine correctness
        raw_correct = (raw_pred == gt_label)
        enh_correct = (enh_pred == gt_label)
        
        # Color coding
        raw_color = 'green' if raw_correct else 'red'
        enh_color = 'green' if enh_correct else 'red'
        
        # Special marker if enhancement corrects a misclassification
        improved = (not raw_correct) and enh_correct
        
        # Calculate metrics
        metrics = enhancer.calculate_metrics(img_rgb, enhanced)
        
        # Plot original image
        ax1 = plt.subplot(3, 2, idx*2 + 1)
        ax1.imshow(img_rgb)
        title = f"Original\nGT: {gt_label} | Pred: {int(raw_pred)}\nPSNR: {metrics['psnr']:.2f}, SSIM: {metrics['ssim']:.2f}"
        ax1.set_title(title, color=raw_color)
        ax1.axis('off')
        # Add colored border
        for spine in ax1.spines.values():
            spine.set_edgecolor(raw_color)
            spine.set_linewidth(4)
        
        # Plot enhanced image
        ax2 = plt.subplot(3, 2, idx*2 + 2)
        ax2.imshow(enhanced)
        enh_title = f"Enhanced\nGT: {gt_label} | Pred: {int(enh_pred)}"
        if improved:
            enh_title += "\nImproved!"
        ax2.set_title(enh_title, color=enh_color)
        ax2.axis('off')
        for spine in ax2.spines.values():
            spine.set_edgecolor(enh_color)
            spine.set_linewidth(4)
        
    # Add legend
    import matplotlib.lines as mlines
    green_line = mlines.Line2D([], [], color='green', marker='s', linestyle='None', markersize=10, label='Correct')
    red_line = mlines.Line2D([], [], color='red', marker='s', linestyle='None', markersize=10, label='Incorrect')
    star = mlines.Line2D([], [], color='black', marker='*', linestyle='None', markersize=10, label='Improved (Enhanced corrects Raw)')
    plt.figlegend(handles=[green_line, red_line, star], loc='lower center', ncol=3, fontsize=12, frameon=False)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(os.path.join(save_dir, 'enhancement_comparison.png'))
    plt.close()

def evaluate_model(model, dataloader, device):
    """Evaluate model and return classification metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Evaluating', leave=False):
            images = images.to(device)
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy()
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

def main(args):
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
    
    # Create val and test datasets
    val_dataset = FL3DDataset(
        data_splits['val'][0],
        data_splits['val'][1],
        transform=transform,
        enhanced=False
    )
    val_dataset_enhanced = FL3DDataset(
        data_splits['val'][0],
        data_splits['val'][1],
        transform=transform,
        enhanced=True
    )
    test_dataset = FL3DDataset(
        data_splits['test'][0],
        data_splits['test'][1],
        transform=transform,
        enhanced=False
    )
    test_dataset_enhanced = FL3DDataset(
        data_splits['test'][0],
        data_splits['test'][1],
        transform=transform,
        enhanced=True
    )
    # Create dataloaders
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=2)
    val_loader_enhanced = torch.utils.data.DataLoader(
        val_dataset_enhanced, batch_size=32, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=2)
    test_loader_enhanced = torch.utils.data.DataLoader(
        test_dataset_enhanced, batch_size=32, shuffle=False, num_workers=2)
    
    # Load models
    model_raw = DrowsinessDetector(pretrained=True).to(device)
    model_raw.load_state_dict(torch.load(args.raw_model, map_location=device))
    model_enhanced = DrowsinessDetector(pretrained=True).to(device)
    model_enhanced.load_state_dict(torch.load(args.enhanced_model, map_location=device))
    
    # Visualize enhancement results with predictions (using enhanced model, val set)
    print("Generating enhancement visualizations with predictions...")
    visualize_enhancement_with_predictions(data_splits['val'][0], model_enhanced, transform, device, data_splits['val'][1])
    
    # Evaluate and compare metrics on val set
    print("Evaluating model performance on validation set...")
    metrics_val_raw = evaluate_model(model_raw, val_loader, device)
    metrics_val_enhanced = evaluate_model(model_enhanced, val_loader_enhanced, device)
    print("\nValidation Set Metrics:")
    print("Raw Images:")
    for metric, value in metrics_val_raw.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    print("Enhanced Images:")
    for metric, value in metrics_val_enhanced.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    plot_metrics(metrics_val_raw, metrics_val_enhanced, save_dir='visualization_results')
    
    # Evaluate and compare metrics on test set
    print("\nEvaluating model performance on test set...")
    metrics_test_raw = evaluate_model(model_raw, test_loader, device)
    metrics_test_enhanced = evaluate_model(model_enhanced, test_loader_enhanced, device)
    print("\nTest Set Metrics:")
    print("Raw Images:")
    for metric, value in metrics_test_raw.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    print("Enhanced Images:")
    for metric, value in metrics_test_enhanced.items():
        print(f"{metric.capitalize()}: {value:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_model', type=str, default='checkpoints/best_model_raw.pth')
    parser.add_argument('--enhanced_model', type=str, default='checkpoints/best_model_enhanced.pth')
    args = parser.parse_args()
    main(args) 