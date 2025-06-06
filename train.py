import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import os
from tqdm import tqdm
import warnings
import time

from data_utils import FL3DDataset, organize_dataset
from model import DrowsinessDetector, train_epoch, evaluate
from enhancement import ImageEnhancer

def get_device():
    if torch.backends.mps.is_available():
        # Set MPS specific configurations
        torch.backends.mps.enable_fallback_to_cpu = True
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def main(args):
    print("\n=== Starting Training Process ===")
    start_time = time.time()
    
    # Set device
    device = get_device()
    print(f"Using device: {device}")
    
    # Data transforms
    print("\nSetting up data transforms...")
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Organize dataset
    print("\nOrganizing dataset...")
    data_splits = organize_dataset(args.data_dir)
    print(f"Dataset sizes:")
    print(f"Train: {len(data_splits['train'][0])} images")
    print(f"Val: {len(data_splits['val'][0])} images")
    print(f"Test: {len(data_splits['test'][0])} images")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = FL3DDataset(
        data_splits['train'][0],
        data_splits['train'][1],
        transform=transform,
        enhanced=args.use_enhanced
    )
    
    val_dataset = FL3DDataset(
        data_splits['val'][0],
        data_splits['val'][1],
        transform=transform,
        enhanced=args.use_enhanced
    )
    
    # Create dataloaders with optimized settings for MPS
    print("\nSetting up data loaders...")
    num_workers = 2 if device.type == 'mps' else 4
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=num_workers,
                            pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                          shuffle=False, num_workers=num_workers,
                          pin_memory=True)
    print(f"Batch size: {args.batch_size}")
    print(f"Number of workers: {num_workers}")
    
    # Initialize model
    print("\nInitializing model...")
    model = DrowsinessDetector(pretrained=True).to(device)
    print("Model architecture:")
    print(model)
    
    # Loss and optimizer
    print("\nSetting up loss function and optimizer...")
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)
    print(f"Learning rate: {args.lr}")
    
    # Training loop
    print("\n=== Starting Training Loop ===")
    best_val_acc = 0
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 50)
        
        try:
            # Training phase
            print("\nTraining phase:")
            train_loss, train_acc = train_epoch(model, train_loader, criterion,
                                              optimizer, device)
            
            # Validation phase
            print("\nValidation phase:")
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Print epoch results
            print(f"\nEpoch {epoch+1} Results:")
            print(f"Time taken: {epoch_time:.2f} seconds")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Learning rate adjustment
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_acc)
            new_lr = optimizer.param_groups[0]['lr']
            if old_lr != new_lr:
                print(f"Learning rate adjusted: {old_lr:.6f} -> {new_lr:.6f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_path = os.path.join(args.save_dir, 'best_model.pth')
                torch.save(model.state_dict(), save_path)
                print(f"\nNew best model saved! (Validation Accuracy: {val_acc:.4f})")
            
            # Clear memory cache for MPS device
            if device.type == 'mps':
                torch.mps.empty_cache()
                
        except RuntimeError as e:
            if "mps" in str(e).lower():
                print(f"\nWarning: MPS error occurred, falling back to CPU for this batch")
                # Fall back to CPU for this batch
                model = model.cpu()
                train_loss, train_acc = train_epoch(model, train_loader, criterion,
                                                  optimizer, torch.device('cpu'))
                model = model.to(device)
            else:
                raise e
    
    # Training completion
    total_time = time.time() - start_time
    print("\n=== Training Complete ===")
    print(f"Total training time: {total_time/60:.2f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved at: {os.path.join(args.save_dir, 'best_model.pth')}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Path to dataset directory')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                      help='Directory to save model checkpoints')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--use_enhanced', action='store_true',
                      help='Use enhanced images for training')
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    main(args) 