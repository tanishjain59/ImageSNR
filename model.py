import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from tqdm import tqdm

class DrowsinessDetector(nn.Module):
    def __init__(self, pretrained=True):
        super(DrowsinessDetector, self).__init__()
        # Use ResNet18 as backbone with modern weights API
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        
        # Replace the final layer for binary classification
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
    
    def forward(self, x):
        return self.backbone(x)

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # Create progress bar
    pbar = tqdm(dataloader, desc='Training', leave=False)
    
    for batch_idx, (images, labels) in enumerate(pbar):
        # Ensure consistent tensor types
        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.float32)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels.unsqueeze(1))
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        probs = torch.sigmoid(outputs)
        predicted = (probs > 0.5).float()
        total += labels.size(0)
        correct += (predicted.squeeze() == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct/total:.4f}'
        })
    
    return total_loss / len(dataloader), correct / total

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    # Create progress bar
    pbar = tqdm(dataloader, desc='Evaluating', leave=False)
    
    with torch.no_grad():
        for images, labels in pbar:
            # Ensure consistent tensor types
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)
            
            outputs = model(images)
            loss = criterion(outputs, labels.unsqueeze(1))
            
            total_loss += loss.item()
            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).float()
            total += labels.size(0)
            correct += (predicted.squeeze() == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct/total:.4f}'
            })
    
    return total_loss / len(dataloader), correct / total 