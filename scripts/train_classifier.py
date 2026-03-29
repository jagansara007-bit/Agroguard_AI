"""
Classification Model Training Script
Trains a ResNet50 or EfficientNet model for plant disease classification.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path
import time
from torch.utils.tensorboard import SummaryWriter


# Configuration
DATA_DIR = Path("data/classification")
MODEL_TYPE = "efficientnet_b0"  # or "resnet50"
BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 0.001
NUM_CLASSES = 38 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ... (omitted for brevity, just replacing the config block or relevant lines)
# Wait, replace_file_content is for contiguous block.
# I'll just replace the config section.


# Data augmentation and normalization
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_data():
    """Load training and validation datasets."""
    train_dataset = datasets.ImageFolder(DATA_DIR / 'train', transform=train_transforms)
    val_dataset = datasets.ImageFolder(DATA_DIR / 'val', transform=val_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"📊 Training samples: {len(train_dataset)}")
    print(f"📊 Validation samples: {len(val_dataset)}")
    print(f"📊 Classes: {train_dataset.classes}")
    
    return train_loader, val_loader, train_dataset.classes

def create_model():
    """Create and configure the model."""
    if MODEL_TYPE == "resnet50":
        model = models.resnet50(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, NUM_CLASSES)
    elif MODEL_TYPE == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=True)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model type: {MODEL_TYPE}")
    
    return model.to(DEVICE)

def train_epoch(model, train_loader, criterion, optimizer):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def train():
    """Main training function."""
    print(f"🚀 Starting training on {DEVICE}")
    print(f"📦 Model: {MODEL_TYPE}")
    print(f"🔢 Batch size: {BATCH_SIZE}")
    print(f"📈 Epochs: {EPOCHS}")
    print("="*60)
    
    # Load data
    train_loader, val_loader, classes = load_data()
    
    # Create model
    model = create_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    writer = SummaryWriter(f'backend/runs/classification/{MODEL_TYPE}')

    
    # Training loop
    best_val_acc = 0.0
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        scheduler.step(val_loss)
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{EPOCHS} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'classes': classes
            }, f'backend/models/classifier_{MODEL_TYPE}_best.pth')
            print(f"  ✅ Saved best model (Val Acc: {val_acc:.2f}%)")
        
        print("-"*60)
    
    print(f"\n🎉 Training complete! Best Val Acc: {best_val_acc:.2f}%")
    print(f"💾 Model saved to: backend/models/classifier_{MODEL_TYPE}_best.pth")

if __name__ == "__main__":
    # Create models directory
    Path("backend/models").mkdir(parents=True, exist_ok=True)
    train()
