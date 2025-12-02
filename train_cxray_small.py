# train_cxr_small.py - QUICK TEST VERSION
# Trains on a small subset of data for testing (much faster!)
# Use this to verify your setup works before full training

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import cv2
from PIL import Image

# ---------------------------
# Config - SMALL VERSION FOR TESTING
# ---------------------------
DATA_DIR = "chest_xray"
BATCH_SIZE = 8              # Smaller batch size
NUM_EPOCHS = 5              # Only 5 epochs for medium test
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_OUT = "mobilenet_cxr_test.pth"
SEED = 42

# LIMITING DATA - Only use first N images per class
MAX_TRAIN_PER_CLASS = 250   # 250 NORMAL + 250 PNEUMONIA = 500 train images
MAX_VAL_PER_CLASS = 8       # 8 NORMAL + 8 PNEUMONIA = 16 val images
MAX_TEST_PER_CLASS = 100    # 100 NORMAL + 100 PNEUMONIA = 200 test images

print("="*70)
print("🚀 QUICK TEST MODE - Training on Small Dataset")
print("="*70)
print(f"Train images per class: {MAX_TRAIN_PER_CLASS}")
print(f"Val images per class:   {MAX_VAL_PER_CLASS}")
print(f"Test images per class:  {MAX_TEST_PER_CLASS}")
print(f"Total train images:     ~{MAX_TRAIN_PER_CLASS * 2}")
print(f"Epochs:                 {NUM_EPOCHS}")
print(f"Expected time:          ~5-15 minutes (CPU)")
print("="*70 + "\n")

# ---------------------------
# Reproducibility
# ---------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ---------------------------
# Helper to convert single-channel to 3-channel
# ---------------------------
class GrayToRGB(object):
    def __call__(self, img):
        arr = np.array(img)
        if len(arr.shape)==2:
            arr = np.stack([arr]*3, axis=-1)
        return transforms.ToPILImage()(arr)

# ---------------------------
# Data transforms
# ---------------------------
train_pipeline = transforms.Compose([
    GrayToRGB(),
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.485,0.485],[0.229,0.229,0.229])
])
val_pipeline = transforms.Compose([
    GrayToRGB(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.485,0.485],[0.229,0.229,0.229])
])

# ---------------------------
# Function to create subset of dataset
# ---------------------------
def create_subset(dataset, max_per_class):
    """Create a balanced subset with max_per_class samples per class"""
    targets = [dataset[i][1] for i in range(len(dataset))]
    class_indices = {}
    
    # Group indices by class
    for idx, target in enumerate(targets):
        if target not in class_indices:
            class_indices[target] = []
        class_indices[target].append(idx)
    
    # Select max_per_class from each class
    selected_indices = []
    for class_idx, indices in class_indices.items():
        random.shuffle(indices)  # Shuffle for randomness
        selected = indices[:max_per_class]
        selected_indices.extend(selected)
    
    return Subset(dataset, selected_indices)

# ---------------------------
# Load datasets
# ---------------------------
print("📁 Loading datasets...")
train_dir = os.path.join(DATA_DIR, "train")
val_dir = os.path.join(DATA_DIR, "val")
test_dir = os.path.join(DATA_DIR, "test")

# Load full datasets
train_dataset_full = datasets.ImageFolder(train_dir, transform=train_pipeline)
val_dataset_full = datasets.ImageFolder(val_dir, transform=val_pipeline)
test_dataset_full = datasets.ImageFolder(test_dir, transform=val_pipeline)

# Create small subsets
print(f"Creating small subsets for quick testing...")
train_dataset = create_subset(train_dataset_full, MAX_TRAIN_PER_CLASS)
val_dataset = create_subset(val_dataset_full, MAX_VAL_PER_CLASS)
test_dataset = create_subset(test_dataset_full, MAX_TEST_PER_CLASS)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)  # 0 for Windows compatibility
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

class_names = train_dataset_full.classes
print(f"✓ Classes: {class_names}")
print(f"✓ Train samples: {len(train_dataset)} (reduced from {len(train_dataset_full)})")
print(f"✓ Val samples: {len(val_dataset)} (reduced from {len(val_dataset_full)})")
print(f"✓ Test samples: {len(test_dataset)} (reduced from {len(test_dataset_full)})")
print()

# ---------------------------
# Model: Transfer learning MobileNetV2
# ---------------------------
print("🧠 Loading MobileNetV2 model...")
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
model = model.to(DEVICE)
print(f"✓ Model loaded on device: {DEVICE}")
print()

# Loss, optimizer, scheduler
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)

# ---------------------------
# Training loop
# ---------------------------
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return running_loss / total, correct / total

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Val/Test", leave=False):
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    return running_loss / total, correct / total, all_labels, all_preds

# ---------------------------
# Training
# ---------------------------
print("🎓 Starting training...")
print("="*70)

best_val_f1 = 0.0
history = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[]}

for epoch in range(1, NUM_EPOCHS+1):
    print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc, y_true_val, y_pred_val = validate(model, val_loader, criterion)
    
    val_f1 = f1_score(y_true_val, y_pred_val, average='binary')
    
    print(f"  Train loss: {train_loss:.4f}, acc: {train_acc:.4f}")
    print(f"  Val   loss: {val_loss:.4f}, acc: {val_acc:.4f}, f1: {val_f1:.4f}")
    
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    scheduler.step(val_f1)
    
    # Save best model
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), MODEL_OUT)
        print(f"  ✓ Saved best model: {MODEL_OUT}")

# ---------------------------
# Test evaluation
# ---------------------------
print("\n" + "="*70)
print("📊 Evaluating on test set...")
test_loss, test_acc, y_true, y_pred = validate(model, test_loader, criterion)
print(f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")

# Get classification report
report = classification_report(y_true, y_pred, target_names=class_names)
print("\nClassification Report:")
print(report)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)
print()

# ---------------------------
# Save all results to file
# ---------------------------
import json
from datetime import datetime

results_file = f"training_results_small_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
print(f"💾 Saving results to: {results_file}")

with open(results_file, 'w') as f:
    f.write("="*70 + "\n")
    f.write("CHEST X-RAY CLASSIFICATION - QUICK TEST RESULTS\n")
    f.write("="*70 + "\n\n")
    
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Device: {DEVICE}\n")
    f.write(f"Model: MobileNetV2\n")
    f.write(f"Model saved as: {MODEL_OUT}\n")
    f.write(f"Mode: QUICK TEST (limited data)\n\n")
    
    f.write("CONFIGURATION:\n")
    f.write(f"  Epochs: {NUM_EPOCHS}\n")
    f.write(f"  Batch Size: {BATCH_SIZE}\n")
    f.write(f"  Learning Rate: {LR}\n")
    f.write(f"  Max Train Per Class: {MAX_TRAIN_PER_CLASS}\n")
    f.write(f"  Max Val Per Class: {MAX_VAL_PER_CLASS}\n")
    f.write(f"  Max Test Per Class: {MAX_TEST_PER_CLASS}\n\n")
    
    f.write("DATASET SIZE:\n")
    f.write(f"  Training samples: {len(train_dataset)} (reduced from {len(train_dataset_full)})\n")
    f.write(f"  Validation samples: {len(val_dataset)} (reduced from {len(val_dataset_full)})\n")
    f.write(f"  Test samples: {len(test_dataset)} (reduced from {len(test_dataset_full)})\n")
    f.write(f"  Classes: {class_names}\n\n")
    
    f.write("="*70 + "\n")
    f.write("TRAINING HISTORY:\n")
    f.write("="*70 + "\n")
    for epoch in range(NUM_EPOCHS):
        f.write(f"\nEpoch {epoch+1}/{NUM_EPOCHS}:\n")
        f.write(f"  Train Loss: {history['train_loss'][epoch]:.4f}, Train Acc: {history['train_acc'][epoch]:.4f}\n")
        f.write(f"  Val Loss:   {history['val_loss'][epoch]:.4f}, Val Acc:   {history['val_acc'][epoch]:.4f}\n")
    
    f.write(f"\nBest Validation F1-Score: {best_val_f1:.4f}\n")
    
    f.write("\n" + "="*70 + "\n")
    f.write("TEST SET RESULTS:\n")
    f.write("="*70 + "\n")
    f.write(f"Test Loss: {test_loss:.4f}\n")
    f.write(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)\n\n")
    
    f.write("CLASSIFICATION REPORT:\n")
    f.write(str(report))
    f.write("\n")
    
    f.write("CONFUSION MATRIX:\n")
    f.write(f"                 Predicted\n")
    f.write(f"              NORMAL  PNEUMONIA\n")
    f.write(f"Actual NORMAL    {cm[0][0]:4d}     {cm[0][1]:4d}\n")
    f.write(f"       PNEUMONIA {cm[1][0]:4d}     {cm[1][1]:4d}\n\n")
    
    # Calculate additional metrics
    from sklearn.metrics import f1_score, precision_score, recall_score
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    
    f.write("SUMMARY METRICS:\n")
    f.write(f"  Overall F1-Score: {f1:.4f}\n")
    f.write(f"  Overall Precision: {precision:.4f}\n")
    f.write(f"  Overall Recall: {recall:.4f}\n\n")
    
    f.write("="*70 + "\n")
    f.write("NOTE: This is a quick test with limited data.\n")
    f.write("For production use, train with full dataset (train_cxray.py)\n")
    f.write("="*70 + "\n")

print(f"✓ Results saved to: {results_file}")

# Save training history as JSON
history_json = f"training_history_small_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(history_json, 'w') as f:
    json.dump({
        'mode': 'quick_test',
        'config': {
            'epochs': NUM_EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LR,
            'max_train_per_class': MAX_TRAIN_PER_CLASS,
            'max_val_per_class': MAX_VAL_PER_CLASS,
            'max_test_per_class': MAX_TEST_PER_CLASS,
            'device': str(DEVICE),
            'model': 'MobileNetV2'
        },
        'dataset_size': {
            'train': len(train_dataset),
            'val': len(val_dataset),
            'test': len(test_dataset),
            'train_full': len(train_dataset_full),
            'val_full': len(val_dataset_full),
            'test_full': len(test_dataset_full)
        },
        'history': history,
        'test_results': {
            'loss': test_loss,
            'accuracy': test_acc,
            'f1_score': f1,
            'precision': precision,
            'recall': recall
        },
        'best_val_f1': best_val_f1
    }, f, indent=2)
print(f"✓ History saved to: {history_json}")

# ---------------------------
# Plot training curves
# ---------------------------
print("📈 Generating training plots...")
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history['train_loss'], label='train_loss', marker='o')
plt.plot(history['val_loss'], label='val_loss', marker='o')
plt.legend()
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(alpha=0.3)

plt.subplot(1,2,2)
plt.plot(history['train_acc'], label='train_acc', marker='o')
plt.plot(history['val_acc'], label='val_acc', marker='o')
plt.legend()
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(alpha=0.3)

plt.tight_layout()
plot_file = f"training_history_small_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
plt.savefig(plot_file, dpi=150, bbox_inches='tight')
print(f"✓ Saved plot: {plot_file}")
plt.show()

print("="*70)
print("✅ QUICK TEST COMPLETE!")
print("="*70)
print("="*70)
print(f"\n📁 Output files created:")
print(f"   - {MODEL_OUT} (trained model)")
print(f"   - {results_file} (detailed results)")
print(f"   - {history_json} (JSON data)")
print(f"   - {plot_file} (training curves)")
print(f"\n💡 This was a quick test on {len(train_dataset)} training images")
print(f"   For full training with all {len(train_dataset_full)} images:")
print(f"   Run: python train_cxray.py")
print(f"\n🔮 To test predictions:")
print(f"   python predict.py --image chest_xray/test/NORMAL/IM-0001-0001.jpeg --model {MODEL_OUT}")
print("="*70)
