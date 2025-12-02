# train_cxr_mobilenet.py
# Works in Colab and VS Code
# Requirements: torch torchvision matplotlib scikit-learn opencv-python tqdm
# To run in Colab: pip install -q torch torchvision matplotlib scikit-learn opencv-python tqdm

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import cv2
from PIL import Image

# ---------------------------
# Config
# ---------------------------
DATA_DIR = "chest_xray"  # folder that contains train/val/test subfolders
BATCH_SIZE = 16
NUM_EPOCHS = 8
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_OUT = "mobilenet_cxr.pth"
SEED = 42

# ---------------------------
# Reproducibility
# ---------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ---------------------------
# Data transforms
# ---------------------------
train_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485],[0.229])  # grayscale mean/std placeholder (we'll duplicate channel)
])

val_test_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485],[0.229])
])

# Helper to convert single-channel to 3-channel by duplicating
class GrayToRGB(object):
    def __call__(self, img):
        # img: PIL Image
        arr = np.array(img)
        if len(arr.shape)==2:
            arr = np.stack([arr]*3, axis=-1)
        return transforms.ToPILImage()(arr)

# Pipeline with GrayToRGB for datasets
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
# Load datasets (expects folder structure: DATA_DIR/train/NORMAL, DATA_DIR/train/PNEUMONIA ...)
# ---------------------------
train_dir = os.path.join(DATA_DIR, "train")
val_dir = os.path.join(DATA_DIR, "val")
test_dir = os.path.join(DATA_DIR, "test")

train_dataset = datasets.ImageFolder(train_dir, transform=train_pipeline)
val_dataset = datasets.ImageFolder(val_dir, transform=val_pipeline)
test_dataset = datasets.ImageFolder(test_dir, transform=val_pipeline)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)  # Changed to 0 for Windows
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)  # Changed to 0 for Windows
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)  # Changed to 0 for Windows

class_names = train_dataset.classes
print("Classes:", class_names)
print("Train samples:", len(train_dataset), "Val:", len(val_dataset), "Test:", len(test_dataset))

# ---------------------------
# Model: Transfer learning MobileNetV2
# ---------------------------
model = models.mobilenet_v2(pretrained=True)
# replace classifier
model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
model = model.to(DEVICE)

# Loss, optimizer, scheduler
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

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

best_val_f1 = 0.0
history = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[]}

for epoch in range(1, NUM_EPOCHS+1):
    print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc, _, _ = validate(model, val_loader, criterion)
    # compute a simple val_f1 via predictions (run validation preds)
    _, _, y_true_val, y_pred_val = validate(model, val_loader, criterion)
    # f1
    from sklearn.metrics import f1_score
    val_f1 = f1_score(y_true_val, y_pred_val, average='binary')
    print(f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f}")
    print(f"Val   loss: {val_loss:.4f}, acc: {val_acc:.4f}, f1: {val_f1:.4f}")
    history['train_loss'].append(train_loss); history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss); history['val_acc'].append(val_acc)
    scheduler.step(val_f1)
    # Save best model by val_f1
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), MODEL_OUT)
        print("Saved best model:", MODEL_OUT)

# ---------------------------
# Test evaluation
# ---------------------------
print("\nEvaluating on test set")
test_loss, test_acc, y_true, y_pred = validate(model, test_loader, criterion)
print(f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")

# Get classification report as string
report = classification_report(y_true, y_pred, target_names=class_names)
print("Classification Report:")
print(report)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion matrix:\n", cm)

# ---------------------------
# Save all results to file
# ---------------------------
import json
from datetime import datetime

results_file = f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
print(f"\n💾 Saving results to: {results_file}")

with open(results_file, 'w') as f:
    f.write("="*70 + "\n")
    f.write("CHEST X-RAY CLASSIFICATION - TRAINING RESULTS\n")
    f.write("="*70 + "\n\n")
    
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Device: {DEVICE}\n")
    f.write(f"Model: MobileNetV2\n")
    f.write(f"Model saved as: {MODEL_OUT}\n\n")
    
    f.write("CONFIGURATION:\n")
    f.write(f"  Epochs: {NUM_EPOCHS}\n")
    f.write(f"  Batch Size: {BATCH_SIZE}\n")
    f.write(f"  Learning Rate: {LR}\n")
    f.write(f"  Dataset: {DATA_DIR}\n\n")
    
    f.write("DATASET SIZE:\n")
    f.write(f"  Training samples: {len(train_dataset)}\n")
    f.write(f"  Validation samples: {len(val_dataset)}\n")
    f.write(f"  Test samples: {len(test_dataset)}\n")
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
    f.write("END OF REPORT\n")
    f.write("="*70 + "\n")

print(f"✓ Results saved to: {results_file}")

# Save training history as JSON for later analysis
history_json = f"training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(history_json, 'w') as f:
    json.dump({
        'config': {
            'epochs': NUM_EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LR,
            'device': str(DEVICE),
            'model': 'MobileNetV2',
            'dataset': DATA_DIR
        },
        'dataset_size': {
            'train': len(train_dataset),
            'val': len(val_dataset),
            'test': len(test_dataset)
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

# Plot training curves
plt.figure(figsize=(10,4))
plt.subplot(1,2,1); plt.plot(history['train_loss'], label='train_loss'); plt.plot(history['val_loss'], label='val_loss'); plt.legend(); plt.title('Loss')
plt.subplot(1,2,2); plt.plot(history['train_acc'], label='train_acc'); plt.plot(history['val_acc'], label='val_acc'); plt.legend(); plt.title('Accuracy')
plot_file = f"training_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
plt.savefig(plot_file, dpi=150, bbox_inches='tight')
print(f"✓ Training plot saved to: {plot_file}")
plt.show()

# ---------------------------
# Grad-CAM helper (simple implementation)
# ---------------------------
def show_gradcam(model, img_path, transform, target_class=None):
    model.eval()
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image from {img_path}")
    img_resized = cv2.resize(img, (224, 224))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)  # 3-channel
    inp = transform(Image.fromarray(img_rgb)).unsqueeze(0).to(DEVICE)
    # register hook to last conv layer
    final_conv = model.features[-1]  # for mobilenet_v2
    grads = []
    acts = []
    def save_grad(g):
        grads.append(g.cpu().detach().numpy())
    def forward_hook(module, inp, out):
        acts.append(out.cpu().detach().numpy())
    h1 = final_conv.register_forward_hook(forward_hook)
    # forward
    out = model(inp)
    if target_class is None:
        pred_class = out.argmax(dim=1).item()
    else:
        pred_class = target_class
    # backward on the predicted class
    model.zero_grad()
    one_hot = torch.zeros_like(out); one_hot[0, pred_class] = 1
    out.backward(gradient=one_hot, retain_graph=True)
    # remove hooks
    h1.remove()
    # compute CAM (this is a crude CAM)
    activation = acts[0][0]  # C x H x W
    weights = np.mean(grads[0][0], axis=(1,2)) if grads else np.mean(activation, axis=(1,2))
    cam = np.zeros(activation.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * activation[i]
    cam = np.maximum(cam, 0)
    cam_resized = cv2.resize(cam, (224, 224))
    heatmap = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-8)
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)  # type: ignore
    overlay = cv2.addWeighted(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), 0.6, heatmap_colored, 0.4, 0)  # type: ignore
    return overlay, pred_class

# Usage example for GradCAM (after training) - pick one test image path
# from PIL import Image
# overlay, pred = show_gradcam(model, 'path/to/test_image.jpeg', val_pipeline)
# plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)); plt.title(f"Pred: {class_names[pred]}"); plt.axis('off')

print("\n" + "="*70)
print("✅ TRAINING COMPLETE!")
print("="*70)
print(f"\n📁 Output files created:")
print(f"   - {MODEL_OUT} (trained model)")
print(f"   - {results_file} (detailed results)")
print(f"   - {history_json} (JSON data)")
print(f"   - {plot_file} (training curves)")
print(f"\n🔮 To test predictions:")
print(f"   python predict.py --image chest_xray/test/NORMAL/IM-0001-0001.jpeg")
print(f"\n📊 To visualize results:")
print(f"   python visualize_results.py")
print("="*70)

print("Done.")
