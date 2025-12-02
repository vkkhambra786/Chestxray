"""
Visualization script for model results and predictions
Run after training to analyze model performance
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
try:
    import seaborn as sns  # type: ignore
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("⚠️  Warning: seaborn not installed. Using matplotlib only.")
    print("Install with: pip install seaborn")
from sklearn.metrics import confusion_matrix, roc_curve, auc
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# Configuration
MODEL_PATH = "mobilenet_cxr.pth"
DATA_DIR = "chest_xray"
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms (same as training validation)
from PIL import Image

class GrayToRGB(object):
    def __call__(self, img):
        arr = np.array(img)
        if len(arr.shape) == 2:
            arr = np.stack([arr]*3, axis=-1)
        return transforms.ToPILImage()(arr)

test_transform = transforms.Compose([
    GrayToRGB(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.485, 0.485], [0.229, 0.229, 0.229])
])

def load_model(model_path, num_classes=2):
    """Load the trained model"""
    from torchvision import models
    import torch.nn as nn
    
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

def get_predictions(model, loader):
    """Get all predictions and true labels"""
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot confusion matrix heatmap"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    
    if HAS_SEABORN:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={'label': 'Count'})  # type: ignore
    else:
        # Fallback to matplotlib only
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar(label='Count')
        
        # Add text annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color='white' if cm[i, j] > cm.max()/2 else 'black',
                        fontsize=20, fontweight='bold')
        
        # Set labels
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names)
        plt.yticks(tick_marks, class_names)
    
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix - Chest X-Ray Classification', fontsize=14, fontweight='bold')
    
    # Add accuracy text
    accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
    plt.text(1, -0.3, f'Overall Accuracy: {accuracy:.2%}', 
             ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: confusion_matrix.png")
    plt.show()

def plot_roc_curve(y_true, y_probs):
    """Plot ROC curve for pneumonia detection"""
    # For binary classification, use probability of class 1 (PNEUMONIA)
    fpr, tpr, thresholds = roc_curve(y_true, y_probs[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('ROC Curve - Pneumonia Detection', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: roc_curve.png")
    plt.show()

def plot_prediction_confidence(y_true, y_probs, class_names):
    """Plot prediction confidence distribution"""
    # Get max probability for each prediction
    max_probs = np.max(y_probs, axis=1)
    correct = (y_true == np.argmax(y_probs, axis=1))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Correct vs Incorrect predictions
    axes[0].hist(max_probs[correct], bins=30, alpha=0.7, label='Correct', color='green', edgecolor='black')
    axes[0].hist(max_probs[~correct], bins=30, alpha=0.7, label='Incorrect', color='red', edgecolor='black')
    axes[0].set_xlabel('Prediction Confidence', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Prediction Confidence Distribution', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3)
    
    # Plot 2: Confidence by class
    for i, class_name in enumerate(class_names):
        mask = (y_true == i)
        axes[1].hist(np.max(y_probs[mask], axis=1), bins=20, alpha=0.6, 
                    label=f'{class_name} (n={mask.sum()})', edgecolor='black')
    axes[1].set_xlabel('Prediction Confidence', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Confidence by True Class', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('prediction_confidence.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: prediction_confidence.png")
    plt.show()

def analyze_errors(y_true, y_pred, y_probs, class_names):
    """Analyze misclassified samples"""
    misclassified = y_true != y_pred
    n_errors = misclassified.sum()
    
    print("\n" + "="*60)
    print("ERROR ANALYSIS")
    print("="*60)
    print(f"Total misclassified samples: {n_errors} / {len(y_true)} ({n_errors/len(y_true)*100:.2f}%)")
    
    # False Positives and False Negatives
    false_positives = ((y_true == 0) & (y_pred == 1)).sum()  # Predicted PNEUMONIA, actually NORMAL
    false_negatives = ((y_true == 1) & (y_pred == 0)).sum()  # Predicted NORMAL, actually PNEUMONIA
    
    print(f"\nFalse Positives (NORMAL → PNEUMONIA): {false_positives}")
    print(f"False Negatives (PNEUMONIA → NORMAL): {false_negatives}")
    
    if n_errors > 0:
        # Confidence of errors
        error_confidences = np.max(y_probs[misclassified], axis=1)
        print(f"\nMisclassification confidence:")
        print(f"  Mean: {error_confidences.mean():.3f}")
        print(f"  Median: {np.median(error_confidences):.3f}")
        print(f"  Min: {error_confidences.min():.3f}")
        print(f"  Max: {error_confidences.max():.3f}")
        
        # High confidence errors (>0.8)
        high_conf_errors = (error_confidences > 0.8).sum()
        print(f"\nHigh-confidence errors (>80%): {high_conf_errors} / {n_errors}")
    
    print("="*60 + "\n")

def main():
    """Main execution"""
    print("🔍 Chest X-Ray Classification - Results Visualization")
    print("="*60)
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file '{MODEL_PATH}' not found!")
        print("Please train the model first using train_cxray.py")
        return
    
    # Load test dataset
    test_dir = os.path.join(DATA_DIR, "test")
    if not os.path.exists(test_dir):
        print(f"❌ Error: Test directory '{test_dir}' not found!")
        return
    
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)  # 0 for Windows
    class_names = test_dataset.classes
    
    print(f"✓ Loaded test dataset: {len(test_dataset)} samples")
    print(f"✓ Classes: {class_names}")
    
    # Load model
    print(f"\n✓ Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH, num_classes=len(class_names))
    
    # Get predictions
    print("✓ Generating predictions...")
    y_true, y_pred, y_probs = get_predictions(model, test_loader)
    
    # Calculate metrics
    accuracy = (y_true == y_pred).mean()
    print(f"\n✓ Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    plot_confusion_matrix(y_true, y_pred, class_names)
    plot_roc_curve(y_true, y_probs)
    plot_prediction_confidence(y_true, y_probs, class_names)
    
    # Error analysis
    analyze_errors(y_true, y_pred, y_probs, class_names)
    
    print("✅ All visualizations complete!")
    print("\nGenerated files:")
    print("  - confusion_matrix.png")
    print("  - roc_curve.png")
    print("  - prediction_confidence.png")

if __name__ == "__main__":
    main()
