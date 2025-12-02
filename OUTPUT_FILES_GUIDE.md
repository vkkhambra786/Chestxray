# 📁 Output Files Guide

This document explains all the files that are automatically saved during and after training.

---

## 🎯 Files Created During Training

### 1. **Model Files** (Most Important!)

#### `mobilenet_cxr.pth`
- **What it is:** Your trained AI model weights
- **Size:** ~14 MB
- **When created:** During training (saved when validation improves)
- **Use for:** Making predictions on new X-ray images
- **Important:** Keep this safe! It contains all the learned knowledge

#### `mobilenet_cxr_test.pth` (from small training)
- **What it is:** Model from quick test with limited data
- **Size:** ~14 MB
- **When created:** After quick test training
- **Use for:** Testing code, not for production

---

### 2. **Results Files** (Human Readable)

#### `training_results_YYYYMMDD_HHMMSS.txt`
**Example:** `training_results_20251202_143052.txt`

**Contains:**
- Date and time of training
- Configuration (epochs, batch size, learning rate)
- Dataset sizes (train/val/test)
- Training history for each epoch
- Final test accuracy and metrics
- Classification report
- Confusion matrix
- Summary statistics

**Sample content:**
```
======================================================================
CHEST X-RAY CLASSIFICATION - TRAINING RESULTS
======================================================================

Date: 2025-12-02 14:30:52
Device: cpu
Model: MobileNetV2
Model saved as: mobilenet_cxr.pth

CONFIGURATION:
  Epochs: 8
  Batch Size: 16
  Learning Rate: 0.0001

TRAINING HISTORY:
Epoch 1/8:
  Train Loss: 0.4123, Train Acc: 0.8234
  Val Loss:   0.3891, Val Acc:   0.8500
...

TEST SET RESULTS:
Test Accuracy: 0.9122 (91.22%)

CLASSIFICATION REPORT:
              precision    recall  f1-score   support
      NORMAL       0.91      0.90      0.91       234
   PNEUMONIA       0.92      0.93      0.92       390
```

**Use this to:**
- ✅ Review training performance
- ✅ Share results with colleagues
- ✅ Compare different training runs
- ✅ Document your experiments

---

### 3. **JSON Data Files** (Machine Readable)

#### `training_history_YYYYMMDD_HHMMSS.json`
**Example:** `training_history_20251202_143052.json`

**Contains:**
- All configuration parameters
- Complete training history (loss/accuracy per epoch)
- Test results
- Metrics in structured format

**Sample content:**
```json
{
  "config": {
    "epochs": 8,
    "batch_size": 16,
    "learning_rate": 0.0001,
    "device": "cpu",
    "model": "MobileNetV2"
  },
  "history": {
    "train_loss": [0.4123, 0.3456, 0.2891, ...],
    "train_acc": [0.8234, 0.8567, 0.8891, ...],
    "val_loss": [0.3891, 0.3234, 0.2776, ...],
    "val_acc": [0.8500, 0.8750, 0.9000, ...]
  },
  "test_results": {
    "accuracy": 0.9122,
    "f1_score": 0.9140,
    "precision": 0.9150,
    "recall": 0.9165
  }
}
```

**Use this to:**
- ✅ Load data into Python/Excel for analysis
- ✅ Create custom visualizations
- ✅ Compare experiments programmatically
- ✅ Track hyperparameter tuning

---

### 4. **Plot Images**

#### `training_plot_YYYYMMDD_HHMMSS.png`
**Example:** `training_plot_20251202_143052.png`

**Contains:**
- Two side-by-side graphs:
  1. **Loss plot:** Training vs Validation loss over epochs
  2. **Accuracy plot:** Training vs Validation accuracy over epochs

**Visual example:**
```
[Loss Graph]              [Accuracy Graph]
Train ──── Val ────       Train ──── Val ────
  │                         │
  │  ╲                      │      ╱
  │   ╲                     │    ╱
  │    ╲___                 │  ╱
  │        ╲___             │╱______
  └──────────────           └──────────────
   Epochs →                  Epochs →
```

**Use this to:**
- ✅ Visualize learning progress
- ✅ Detect overfitting (val worse than train)
- ✅ Include in reports/presentations
- ✅ Quick visual assessment

---

## 📊 File Organization

After training, your directory will look like:

```
Chestxray/
├── mobilenet_cxr.pth                          ← Your trained model
├── training_results_20251202_143052.txt       ← Detailed text results
├── training_history_20251202_143052.json      ← JSON data
├── training_plot_20251202_143052.png          ← Visual plots
├── chest_xray/                                ← Dataset (not changed)
├── train_cxray.py                             ← Training script
└── ... (other project files)
```

---

## 🔍 Quick File Reference

| File Type | Extension | Keep? | Share? | Purpose |
|-----------|-----------|-------|--------|---------|
| Model | `.pth` | ✅ YES | ⚠️ Large | Make predictions |
| Results | `.txt` | ✅ YES | ✅ YES | Human review |
| History | `.json` | ✅ YES | ✅ YES | Data analysis |
| Plots | `.png` | ✅ YES | ✅ YES | Visualization |

---

## 💡 How to Use These Files

### Load Model for Predictions
```python
import torch
from torchvision import models

model = models.mobilenet_v2(pretrained=False)
model.load_state_dict(torch.load('mobilenet_cxr.pth'))
model.eval()
```

### Load JSON for Analysis
```python
import json

with open('training_history_20251202_143052.json', 'r') as f:
    data = json.load(f)

print(f"Final accuracy: {data['test_results']['accuracy']}")
print(f"Training losses: {data['history']['train_loss']}")
```

### Compare Multiple Runs
```python
import json
import matplotlib.pyplot as plt

# Load multiple JSON files
runs = ['run1.json', 'run2.json', 'run3.json']
for run_file in runs:
    with open(run_file, 'r') as f:
        data = json.load(f)
    plt.plot(data['history']['val_acc'], label=run_file)

plt.legend()
plt.show()
```

---

## 🗂️ File Naming Convention

All files include timestamp: `YYYYMMDD_HHMMSS`

**Format:** Year Month Day _ Hour Minute Second

**Example:** `20251202_143052` means:
- **2025** - Year
- **12** - December
- **02** - 2nd day
- **14** - 2 PM
- **30** - 30 minutes
- **52** - 52 seconds

This ensures:
- ✅ No overwriting old results
- ✅ Chronological sorting
- ✅ Easy identification
- ✅ Multiple experiments tracked

---

## 📦 Backing Up Important Files

### Essential Files to Keep:
1. `mobilenet_cxr.pth` - Your trained model (14 MB)
2. Latest `training_results_*.txt` - Best results
3. Latest `training_history_*.json` - Raw data

### Optional to Archive:
- `training_plot_*.png` - Can regenerate from JSON
- `mobilenet_cxr_test.pth` - Only for testing

### Safe to Delete:
- Old test/experiment runs after reviewing
- Duplicate plots

---

## 🎯 Example: After Full Training

You'll have these files:

```
mobilenet_cxr.pth                              (14 MB)  ← Use this for predictions!
training_results_20251202_143052.txt           (5 KB)   ← Read this for results
training_history_20251202_143052.json          (3 KB)   ← Use for analysis
training_plot_20251202_143052.png              (50 KB)  ← View progress

Summary:
✅ Test Accuracy: 91.22%
✅ F1-Score: 0.914
✅ Training time: 2.5 hours
✅ All metrics saved!
```

---

## 🆘 Troubleshooting

### "Cannot find results file"
→ Files are saved with timestamp, check the latest one:
```powershell
dir training_results_*.txt | sort LastWriteTime -Descending | select -First 1
```

### "Model file not found"
→ Make sure training completed successfully and saved the model

### "Want to see old results"
→ All files are timestamped, just open older dated files

---

## 📞 Summary

**Every training run automatically saves:**
- ✅ Trained model (`.pth`)
- ✅ Detailed text report (`.txt`)
- ✅ JSON data (`.json`)
- ✅ Training plots (`.png`)

**All timestamped, never overwritten, ready to share!**

---

**No more lost results! 🎉**
