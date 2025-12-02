# Usage Guide - Chest X-Ray Classification

## Quick Start Guide

### 1️⃣ Setup Environment

```powershell
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Prepare Dataset

Download the [Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and organize it as:

```
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

### 3️⃣ Train the Model

```powershell
python train_cxray.py
```

**Training Output:**
- Progress bars for each epoch
- Training and validation metrics (loss, accuracy, F1-score)
- Best model saved as `mobilenet_cxr.pth`
- Training plots displayed at the end

**Expected Results:**
- Training time: ~2-5 minutes per epoch (GPU) or ~20-30 minutes (CPU)
- Final test accuracy: ~91%

---

## 📊 Visualize Results

After training, analyze model performance:

```powershell
python visualize_results.py
```

**Generates:**
- `confusion_matrix.png` - Detailed breakdown of predictions
- `roc_curve.png` - ROC curve with AUC score
- `prediction_confidence.png` - Confidence distribution analysis
- Error analysis in console

---

## 🔮 Make Predictions

### Single Image Prediction

```powershell
# Predict on a single X-ray image
python predict.py --image path/to/xray.jpg
```

**Output:**
```
PREDICTION RESULTS
============================================================
Prediction: PNEUMONIA
Confidence: 94.52%

Class Probabilities:
  NORMAL      : 5.48%
  PNEUMONIA   : 94.52%
============================================================
```

Also displays visualization with X-ray image and probability bars.

### Batch Prediction

```powershell
# Predict on all images in a directory
python predict.py --dir path/to/images/
```

**Output:**
- Console output for each image
- `predictions.csv` with all results
- Summary statistics

### Options

```powershell
# Use different model file
python predict.py --image xray.jpg --model custom_model.pth

# Skip visualization (faster)
python predict.py --image xray.jpg --no-viz
```

---

## 🔧 Customization

### Modify Training Parameters

Edit `train_cxray.py`:

```python
# Change these values at the top of the file:
DATA_DIR = "chest_xray"          # Your dataset location
BATCH_SIZE = 16                  # Increase if you have more GPU memory
NUM_EPOCHS = 8                   # More epochs may improve accuracy
LR = 1e-4                        # Learning rate
MODEL_OUT = "mobilenet_cxr.pth"  # Output model name
```

### Try Different Models

Replace MobileNetV2 with other architectures:

```python
# In train_cxray.py, replace this line:
model = models.mobilenet_v2(pretrained=True)

# With alternatives:
model = models.resnet18(pretrained=True)
model = models.resnet50(pretrained=True)
model = models.efficientnet_b0(pretrained=True)
model = models.densenet121(pretrained=True)

# Then adjust the classifier accordingly
```

### Adjust Data Augmentation

Modify the `train_pipeline` in `train_cxray.py`:

```python
train_pipeline = transforms.Compose([
    GrayToRGB(),
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),        # Keep
    transforms.RandomRotation(10),            # Increase angle: 15, 20
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Increase values
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Add shift
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.485,0.485],[0.229,0.229,0.229])
])
```

---

## 📈 Performance Benchmarks

### Current Results (MobileNetV2)

| Metric | Value |
|--------|------:|
| Test Accuracy | 91.22% |
| Test Loss | 0.2556 |
| NORMAL F1-Score | 0.9062 |
| PNEUMONIA F1-Score | 0.9218 |
| Overall F1-Score | 0.914 |

### Training Time (8 epochs)
- **GPU (CUDA)**: ~20-40 minutes
- **CPU**: ~2-4 hours

### Inference Speed
- **Single image**: ~0.01-0.05 seconds (GPU) / ~0.1-0.3 seconds (CPU)
- **Batch (100 images)**: ~1-3 seconds (GPU) / ~20-30 seconds (CPU)

---

## 🐛 Troubleshooting

### Issue: CUDA Out of Memory

**Solution:** Reduce batch size in `train_cxray.py`:
```python
BATCH_SIZE = 8  # or 4
```

### Issue: Module not found

**Solution:** Make sure virtual environment is activated:
```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Issue: Dataset not found

**Solution:** Verify directory structure matches exactly:
```powershell
ls chest_xray
# Should show: train/, val/, test/
```

### Issue: Low accuracy (<80%)

**Possible causes:**
- Dataset not properly organized
- Insufficient training epochs
- Learning rate too high/low
- Need more data augmentation

**Solutions:**
- Verify dataset folder structure
- Train for more epochs (12-15)
- Try learning rate: 1e-3 or 5e-5
- Add more augmentation techniques

---

## 📚 Advanced Usage

### Export Model for Deployment

```python
# Convert to TorchScript
import torch
model.eval()
example = torch.rand(1, 3, 224, 224).to(DEVICE)
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("mobilenet_cxr_traced.pt")
```

### Use with ONNX

```python
# Export to ONNX format
torch.onnx.export(model, example, "mobilenet_cxr.onnx",
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 
                               'output': {0: 'batch_size'}})
```

### Grad-CAM Visualization

Use the built-in `show_gradcam()` function in `train_cxray.py`:

```python
from PIL import Image
import matplotlib.pyplot as plt

# After training
overlay, pred = show_gradcam(model, 'path/to/test_image.jpeg', val_pipeline)
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.title(f"Pred: {class_names[pred]}")
plt.axis('off')
plt.show()
```

---

## 📝 Tips for Best Results

### Data Quality
✅ Use high-resolution X-ray images  
✅ Ensure consistent image format  
✅ Balance dataset if possible  
✅ Clean corrupted/mislabeled images  

### Training
✅ Monitor validation metrics to avoid overfitting  
✅ Save multiple checkpoints  
✅ Use learning rate scheduling  
✅ Try different random seeds  

### Evaluation
✅ Test on completely unseen data  
✅ Analyze misclassified samples  
✅ Check confidence scores  
✅ Use multiple evaluation metrics  

---

## 🎯 Project Structure

```
Chestxray/
├── train_cxray.py           # Main training script
├── predict.py               # Inference script
├── visualize_results.py     # Results visualization
├── requirements.txt         # Dependencies
├── README.md               # Project documentation
├── RESULTS.md              # Experiment tracking
├── USAGE.md                # This file
├── mobilenet_cxr.pth       # Trained model (after training)
├── chest_xray/             # Dataset directory
│   ├── train/
│   ├── val/
│   └── test/
└── venv/                   # Virtual environment
```

---

## 📞 Support

For issues or questions:
1. Check this usage guide
2. Review README.md and RESULTS.md
3. Verify your dataset structure
4. Check Python and package versions
5. Open an issue on GitHub

---

**Happy Training! 🚀**
