# Experiment Results

## Experiment 1: MobileNetV2 Transfer Learning (Baseline)

**Date**: December 2, 2025  
**Model**: MobileNetV2 (pretrained on ImageNet)  
**Status**: ✅ Completed

### Hyperparameters
- **Epochs**: 8
- **Batch Size**: 16
- **Learning Rate**: 1e-4
- **Optimizer**: Adam
- **Scheduler**: ReduceLROnPlateau (mode='max', factor=0.5, patience=2)
- **Loss Function**: CrossEntropyLoss

### Data Augmentation
- Grayscale to RGB conversion (channel duplication)
- Resize to 224x224
- Random Horizontal Flip
- Random Rotation (±10°)
- Color Jitter (brightness=0.1, contrast=0.1)
- ImageNet normalization

### Results

#### Test Set Performance
```
Test Loss: 0.2556
Test Accuracy: 91.22%
```

#### Classification Report
```
              precision    recall    f1-score
NORMAL          0.9103     0.9021     0.9062
PNEUMONIA       0.9175     0.9262     0.9218
```

#### Key Metrics
- **Overall F1-Score**: 0.914
- **Pneumonia Precision**: 91.75% (low false positives)
- **Pneumonia Recall**: 92.62% (good at catching pneumonia cases)
- **Balanced Performance**: Similar performance for both classes

### Analysis
✅ **Strengths**:
- High accuracy (>91%) with relatively few epochs
- Well-balanced precision/recall for both classes
- Low false positive rate for pneumonia detection
- High recall for pneumonia (important for medical screening)

⚠️ **Observations**:
- Slightly better performance on PNEUMONIA class (F1: 0.9218 vs 0.9062)
- This is actually desirable in medical imaging (better to catch disease)

### Model Save
- **Filename**: `mobilenet_cxr.pth`
- **Selection Criterion**: Best validation F1-score during training

---

## Future Experiments

### Experiment Ideas:
1. **Different Architectures**
   - ResNet18/ResNet50
   - EfficientNet-B0
   - DenseNet121
   - Compare inference speed vs accuracy

2. **Fine-tuning Strategies**
   - Unfreeze last N layers
   - Progressive unfreezing
   - Different learning rates for backbone vs classifier

3. **Advanced Augmentation**
   - AutoAugment
   - CutMix/MixUp
   - Elastic deformations
   - Random erasing

4. **Class Imbalance Handling**
   - Weighted loss function
   - Focal loss
   - SMOTE or oversampling

5. **Ensemble Methods**
   - Multiple models voting
   - Snapshot ensemble
   - Test-time augmentation

6. **Regularization**
   - Dropout variations
   - Label smoothing
   - Weight decay tuning

---

## Comparison Template

| Experiment | Model | Epochs | Batch Size | LR | Test Acc | F1-Score | Notes |
|------------|-------|--------|------------|----|---------:|----------|-------|
| 1 (Baseline) | MobileNetV2 | 8 | 16 | 1e-4 | 91.22% | 0.914 | Good baseline |
| 2 | ... | ... | ... | ... | ... | ... | ... |

---

## Notes
- All experiments use the same train/val/test split for fair comparison
- Random seed set to 42 for reproducibility
- Models trained on: CPU/GPU (specify your hardware)
