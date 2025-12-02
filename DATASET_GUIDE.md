# Chest X-Ray Dataset Guide

## 📥 Where to Download the Dataset

### Option 1: Kaggle (Recommended - FREE)

1. **Go to Kaggle Dataset Page:**
   - Visit: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
   - OR search "chest x-ray pneumonia kaggle" on Google

2. **Download the Dataset:**
   - Click the "Download" button (need free Kaggle account)
   - File size: ~1.15 GB (zip file)
   - Contains: 5,863 X-Ray images (JPEG)

3. **Dataset Information:**
   - **Training set**: ~5,216 images
   - **Validation set**: ~16 images  
   - **Test set**: ~624 images
   - **Classes**: NORMAL (1,583 images) and PNEUMONIA (4,273 images)

### Option 2: Using Kaggle API (Automated)

If you have Kaggle API installed:

```powershell
# Install kaggle
pip install kaggle

# Download dataset (after setting up Kaggle API credentials)
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia

# Unzip
Expand-Archive -Path chest-xray-pneumonia.zip -DestinationPath .
```

### Option 3: Direct Download Links

Alternative sources:
- Mendeley Data: https://data.mendeley.com/datasets/rscbjbr9sj/2
- Original Paper: https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5

---

## 📁 Extract and Organize

After downloading, extract the ZIP file. You should see this structure:

```
chest_xray/
├── train/
│   ├── NORMAL/           (1,341 images)
│   │   ├── IM-0001-0001.jpeg
│   │   ├── IM-0001-0002.jpeg
│   │   └── ...
│   └── PNEUMONIA/        (3,875 images)
│       ├── person1_bacteria_1.jpeg
│       ├── person1_virus_2.jpeg
│       └── ...
├── val/
│   ├── NORMAL/           (8 images)
│   └── PNEUMONIA/        (8 images)
└── test/
    ├── NORMAL/           (234 images)
    └── PNEUMONIA/        (390 images)
```

---

## ✅ Verify Dataset Setup

Run this command to check if your dataset is correctly organized:

```powershell
cd "d:\old computer\MyProject\Chestxray"
.\venv\Scripts\python.exe -c "import os; dirs=['chest_xray/train/NORMAL','chest_xray/train/PNEUMONIA','chest_xray/val/NORMAL','chest_xray/val/PNEUMONIA','chest_xray/test/NORMAL','chest_xray/test/PNEUMONIA']; [print(f'✓ {d}: {len(os.listdir(d)) if os.path.exists(d) else 0} images') if os.path.exists(d) else print(f'✗ {d}: NOT FOUND') for d in dirs]"
```

Expected output:
```
✓ chest_xray/train/NORMAL: 1341 images
✓ chest_xray/train/PNEUMONIA: 3875 images
✓ chest_xray/val/NORMAL: 8 images
✓ chest_xray/val/PNEUMONIA: 8 images
✓ chest_xray/test/NORMAL: 234 images
✓ chest_xray/test/PNEUMONIA: 390 images
```

---

## 🔍 Sample Images to Understand the Data

### Normal Chest X-Ray Characteristics:
- Clear lung fields
- Well-defined heart borders
- No opacities or infiltrates
- Normal vascular markings

### Pneumonia Chest X-Ray Characteristics:
- Cloudy/hazy areas (infiltrates)
- White patches (consolidation)
- Air bronchograms visible
- Blurred heart/diaphragm borders

---

## 📊 Dataset Statistics

| Split | NORMAL | PNEUMONIA | Total |
|-------|-------:|----------:|------:|
| Train | 1,341  | 3,875     | 5,216 |
| Val   | 8      | 8         | 16    |
| Test  | 234    | 390       | 624   |
| **Total** | **1,583** | **4,273** | **5,856** |

**Note:** Dataset is imbalanced (~73% pneumonia, ~27% normal)

---

## 🎓 Dataset Citation

If you use this dataset for research, please cite:

```
Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018), 
"Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification", 
Mendeley Data, V2, doi: 10.17632/rscbjbr9sj.2
```

Original Paper:
```
Kermany DS, Goldbaum M, Cai W, et al. Identifying Medical Diagnoses and Treatable 
Diseases by Image-Based Deep Learning. Cell. 2018;172(5):1122-1131.e9.
```

---

## ⚠️ Important Notes

1. **Medical Use Disclaimer**: This dataset is for educational/research purposes only
2. **Patient Privacy**: All images are de-identified
3. **Age Group**: Pediatric patients (ages 1-5 from Guangzhou, China)
4. **Quality Control**: All images were screened and graded by two expert physicians

---

## 🆘 Troubleshooting

### Problem: Download is too slow
**Solution:** Use a download manager or try at different times

### Problem: Can't find dataset on Kaggle
**Solution:** Make sure you're logged in. Search: "chest xray pneumonia"

### Problem: Dataset structure is different
**Solution:** Manually organize images into the required folder structure

### Problem: Not enough disk space (need ~2 GB)
**Solution:** Free up space or use external drive

---

## 📞 Need Help?

- Kaggle Dataset Page: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- Original Source: https://data.mendeley.com/datasets/rscbjbr9sj/2
- Paper: https://doi.org/10.1016/j.cell.2018.02.010
