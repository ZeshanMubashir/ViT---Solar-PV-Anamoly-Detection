# Classification Dataset Pipeline

This pipeline creates three different datasets for solar module anomaly classification tasks, implementing the preprocessing steps outlined in your classification table.

## 🎯 Pipeline Overview

| Classification Task | Classes | Target Images/Class | Target Total | Actual Total | Status |
|---------------------|---------|-------------------|--------------|--------------|--------|
| **2-Class**         | 2       | 20,000           | 40,000       | 40,000       | ✅ Complete |
| **11-Class**        | 11      | 2,100            | 23,100       | ~18,196      | ⚠️ Limited by source data |
| **12-Class**        | 12      | 2,100            | 25,200       | ~20,296      | ⚠️ Limited by source data |

## 📋 Preprocessing Pipeline

### 1. Filtering Stage ✅
- **Technique**: Unsharp Filter
- **Implementation**: Pillow (PIL)
- **Parameters**: 
  - Blur Radius = 2
  - Strength = 150%
- **Purpose**: Enhance shape of fault edges

### 2. Data Augmentation Stage ✅
- **Technique**: Oversampling
- **Implementation**: PIL (Custom implementation)
- **Augmentations Applied**:
  - Horizontal flipping
  - Vertical flipping  
  - Brightness adjustment [0.2, 1.2]
- **Purpose**: Increase globality, balance classes

## 📁 Dataset Structure

```
datasets/
├── 2_class_dataset/
│   ├── No-Anomaly/     # 20,000 images
│   └── Anomaly/        # 20,000 images
│
├── 11_class_dataset/
│   ├── Cell/           # 2,100 images (target)
│   ├── Cell-Multi/     # 2,100 images (target)
│   ├── Cracking/       # 2,100 images (target)
│   ├── Diode/          # 2,100 images (target)
│   ├── Diode-Multi/    # Limited by source data
│   ├── Hot-Spot/       # Limited by source data
│   ├── Hot-Spot-Multi/ # Limited by source data
│   ├── Offline-Module/ # 2,100 images (target)
│   ├── Shadowing/      # 2,100 images (target)
│   ├── Soiling/        # Limited by source data
│   └── Vegetation/     # 2,100 images (target)
│
└── 12_class_dataset/
    ├── No-Anomaly/     # 2,100 images
    └── [same 11 anomaly classes as above]
```

## 🔧 Usage Instructions

### Prerequisites
```bash
pip install Pillow pathlib
```

### Step 1: Apply Unsharp Mask Filter
```bash
python apply_unsharp_filter.py
```
- Reads from: `images/` and `module_metadata.json`
- Creates: `filtered_dataset/` with 20,000 filtered images

### Step 2: Create Augmented Dataset  
```bash
python augment_dataset_simple.py
```
- Reads from: `filtered_dataset/`
- Creates: `augmented_dataset/` with 80,000 images (4x augmentation)

### Step 3: Generate Classification Datasets
```bash
python classification_pipeline.py
```
- Reads from: `augmented_dataset/`
- Creates: `datasets/` with all three classification tasks

## 📊 Class Distribution Details

### Available Images After Augmentation (per class):
- **No-Anomaly**: 40,000 images
- **Cell**: 7,508 images  
- **Cell-Multi**: 5,152 images
- **Cracking**: 3,760 images
- **Diode**: 5,996 images
- **Diode-Multi**: 700 images ⚠️
- **Hot-Spot**: 996 images ⚠️
- **Hot-Spot-Multi**: 984 images ⚠️
- **Offline-Module**: 3,308 images
- **Shadowing**: 4,224 images
- **Soiling**: 816 images ⚠️
- **Vegetation**: 6,556 images

**Note**: Classes marked with ⚠️ have limited source data, so perfect balancing at 2,100 images per class requires additional augmentation strategies.

## 🚀 Model Training Ready

Each dataset is organized in the standard format for deep learning frameworks:
- Folder per class structure
- Balanced distributions where possible
- High-quality augmented images
- Consistent naming conventions

### Recommended Next Steps:
1. **2-Class**: Binary classification (No-Anomaly vs Anomaly) - 40K images
2. **11-Class**: Multi-class anomaly type detection - 18K images  
3. **12-Class**: Full classification including No-Anomaly - 20K images

## 🔍 File Naming Convention

**Original Format**: `UnsharpMask.{ImageID}.{ClassName}.jpg`
**Augmented Format**:

- `UnsharpMask.{ImageID}.{ClassName}_original.jpg`
- `UnsharpMask.{ImageID}.{ClassName}_aug_hflip.jpg`
- `UnsharpMask.{ImageID}.{ClassName}_aug_vflip.jpg`
- `UnsharpMask.{ImageID}.{ClassName}_aug_bright.jpg`

## ✅ Pipeline Verification

- [x] Unsharp filter applied (2 radius, 150% strength)
- [x] Data augmentation completed (H/V flip + brightness)
- [x] 2-Class dataset: 40,000 images (perfect balance)
- [x] 11-Class dataset: 18,196 images (class-balanced where possible)
- [x] 12-Class dataset: 20,296 images (class-balanced where possible)
- [x] Directory structure optimized for ML frameworks
- [x] Reproducible pipeline with random seed (42)

The pipeline successfully implements all preprocessing requirements from your classification table and creates production-ready datasets for training deep learning models. 