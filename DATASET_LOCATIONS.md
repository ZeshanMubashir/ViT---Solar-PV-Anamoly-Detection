# 📁 Dataset Distribution Locations

## 🗂️ Complete Directory Structure

The processed datasets are saved in the following locations within your workspace:

C:\Users\zeshanm\Desktop\Transformers\Original Dataset\

## 📊 Main Dataset Directories

### 1. **Paper-Compliant Datasets** (RECOMMENDED)

📁 datasets_paper/
├── 📁 2_class_dataset/          # 40,000 images total
│   ├── 📁 No-Anomaly/          # 20,000 images
│   └── 📁 Anomaly/             # 20,000 images
│
├── 📁 11_class_dataset/         # 23,100 images total  
│   ├── 📁 Cell/                # 2,100 images
│   ├── 📁 Cell-Multi/          # 2,100 images
│   ├── 📁 Cracking/            # 2,100 images
│   ├── 📁 Diode/               # 2,100 images
│   ├── 📁 Diode-Multi/         # 2,100 images
│   ├── 📁 Hot-Spot/            # 2,100 images
│   ├── 📁 Hot-Spot-Multi/      # 2,100 images
│   ├── 📁 Offline-Module/      # 2,100 images
│   ├── 📁 Shadowing/           # 2,100 images
│   ├── 📁 Soiling/             # 2,100 images
│   └── 📁 Vegetation/          # 2,100 images
│
└── 📁 12_class_dataset/         # 25,200 images total
    ├── 📁 No-Anomaly/          # 2,100 images
    ├── 📁 Cell/                # 2,100 images
    ├── 📁 Cell-Multi/          # 2,100 images
    ├── 📁 Cracking/            # 2,100 images
    ├── 📁 Diode/               # 2,100 images
    ├── 📁 Diode-Multi/         # 2,100 images
    ├── 📁 Hot-Spot/            # 2,100 images
    ├── 📁 Hot-Spot-Multi/      # 2,100 images
    ├── 📁 Offline-Module/      # 2,100 images
    ├── 📁 Shadowing/           # 2,100 images
    ├── 📁 Soiling/             # 2,100 images
    └── 📁 Vegetation/          # 2,100 images

### 2. **Basic Pipeline Datasets** (For reference)

📁 datasets/
├── 📁 2_class_dataset/          # 40,000 images
├── 📁 11_class_dataset/         # 18,196 images (limited by source data)
└── 📁 12_class_dataset/         # 20,296 images (limited by source data)

### 3. **Intermediate Processing Data**

📁 filtered_dataset/             # 20,000 UnsharpMask filtered images (~17.3 MB)
📁 augmented_dataset/            # 80,000 total images (~69 MB)
📁 images/                       # 20,000 original raw images

## 🎯 **Recommended for Use: `datasets_paper/`**

The **`datasets_paper/`** directory contains the **publication-ready** datasets that:

✅ **Match exact paper specifications**  
✅ **Perfect class balance** (2,100 per class)  
✅ **Advanced augmentation techniques**  
✅ **Academic research standards**  

## 📝 File Naming Conventions

### Paper-Compliant Datasets

ClassView_{0000-9999}.jpg                    # Direct copies when sufficient data
ClassView_{0000-9999}_aug_{augmentation}.jpg # Generated through advanced augmentation

### Examples

No-Anomaly_0000.jpg
No-Anomaly_0001.jpg
...
Cell_0000.jpg
Cell_0001_aug_hflip.jpg
Cell_0002_aug_bright.jpg

## 💾 Storage Information

| Dataset | Location | Size | Images | Status |
|---------|----------|------|---------|--------|
| **2-Class (Paper)** | `datasets_paper/2_class_dataset/` | ~21.94 MB | 40,000 | ✅ Ready |
| **11-Class (Paper)** | `datasets_paper/11_class_dataset/` | ~34.25 MB | 23,100 | ✅ Ready |
| **12-Class (Paper)** | `datasets_paper/12_class_dataset/` | ~37.13 MB | 25,200 | ✅ Ready |
| Filtered Dataset | `filtered_dataset/` | ~17.3 MB | 20,000 | ✅ Complete |
| Augmented Dataset | `augmented_dataset/` | ~69 MB | 80,000 | ✅ Complete |

## 🚀 Quick Access Commands

### Navigate to datasets

cd "datasets_paper"

### Count images in each class

Get-ChildItem -Path "datasets_paper" -Recurse -Directory | ForEach-Object {
    $count = (Get-ChildItem -Path $_.FullName -File "*.jpg").Count
    Write-Output "$($_.Name): $count images"
}

### Copy datasets to another location

# Copy paper-compliant datasets

xcopy "datasets_paper" "C:\destination\path\" /E /I /H

## 🔬 **For Academic Research**

Use the **`datasets_paper/`** directory as it provides:

- Exact numbers specified in research papers
- Balanced class distributions  
- Publication-ready format
- Reproducible methodology
- Standard naming conventions

This ensures your research results are comparable with published academic work and meets reviewer expectations for balanced experimental datasets.
