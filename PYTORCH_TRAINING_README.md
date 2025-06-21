# ViT-B/32 Training with PyTorch - Solar Module Classification

This repository contains three Jupyter notebooks for training Vision Transformer (ViT-B/32) models using PyTorch, TIMM, and Hugging Face libraries for solar module infrared image classification.

## 📋 Overview

### Models and Tasks

1. **2-Class Classification**: `ViT_2Class_Classification_PyTorch.ipynb`
   - Binary classification: No-Anomaly vs Anomaly
   - 40,000 images (20,000 per class)

2. **11-Class Classification**: `ViT_11Class_Classification_PyTorch.ipynb`
   - Anomaly type classification only
   - 23,100 images (2,100 per class)

3. **12-Class Classification**: `ViT_12Class_Classification_PyTorch.ipynb`
   - Complete classification including No-Anomaly
   - 25,200 images (2,100 per class)

## 🏗️ Architecture

### Model Specifications
- **Base Model**: ViT-B/32 (Vision Transformer Base with 32×32 patches)
- **Framework**: PyTorch with TIMM library
- **Input Size**: 160 × 160 × 3
- **Patch Size**: 32 × 32
- **Pre-training**: ImageNet-21k → ImageNet-1k

### Key Components
- **Backbone**: Pre-trained ViT-B/32 from TIMM
- **Classification Head**: Custom dense layers with ReLU activation
- **Dropout**: 0.3 for regularization

## 📊 Hyperparameters (From Paper)

| Parameter | Value |
|-----------|-------|
| Input Shape | 160 × 160 × 3 |
| Batch Size | 8 |
| Max Epochs | 60 |
| Learning Rate | 1e-4 |
| Optimizer | AdamW (alternative to Rectified Adam) |
| Weight Decay | 0.01 |
| Learning Rate Decay | 0.2 |
| LR Patience | 5 |
| Early Stopping Patience | 9 |
| Loss Function | CrossEntropy with Label Smoothing (0.2) |
| Validation Split | 10% |
| Test Split | 10% |
| Epsilon | 1e-6 |

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_pytorch.txt
```

### 2. Verify Dataset Structure
Ensure your datasets are organized as:
```
datasets_paper/
├── 2_class_dataset/
│   ├── Anomaly/
│   └── No-Anomaly/
├── 11_class_dataset/
│   ├── Cell/
│   ├── Cell-Multi/
│   ├── Cracking/
│   ├── Diode/
│   ├── Diode-Multi/
│   ├── Hot-Spot/
│   ├── Hot-Spot-Multi/
│   ├── Offline-Module/
│   ├── Shadowing/
│   ├── Soiling/
│   └── Vegetation/
└── 12_class_dataset/
    ├── No-Anomaly/
    ├── Cell/
    ├── Cell-Multi/
    ├── Cracking/
    ├── Diode/
    ├── Diode-Multi/
    ├── Hot-Spot/
    ├── Hot-Spot-Multi/
    ├── Offline-Module/
    ├── Shadowing/
    ├── Soiling/
    └── Vegetation/
```

### 3. Run Training
Start Jupyter and open the desired notebook:
```bash
jupyter notebook ViT_2Class_Classification_PyTorch.ipynb
```

## 🔧 Implementation Details

### Data Augmentation
- Random horizontal flip (p=0.5)
- Random vertical flip (p=0.3)
- Random rotation (±5°)
- Color jitter (brightness, contrast)
- ImageNet normalization

### Model Architecture
```python
class ViTClassifier(nn.Module):
    def __init__(self, num_classes, model_name='vit_base_patch32_224'):
        # Pre-trained ViT-B/32 backbone
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
```

### Training Features
- **Early Stopping**: Monitors validation loss with patience=9
- **Learning Rate Scheduling**: ReduceLROnPlateau with factor=0.2, patience=5
- **Model Checkpointing**: Saves best model based on validation accuracy
- **Progress Tracking**: TQDM progress bars with real-time metrics
- **Comprehensive Logging**: Training history, metrics, and summaries

### Loss Function
Custom Label Smoothing CrossEntropy:
```python
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.2):
        # Implements label smoothing for better generalization
```

## 📈 Output and Results

### Saved Files
Each training run generates:
```
models/{task_name}_pytorch/
├── best_model.pth              # Best model checkpoint
└── model_architecture.json    # Model configuration

results/{task_name}_pytorch/
├── training_history.png       # Training curves
├── training_history.json      # Raw training metrics
├── confusion_matrix.png       # Test set confusion matrix
├── classification_report.json # Detailed performance metrics
└── training_summary.json      # Comprehensive summary
```

### Visualization
- Training/validation accuracy and loss curves
- Learning rate schedule
- Overfitting monitoring (train-val gap)
- Confusion matrix with class labels
- Per-class precision, recall, F1-score

## 🎯 Expected Performance

Based on the paper's methodology, expected results:
- **2-Class**: High accuracy for anomaly detection
- **11-Class**: Detailed anomaly type classification
- **12-Class**: Complete solar module condition assessment

## 🔬 Technical Specifications

### Hardware Requirements
- **GPU**: CUDA-compatible GPU recommended (8GB+ VRAM)
- **RAM**: 16GB+ recommended
- **Storage**: ~2GB for models and results

### Software Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)

## 📚 Key Libraries
- **PyTorch**: Deep learning framework
- **TIMM**: Pre-trained vision models
- **Transformers**: Hugging Face transformers
- **Torchvision**: Computer vision utilities
- **Scikit-learn**: Evaluation metrics

## 🔄 Customization

### Hyperparameter Tuning
Modify the `HYPERPARAMS` dictionary in each notebook:
```python
HYPERPARAMS = {
    'batch_size': 16,        # Increase if you have more GPU memory
    'learning_rate': 5e-5,   # Lower for fine-tuning
    'epochs': 100,           # Increase if needed
    # ... other parameters
}
```

### Model Architecture
Change the backbone model:
```python
# Use different ViT variants
model_name = 'vit_large_patch16_224'  # ViT-L/16
model_name = 'vit_small_patch16_224'  # ViT-S/16
```

## 🐛 Troubleshooting

### Common Issues
1. **CUDA Out of Memory**: Reduce batch size or image resolution
2. **Slow Training**: Ensure GPU is being used (`device = cuda`)
3. **Dataset Not Found**: Check dataset paths in `DATASET_PATH`

### Performance Tips
- Use mixed precision training for faster training
- Increase `num_workers` in DataLoader if CPU allows
- Use larger batch sizes if GPU memory permits

## 📖 References
- Vision Transformer Paper: "An Image is Worth 16x16 Words"
- TIMM Library: PyTorch Image Models
- Solar Module Dataset: Infrared thermal images

## 🤝 Contributing
Feel free to submit issues and enhancement requests!

---
**Note**: This implementation follows the exact hyperparameters and methodology specified in your research paper, adapted for PyTorch with modern best practices. 