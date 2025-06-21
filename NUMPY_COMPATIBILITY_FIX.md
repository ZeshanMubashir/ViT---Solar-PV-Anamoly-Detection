# NumPy 2.0 Compatibility Fix Guide

## Problem
You're encountering this error because your environment has NumPy 2.0.2, but TensorFlow and some other packages were compiled with NumPy 1.x and aren't compatible with NumPy 2.0.

## Quick Fix Options

### Option 1: Downgrade NumPy (Recommended)
```bash
# Uninstall problematic packages
pip uninstall numpy tensorflow transformers -y

# Install fixed requirements
pip install -r requirements_pytorch_fixed.txt

# Verify the fix
python -c "import torch, timm, numpy as np; print(f'PyTorch: {torch.__version__}, TIMM: {timm.__version__}, NumPy: {np.__version__}')"
```

### Option 2: Create a New Environment (Cleanest)
```bash
# Create new conda environment
conda create -n pytorch_vit python=3.10 -y
conda activate pytorch_vit

# Install requirements
pip install -r requirements_pytorch_fixed.txt

# Install PyTorch with CUDA support (if you have GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Option 3: Use Only Essential Imports
If you just want to run the notebook quickly, modify the imports in cell 2:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm
# Remove: from transformers import ViTForImageClassification, ViTConfig
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
```

## Why This Happens

1. **NumPy 2.0 Breaking Changes**: NumPy 2.0 introduced breaking changes in the C API
2. **TensorFlow Dependency**: The `transformers` library imports TensorFlow, which wasn't compiled for NumPy 2.0
3. **Indirect Import**: Even though we only want ViT models, `transformers` brings in TensorFlow dependencies

## Solution Used in Updated Notebook

I've modified the notebook to:
- ✅ Remove `transformers` import (use TIMM exclusively)
- ✅ Pin NumPy to version < 2.0.0
- ✅ Use only TIMM for ViT-B/32 models (which works perfectly)
- ✅ Add proper version constraints to prevent future conflicts

## Verification Commands

After fixing, run these to verify everything works:

```python
# Test basic imports
import torch
import timm
import numpy as np
print("✅ Basic imports successful")

# Test ViT model creation
model = timm.create_model('vit_base_patch32_224', pretrained=True, num_classes=2)
print("✅ ViT model creation successful")

# Test device detection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Device: {device}")
```

## Alternative: Use Google Colab

If you continue having environment issues, consider running the notebook in Google Colab:

1. Upload notebook to Google Drive
2. Open with Google Colab
3. Install requirements: `!pip install timm torch torchvision`
4. Use free GPU: Runtime → Change runtime type → GPU

## Updated Requirements File

Use `requirements_pytorch_fixed.txt` which includes:
- NumPy pinned to < 2.0.0
- No transformers dependency (TIMM only)
- Proper version constraints
- Clear installation instructions

The notebook will work perfectly with just TIMM - you don't need the full `transformers` library for ViT models! 