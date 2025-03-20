# Vision Transformer for Solar PV Anomaly Detection

This project implements a Vision Transformer (ViT) model for detecting anomalies in infrared images of solar photovoltaic (PV) modules.

## Project Overview

Solar photovoltaic systems require regular inspection and maintenance to ensure optimal performance. Infrared (thermal) imaging is a common technique used to identify potential issues in solar panels such as hotspots, cell cracks, or delamination. This project uses a Vision Transformer-based deep learning approach to automatically detect and classify these anomalies from infrared images.

## Dataset

The project uses the [Infrared Solar Modules dataset](https://www.kaggle.com/datasets/pythonafroz/solar-panels-ir-images) which contains:
- Thermal/infrared images of solar PV modules
- Multiple classes of anomalies and normal conditions
- Metadata for each image including module type and anomaly class

## Features

- Data preprocessing and augmentation for infrared images
- Implementation of a Vision Transformer model for anomaly detection
- t-SNE visualization of the learned feature space
- Image filtering techniques for enhanced visualization
- Dataset splitting based on anomaly classes
- Performance evaluation metrics

## Model Architecture

The model uses a pre-trained Vision Transformer (`vit_base_patch16_224`) from the `timm` library, fine-tuned for solar PV anomaly detection. For detailed model architecture information, see [model_architecture.md](model_architecture.md).

## Requirements

To install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

The main code is contained in the Jupyter notebook `ViT-Solar PV.ipynb`. To run the notebook:

1. Install the requirements
2. Download the dataset
3. Update the dataset paths in the notebook
4. Run the notebook cells

## Results

The Vision Transformer model demonstrates effective anomaly detection capabilities on the infrared solar module images, with detailed performance metrics provided in the notebook.

## References

1. Dosovitskiy, A., et al. (2020). An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929.
2. Research paper: "An innovative transformer neural network for fault detection and classification for photovoltaic modules"