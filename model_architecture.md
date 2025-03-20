# Vision Transformer (ViT) Model Architecture for Solar PV Anomaly Detection

This document outlines the architecture of the Vision Transformer (ViT) model used for solar photovoltaic (PV) anomaly detection in this project.

## Model Overview

The model is based on the Vision Transformer architecture, which was introduced in the paper ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"](https://arxiv.org/abs/2010.11929) by Dosovitskiy et al.

For this specific application, we use the `vit_base_patch16_224` model from the `timm` library, which is pre-trained on ImageNet and fine-tuned for our solar PV anomaly detection task.

## Model Architecture

### Base Architecture

- **Model Type**: `vit_base_patch16_224`
- **Patch Size**: 16x16 pixels
- **Input Image Size**: 224x224 pixels
- **Embedding Dimension**: 768
- **Number of Transformer Layers**: 12
- **Number of Attention Heads**: 12
- **MLP Size**: 3072
- **Dropout Rate**: 0.1

### Customized Output Layer

The pre-trained model is modified with a custom fully connected layer to output predictions for our specific classes (normal vs. different types of anomalies in solar PV modules).

```python
class ViTModel(nn.Module):
    def __init__(self, num_classes=10):
        super(ViTModel, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.vit(x)
```

## Data Processing

The input images are processed using the following steps:

1. Resizing to 224x224 pixels
2. Normalization with ImageNet mean and standard deviation
3. Data augmentation techniques (random horizontal flip, random rotation, etc.)

## Training Process

- **Loss Function**: Cross-Entropy Loss
- **Optimizer**: Adam with learning rate of 1e-4
- **Learning Rate Scheduler**: Cosine Annealing
- **Batch Size**: 32
- **Number of Epochs**: 30

## Feature Visualization

The model's feature representations are visualized using t-SNE (t-distributed Stochastic Neighbor Embedding) to understand how well the model separates different classes in the feature space.

## Model Performance

The model achieves the following performance metrics on the test set:

- **Accuracy**: TBD
- **Precision**: TBD
- **Recall**: TBD
- **F1 Score**: TBD

## References

1. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2020). An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929.
2. Ross Wightman. (2019). PyTorch Image Models. https://github.com/rwightman/pytorch-image-models