# Siamese Network for MNIST Similarity Detection

This repository provides an implementation of a Siamese Network for identifying the similarity between pairs of MNIST images. The network uses a custom loss function and Euclidean distance for training and evaluation.

## Overview
The project includes the following key components:

1. **Siamese Network Architecture**: A convolutional neural network (CNN) that processes pairs of images to extract feature embeddings.
2. **Euclidean Distance**: A metric to calculate similarity between the feature embeddings of image pairs.
3. **Meta-Learning Training**: A training procedure using a custom loss function (`MarginRankingLoss`) to learn from image pairs.
4. **Visualization**: Tools to plot loss curves and compare similarity between image pairs.

---

## Requirements
The project is implemented in Python using PyTorch. The following libraries are required:

- `torch`
- `torchvision`
- `matplotlib`

## Code Breakdown
### 1. Siamese Network Definition
The `SiameseNetwork` class defines a convolutional neural network with shared weights to process pairs of images and extract feature embeddings.

### 2. Euclidean Distance Function
A function that computes the Euclidean distance between two feature vectors:
```python
def euclidean_distance(x1, x2):
    return torch.sqrt(torch.sum((x1 - x2) ** 2, dim=1) + 1e-8)
```

### 3. Data Preparation
MNIST dataset is used, with pairs of images and their similarity labels (1 for similar, -1 for different).

### 4. Meta-Learning Training
The `meta_learning_training` function trains the network by optimizing the margin-based ranking loss. The loss encourages the network to bring similar pairs closer while separating dissimilar pairs.

---

## Notes
- The dataset is automatically downloaded if not present in the `./data` directory.
- You can adjust hyperparameters such as learning rate, batch size, and number of epochs in the script.


