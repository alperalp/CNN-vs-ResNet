## CNN vs. ResNet on CIFAR-10 Dataset

This project compares the performance of a standard Convolutional Neural Network (CNN) with that of a Residual Network (ResNet) on the CIFAR-10 dataset using PyTorch.

### Introduction

This project aims to assess and compare the classification accuracy and training efficiency of two popular convolutional neural network architectures, a standard CNN and ResNet, on the CIFAR-10 dataset. CIFAR-10 consists of 60,000 32x32 color images across 10 classes, with 6,000 images per class.

### Model Implementation

- **Standard CNN**: 
  - Implements a basic convolutional neural network architecture with convolutional layers, max-pooling layers, and fully connected layers.
- **ResNet**: 
  - Utilizes a deep residual network architecture with residual blocks and skip connections to address the challenges of training very deep networks.

### Training

Both models are trained on the CIFAR-10 training dataset using stochastic gradient descent (SGD) with momentum. The objective is to minimize the cross-entropy loss between the predicted and ground truth labels during training.

### Evaluation

Following training, the models are evaluated on the CIFAR-10 test dataset to assess their classification accuracy. Performance metrics such as accuracy, loss, and training time are recorded and compared between the standard CNN and ResNet architectures.

### Results

ResNet consistently outperforms the standard CNN in terms of classification accuracy on the CIFAR-10 test dataset. This superior performance can be attributed to ResNet's architecture, which effectively mitigates the vanishing gradient problem and enables the learning of more complex features.


![Screenshot from 2024-04-22 16-34-28](https://github.com/alperalp/CNN-vs-ResNet/assets/58988396/eb36e373-0783-447f-b999-025c647f53c9)
