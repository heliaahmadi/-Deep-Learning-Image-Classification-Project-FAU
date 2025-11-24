# Solar Panel Defect Classification using PyTorch  
This project focuses on building a deep learning model for **multi-label image classification** on a dataset of solar panel images. The goal is to automatically detect different types of defects—such as cracks and inactive regions—using a convolutional neural network implemented in **PyTorch**.

The dataset consists of real-world solar module images, where each panel can contain **zero, one, or multiple defects**, making this a multi-label classification task.  

---

##  Project Overview
The project explores:
- Designing and training a custom **ResNet-based architecture**
- Applying proper image **normalization**, preprocessing, and data loading
- Using **GPU-accelerated training** via PyTorch
- Evaluating performance with metrics suited for multi-label problems (e.g., F1 score)

The primary objective is to build a reliable defect-detection model and experiment with architectural choices, regularization strategies, and data augmentation to improve generalization.

---

##  Dataset
The dataset contains solar module images showing:
- Cracks  
- Inactive regions  
- Combinations of both  
- Undamaged panels  

Images come from real solar installations and reflect natural variations such as lighting, degradation, and structure.

This is a **multi-label** problem—each image can belong to multiple defect classes.

---

##  Normalization & Preprocessing
All images are normalized using dataset-wide statistics:

\[
x^\* = \frac{x - \mu}{\sigma}
\]

This ensures compatibility with evaluation environments and stable model training.

Additional preprocessing steps:
- Resizing images
- Data augmentation (flips, rotations, contrast changes)
- PyTorch `Dataset` and `DataLoader` setup

---

##  Model Architecture
The core model is a **ResNet-style convolutional neural network** implemented from scratch in PyTorch.  

Key features:
- Residual blocks
- Global average pooling
- Multi-label sigmoid output layer
- Optional dropout and batch normalization
- Customizable depth and width

---

##  Training Pipeline
The training loop includes:
- Multi-label loss functions (e.g., BCEWithLogitsLoss)
- Optimizers (Adam / SGD)
- Learning rate scheduling
- Periodic validation
- Checkpoint saving for best model performance

The model is evaluated with:
- Precision  
- Recall  
- F1 score (macro/micro)  
- Per-class metrics  

---

##  Results & Experiments
Throughout the project, different strategies were explored to improve accuracy and robustness, including:
- Data augmentation
- Architectural refinements
- Regularization (dropout, weight decay)
- Pretraining vs. training from scratch

The best-performing model achieved strong mean F1 scores on validation data.

---
