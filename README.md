# Solar Panel Defect Classification (PyTorch)

A deep learning project for **multi-label classification** of solar panel defects using a custom **ResNet-based CNN** in PyTorch. The goal is to detect two defect types:
- **Cracks**
- **Inactive regions**

Each image may contain *zero, one, or multiple* defects, making this a multi-label problem.

---

## Features
- Custom **ResNet-style** model (residual blocks, GAP, sigmoid output)
- Full PyTorch training pipeline (Adam optimizer, BCE loss, early stopping)
- Dataset loading, preprocessing, and normalization
- GPU-compatible implementation
- Training/validation loss visualization (`losses.png`)
- Modular and extendable code structure

---

## Dataset & Preprocessing
- Images resized to **1024 px height**
- Normalized using dataset-level mean and std:

\[
x^\* = \frac{x - \mu}{\sigma}
\]

- Supports optional augmentation through the `Dataset` class
- Multi-label targets loaded from `data.csv`
---


## Training

Install dependencies and run:

\`\`\`bash
python train.py
\`\`\`

This will:

- Load and preprocess data  
- Train the ResNet model  
- Plot the loss curve  
- Save training history  

---

## Metrics

Evaluation includes:

- **Precision**
- **Recall**
- **F1 (macro & micro)**
- **Per-class accuracy**
- **Loss curves and error analysis**

---

## Results

Experiments explored:

- Data augmentation (flips, rotations, contrast)
- Regularization (dropout, weight decay)
- Changes in CNN depth and width
- Alternative loss functions (e.g., BCEWithLogitsLoss)
- Optimizer and learning rate tuning

The best-performing configuration achieved stable F1 scores and strong validation performance.
