# DL Exercise 4 — PyTorch & Classification Challenge  
*(Based on the FAU Pattern Recognition Lab materials)*  
:contentReference[oaicite:0]{index=0}

##  Overview
This exercise introduces students to **PyTorch**, modern deep learning practices, and real-world multi-label image classification.  
You will implement, train, and evaluate a variation of the **ResNet** architecture on a dataset of solar-panel defects.

The project consists of two main parts:

1. **Mandatory** — Implement, train, and submit a PyTorch model  
2. **Optional** — A challenge to build the *best* model and compete with classmates  

---

##  Goals  
- Learn the basics of **PyTorch** and tensor-based computation  
- Implement & train a **ResNet-like architecture**  
- Perform **multi-label image classification** on solar module images  
- Compete in a friendly deep learning challenge with peers

---

##  Dataset: Solar Panel Defect Classification

Solar modules:
- Contain many cells  
- Can degrade due to wind, transport, hail, etc.  
- Show various defect types (e.g., **cracks**, **inactive regions**)  
- Each panel may have **zero, one, or multiple defects → multi-label classification**

Examples (see page 8 of PDF):  
- Crack on polycrystalline module  
- Inactive region  
- Combined cracks & inactive regions  

---

##  Normalization Requirements
Your preprocessing must exactly match the server normalization:

For each pixel \( x \):
\[
x^\* = \frac{x - \mu}{\sigma}
\]

Where:
- \( \mu \): global mean provided in the exercise  
- \( \sigma \): global standard deviation  

 **Important**: If your normalization differs, the server evaluation will be wrong.

---

##  About PyTorch
You will train your model using **PyTorch**, a widely-used deep learning framework:

- Open-source  
- Tensor-based computation similar to NumPy  
- Dynamic computation graphs  
- Autograd for backpropagation  
- GPU acceleration  
- Rich ecosystem and documentation  
  - 60-min blitz  
  - PyTorch tutorials  
  - Examples and notebooks  

---

##  The Challenge
Improve the baseline ResNet by experimenting with:

-  Architecture design  
-  Pretraining  
-  Regularization techniques  
-  Data augmentation  



---


