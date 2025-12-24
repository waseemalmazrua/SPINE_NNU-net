# 🦴 SPINE Segmentation using nnU-Net v2

This repository presents a **multi-class spine and vertebrae segmentation project**
implemented using **nnU-Net v2**, a state-of-the-art self-configuring framework for
medical image segmentation.

The model is trained to segment individual vertebrae from volumetric medical images
(CT / MRI) and evaluated using Dice and IoU metrics.

---

## 📌 Project Overview

- **Task**: Multi-class vertebrae segmentation  
- **Framework**: nnU-Net v2  
- **Classes**: 25 vertebral labels  
- **Domain**: Medical Imaging (Spine)  
- **Evaluation**: Dice Score, IoU, FP, FN  

---

## 🧠 Model Architecture

The project relies on **nnU-Net v2**, which automatically configures:
- Network architecture
- Patch size
- Normalization
- Loss functions
- Training schedule

No manual architecture engineering is required.

---

## 📈 Training Progress

The following plot shows the **training and validation curves**
(loss + Dice) during training:

<p align="center">
  <img src="progress.png" width="600"/>
</p>

---

## 🦴 Segmentation Demo

Below is a demo visualization of the predicted spine segmentation
showing the 3D structure of the vertebrae:

<p align="center">
  <img src="demo.gif" width="450"/>
</p>

---

## 📊 Evaluation Results (Summary)

- **Foreground Dice**: ~0.70  
- **Mid-spine vertebrae (T/L)**: Dice > 0.80  
- **Extremities (C/S)**: Lower Dice due to class imbalance and anatomical variability  

Key observations:
- Strong performance on large, well-defined vertebrae  
- Reduced accuracy on small or rarely occurring classes  
- No prediction collapse observed  

---

