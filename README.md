![header](https://capsule-render.vercel.app/api?type=venom&height=300&color=gradient&text=Spine%20Segmentation&fontAlignY=40&desc=nnU-Net%20v2%20%7C%20Multi-class%20Vertebrae%20Segmentation&descAlignY=65)

[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/waseemalmazrua/SPINE_NNU-net)
[![Framework](https://img.shields.io/badge/Framework-nnU--Net%20v2-blue)](https://github.com/MIC-DKFZ/nnUNet)
[![Domain](https://img.shields.io/badge/Domain-Medical%20Imaging-red)](#)
[![Status](https://img.shields.io/badge/Status-Research%20%26%20Portfolio-success)](#)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Waseem%20Almazrua-blue?logo=linkedin)](https://www.linkedin.com/in/waseemalmazrua/)

---

## 🦴 Overview

*Welcome to the Spine Segmentation project using **nnU-Net v2***  

This repository presents a **multi-class vertebrae segmentation project**
trained on volumetric **CT scans** using **nnU-Net v2**, a state-of-the-art
self-configuring framework for medical image segmentation.

The project is designed to be:
- **Research-oriented** (clear metrics, configuration, limitations)
- **Portfolio-ready** (visual results, concise explanation, reproducibility)

---

## 🎯 Task Description

- **Problem**: Automatic segmentation of individual vertebrae  
- **Input**: 3D CT volumes (`.nii.gz`)  
- **Output**: Multi-class vertebrae masks  
- **Classes**: 25 anatomical labels  
- **Approach**: Fully automatic nnU-Net v2 pipeline  

---

## 🧠 Model & Framework

This project uses **nnU-Net v2**, which automatically configures:

- Network architecture  
- Patch size & resolution  
- Normalization strategy  
- Loss functions  
- Training schedule  

No manual architecture tuning was performed.

---

## 🏷️ Vertebrae Labels

The segmentation includes the following structures:

- **Cervical**: C1 – C7  
- **Thoracic**: T1 – T12  
- **Lumbar**: L1 – L5  
- **Sacral**: S1  

**Total classes**: 25 (excluding background)

---

## 🦴 Segmentation Demo

<p align="center">
  <img src="demo.gif" width="500"/>
</p>

The demo illustrates 3D vertebral segmentation results produced by the trained model.

---

## 📈 Training Progress

<p align="center">
  <img src="progress.png" width="650"/>
</p>

The plot shows training and validation trends across epochs,
including convergence behavior and stability.

---

## 📊 Results Summary

| Metric | Value |
|------|------|
| Framework | nnU-Net v2 |
| Modality | CT |
| Classes | 25 vertebrae |
| Training cases | 696 |
| Validation cases | 175 |
| Foreground Dice | ~0.70 |

### Key Observations
- Strong performance on **mid-spine vertebrae (T/L)**  
- Lower Dice on **extremities (C/S)** due to class imbalance and anatomical variability  
- No training collapse or overfitting observed  

---

## 🧪 Evaluation Notes

- Dice = NaN indicates the absence of a given vertebra in both prediction and ground truth  
- This behavior is expected and mathematically valid  
- Full metrics are available in `eval_spine.json`

---

