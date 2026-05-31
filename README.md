---
title: Pneumonia Detector
emoji: 🫁
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
---

# 🫁 Pneumonia Detector

A deep learning model that detects pneumonia from chest X-ray images using ResNet50 and FastAI.

## How to use
1. Upload a chest X-ray image
2. Click Submit
3. Get prediction with confidence score

## Model
- Architecture: ResNet50 (pretrained on ImageNet)
- Framework: FastAI + PyTorch
- Validation Accuracy: 97.2%
- Pneumonia Recall: 98%

## Disclaimer
For educational purposes only. Not for medical diagnosis.

## 🔗 Live Demo
👉 [Try it on Gradio spaces](https://f20eef95ee30b73bd2.gradio.live/)

## 📊 Results

| Metric | Score |
|--------|-------|
| Validation Accuracy | 97.2% |
| Test Accuracy | 87% |
| Pneumonia Recall | 98% |
| Normal Precision | 96% |

## 🧠 Model Architecture
- *Base Model:* ResNet50 (pretrained on ImageNet)
- *Framework:* FastAI + PyTorch
- *Training Strategy:* Transfer learning — frozen base → fine-tune full model
- *Loss Function:* Cross Entropy with class weights (3:1) to handle data imbalance

## 📁 Dataset
[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) — Kaggle
