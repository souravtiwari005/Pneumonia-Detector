# Pneumonia-detector
# 🫁 Pneumonia Detection using Deep Learning

![Python](https://img.shields.io/badge/Python-3.10-blue) ![Accuracy](https://img.shields.io/badge/Accuracy-97.2%25-green)

A deep learning model that detects pneumonia from chest X-ray images using ResNet50. Trained on 5,800+ real X-ray images from Kaggle.

## 🔗 Live Demo
👉 [Try it on Gradio](https://f20eef95ee30b73bd2.gradio.live/)

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

| Split | Normal | Pneumonia |
|-------|--------|-----------|
| Train | 1,341 | 3,875 |
| Test | 234 | 390 |

## ⚙️ How to Run Locally

bash
git clone https://github.com/souravtiwari005/pneumonia-detector
cd pneumonia-detector
pip install fastai gradio
python app.py


## 🚧 Challenges Overcome
- *Class Imbalance:* Dataset had 3x more pneumonia cases than normal. Solved using class-weighted loss function (penalizing normal misclassifications 3x more)
- *Runtime Resets:* Learned to export and persist model artifacts immediately after training
- *Transform Compatibility:* Navigated FastAI/Gradio version conflicts during deployment

## 📂 Project Structure
pneumonia-detector/
├── app.py                  # Gradio web app
├── requirements.txt        # Dependencies
├── notebooks/
│   └── training.ipynb      # Full training notebook
└── README.md

## ⚠️ Disclaimer
This model is for educational purposes only and should not be used for medical diagnosis.
