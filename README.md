# 🖼️ Image Captioning using Transformer

An image captioning system built using a Transformer-based architecture. This project generates descriptive captions for images by extracting image features using a CNN (ResNet) and generating captions with a Transformer decoder.

---



## 🎯 Overview

This project aims to generate captions for images using a combination of deep learning models:
- **CNN Encoder** (ResNet-50) extracts features from images.
- **Transformer Decoder** generates captions from the encoded image features.

The model is trained on a popular image captioning dataset and can generate detailed captions for unseen images.

---

## ✅ Features

- Image caption generation with attention-based Transformer
- Preprocessing of images and captions
- Detailed model architecture with ResNet50 and Transformer decoder
- Evaluation using BLEU and ROUGE scores
- Easy-to-use interface for both training and inference

---

## 📂 Dataset

- **Dataset used**: MS COCO / Flickr8k / Flickr30k
  - The dataset contains images and their corresponding captions.
  - The captions are preprocessed, tokenized, and paired with the respective image features.

---

## 🛠 Tech Stack

- Python
- PyTorch
- NumPy, Pandas
- OpenCV, PIL
- Matplotlib (for visualizations)
- HuggingFace Transformers (optional for advanced transformers)

---

## 🧠 Model

- **Encoder**: ResNet-50 (pre-trained) to extract image features.
- **Decoder**: Transformer (with multi-head attention) to generate captions.
- **Training**: Fine-tuned on the image-captioning task using cross-entropy loss.
- **Loss Function**: Cross-Entropy Loss for predicting the next word in the caption.

---

## ⚙️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/image-captioning-transformer.git
cd image-captioning-transformer
