<h1 align="center">🌾 MTL-ViT: Multi-Task Vision Transformer for Rice Crop Health</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Vision_Transformer-ViT_Base-blueviolet?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Tasks-Multi--Task_Learning-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" />
</p>

<p align="center">
  <img src="https://github.com/Abhineetraj07/MTL-ViT-Rice-Disease-Detection/actions/workflows/ci.yml/badge.svg" alt="CI" />
  <img src="https://img.shields.io/badge/Accuracy-96.2%25-brightgreen?style=flat-square" />
  <img src="https://img.shields.io/badge/AUC-0.995-blue?style=flat-square" />
  <img src="https://img.shields.io/github/last-commit/Abhineetraj07/MTL-ViT-Rice-Disease-Detection?style=flat-square&color=orange" />
</p>

<p align="center">
  <b>Simultaneously detects rice leaf diseases AND nutrient deficiencies using a shared Vision Transformer backbone — achieving 96%+ accuracy on both tasks.</b>
</p>

---

## 🎯 What It Does

Most crop disease models tackle **one problem at a time**. This system uses **multi-task learning** to:
- 🦠 Classify **6 types of rice leaf diseases** (Bacterial Leaf Blight, Brown Spot, Leaf Blast, etc.)
- 🌿 Detect **3 types of nutrient deficiencies** (Nitrogen, Phosphorus, Potassium)

Both predictions happen in a **single forward pass** from the same ViT backbone — more efficient and more accurate than two separate models.

---

## 📊 Results

| Task | Accuracy | AUC Score |
|------|----------|-----------|
| Disease Classification | **96.2%** | **0.995** |
| Nutrient Deficiency | **96.0%** | **0.997** |

> Evaluated on held-out test set. Training curves, confusion matrices, ROC and PR curves available in [`/outputs`](./outputs).

---

## 🏗️ Architecture

```
Input Image (224×224×3)
        ↓
┌─────────────────────────────┐
│   ViT-Base-Patch16-224      │
│   (Pretrained on ImageNet)  │
│   86M parameters            │
└─────────────────────────────┘
        ↓
   Shared Feature Vector (768 dims)
        ↓
   ┌────────┴────────┐
   ↓                  ↓
Disease Head       Nutrient Head
(6 classes)        (3 classes)
```

The model uses **transfer learning** from a ViT-Base pretrained on ImageNet-21k, fine-tuned end-to-end with a joint loss function:

```
Total Loss = α × CrossEntropy(disease) + β × CrossEntropy(nutrient)
```

---

## 🔬 Classes

| Task | Classes |
|------|---------|
| **Disease** | Bacterial Leaf Blight, Brown Spot, Healthy, Leaf Blast, Leaf Scald, Narrow Brown Spot |
| **Nutrient Deficiency** | Nitrogen (N), Phosphorus (P), Potassium (K) |

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Abhineetraj07/MTL-ViT-Rice-Disease-Detection.git
cd MTL-ViT-Rice-Disease-Detection
pip install -r requirements.txt
```

### 2. Run Inference on a Leaf Image

```bash
python src/inference.py --model models/MTL_ViT_Complete.pth --image path/to/leaf.jpg
```

**Sample Output:**
```
Disease Prediction:   Leaf Blast       (confidence: 94.3%)
Nutrient Deficiency:  Nitrogen (N)     (confidence: 88.7%)
```

### 3. Launch the Gradio Web App

```bash
python app.py
```

Open `http://localhost:7860` → upload a rice leaf image → get instant predictions.

---

## 📈 Visual Results

| Training History | Confusion Matrix |
|-----------------|-----------------|
| ![Training](outputs/training_history.png) | ![Confusion](outputs/confusion_matrices.png) |

| ROC Curves | Precision-Recall |
|------------|-----------------|
| ![ROC](outputs/roc_curves.png) | ![PR](outputs/precision_recall_curves.png) |

---

## 📁 Project Structure

```
MTL-ViT-Rice-Disease-Detection/
├── src/
│   ├── model.py          # MTL-ViT architecture definition
│   ├── dataset.py        # Custom PyTorch Dataset with augmentations
│   └── inference.py      # Single-image prediction script
├── outputs/              # Training plots & evaluation graphs
├── app.py                # Gradio web demo
├── requirements.txt
└── README.md
```

---

## 🌐 Dataset

Based on publicly available rice disease image datasets. The model was trained on:
- Images resized to **224×224**
- Augmentations: Random flip, rotation, color jitter
- Train/Val/Test split: **70/15/15**

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![timm](https://img.shields.io/badge/timm-ViT_Backbone-orange?style=flat-square)
![Gradio](https://img.shields.io/badge/Gradio-Web_Demo-FF7C00?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![matplotlib](https://img.shields.io/badge/matplotlib-Visualization-blue?style=flat-square)

---

## 👨‍💻 Author

**Abhineet Raj** · CS @ SRM Institute of Science & Technology  
🌐 [Portfolio](https://aabhineet07-portfolio.netlify.app/) · 🐙 [GitHub](https://github.com/Abhineetraj07)

---

## 📄 License

This project is licensed under the MIT License.
