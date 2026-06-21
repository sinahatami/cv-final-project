<div align="center">
  <h1>🧬 Cancer Prediction: Pap Smear Analysis</h1>
  <p>
    <strong>A Computational Vision Project on Cervical Cell Classification</strong>
  </p>
  <p>
    <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python Version">
    <img src="https://img.shields.io/badge/TensorFlow-2.x-orange.svg" alt="TensorFlow">
    <img src="https://img.shields.io/badge/Keras-Pretrained-red.svg" alt="Keras">
    <img src="https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-yellow.svg" alt="Scikit-Learn">
  </p>
</div>

<hr>

## 🔬 Project Overview

This project explores the classification and analysis of cervical cells using the **SIPaKMeD dataset**. By comparing traditional **Handcrafted Feature Engineering** (like HOG) with **Deep Learning-based automated extraction** (via VGG16), this research aims to improve the early detection of abnormalities in Pap smear images.

Developed for the **Computational Vision** course at the **University of Genoa**, this study investigates how various feature extraction methods impact the accuracy of identifying pathological cells.

- **Professor:** Francesca Odone
- **Project Advisor:** Vito Paolo Pastore
- **Team:** Shayan Alvansazyazdi & Sina Hatami

## 📂 Dataset: SIPaKMeD

The dataset comprises **4,049 isolated images** of cells, meticulously categorized into five distinct classes:

| Category | Description |
|:---:|:---|
| **Superficial-Intermediate** | Flat or polygonal cells; the most common type in Pap tests. |
| **Parabasal** | Small, immature epithelial cells with cyanophilic cytoplasm. |
| **Koilocytotic** | Cells with a large perinuclear cavity and hyperchromatic nuclei. |
| **Dyskeratotic** | Prematurely keratinized cells, often found in 3D clusters. |
| **Metaplastic** | Uniform cells with prominent borders and eccentric nuclei. |

## 🤖 Methodology & Pipelines

The project implements three distinct analytical approaches for feature extraction and classification:

### 1. Feature Extraction Strategies
- **Deep Learning (Automated):** Utilizing a **VGG16** model pretrained on ImageNet, extracting high-level semantic features from the `fc2` layer.
- **Handcrafted (Manual):** Computing **Histogram of Oriented Gradients (HOG)** to capture local shape and texture patterns manually.

### 2. Analytical Models
- **Unsupervised Learning:** **K-Means Clustering** applied to VGG16 features to identify natural groupings and patterns in the data without prior labels.
- **Supervised Learning:** **Support Vector Machine (SVM)** with a linear kernel for definitive classification of the extracted features.
- **Validation:** **Stratified K-Fold Cross-validation (5 folds)** is employed to ensure model robustness and prevent overfitting.

## 📊 Results Comparison

Our findings demonstrate that **Pretrained Features (VGG16)** significantly outperform traditional handcrafted methods in terms of both accuracy and F1-score.

| Method | Feature Set | Accuracy | F1-Score |
|:---|:---|:---:|:---:|
| KMeans Clustering | VGG16 Features | 26% | 0.28 |
| SVM Classifier | HOG (Handcrafted) | 40% | 0.36 (Mean) |
| SVM Classifier | VGG16 (Pretrained) | 85% | 0.89 (Mean) |
| **SVM (K-Fold CV)** | **VGG16 (Pretrained)** | **87%** | **0.87** |

> **Note:** The high performance of the VGG16-SVM pipeline (87% accuracy) suggests that pretrained CNN features are highly effective for medical image classification tasks, even when directly compared to domain-specific handcrafted features.

## 🛠 Tech Stack

- **Deep Learning:** TensorFlow, Keras (VGG16 architecture)
- **Machine Learning:** Scikit-learn (SVM, KMeans, K-Fold CV)
- **Image Processing:** OpenCV, Scikit-image (HOG)
- **Data Analysis:** NumPy, Matplotlib

## ⚙️ Setup & Installation

To run this project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sinahatami/cv-final-project.git
   cd cv-final-project
   ```

2. **Create a virtual environment (Optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Quick Start

To replicate the results, explore the Jupyter Notebooks included in the `final-project/notebooks/` directory:

1. **`final-project/notebooks/cv_final.ipynb`**: The main execution pipeline including deep feature extraction, clustering, and SVM classification.
2. **`final-project/notebooks/first_output.ipynb`**: Contains initial visual analysis, data exploration, and cluster visualizations.
3. **`final-project/notebooks/cv_project.ipynb`**: Additional project explorations.

## 📄 References & Documentation

The foundational literature and the final project report can be found in the respective directories:
- **`final-project/papers/`**: Contains the reference materials and research papers.
- **`final-project/docs/`**: Contains the final detailed project report (`final-report.pdf`).

## 📚 Assignments

Lab assignments from the course can be found in the `assignments/` directory, including exercises on corner matching, 3D vision, optical flow, autoencoders, and object detection.
