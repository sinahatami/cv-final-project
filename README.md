# 🧬 Cervical Cancer Prediction: Pap

# Smear Analysis

This project explores the classification and analysis of cervical cells using the **SIPaKMeD
dataset**. By combining traditional **Handcrafted Feature Engineering** with **Deep
Learning-based automated extraction** , this research aims to improve the early detection of
abnormalities in Pap smear images.

## 🔬 Project Overview

Developed for the **Computational Vision** course at the University of Genoa , this study
investigates how various feature extraction methods impact the accuracy of identifying
pathological cervical cells.

```
● Professor: Francesca Odone
● Project Advisor: Vito Paolo Pastore
● Team: Shayan Alvansazyazdi & Sina Hatami
```
## 📂 Dataset: SIPaKMeD

The dataset comprises **4,049 isolated images** of cervical cells, meticulously categorized into

```
Category Description
```
```
Superficial-Intermediate Flat or polygonal cells; the most common type in Pap
tests.
```
```
Parabasal Small, immature epithelial cells with cyanophilic
cytoplasm.
```
```
Koilocytotic Cells with a large perinuclear cavity and hyperchromatic
nuclei.
```

```
Dyskeratotic Prematurely keratinized cells, often found in 3D clusters.
```
```
Metaplastic Uniform cells with prominent borders and eccentric
nuclei.
```
## 🤖 Methodology & Pipelines

The project implements three distinct analytical approaches:

### 1. Feature Extraction Strategies

```
● Deep Learning (Automated): We utilize a VGG16 model pretrained on ImageNet,
extracting features from the 'fc2' layer.
● Handcrafted (Manual): We compute Histogram of Oriented Gradients (HOG) to
capture local shape and texture patterns.
```
### 2. Analytical Models

```
● Unsupervised: KMeans Clustering used on VGG16 features to identify natural
groupings in the data.
● Supervised: Support Vector Machine (SVM) with a linear kernel for definitive
classification.
● Validation: Stratified K-Fold Cross-validation (5 folds) to ensure model robustness.
```
## 📊 Results Comparison

Our findings demonstrate that **Pretrained Features (VGG16)** significantly outperform traditional
handcrafted methods.

```
Method Feature Set Accurac
y
```
```
F1-Score
```
```
KMeans Clustering VGG16 Features 26% 0.
```

```
SVM Classifier HOG (Handcrafted) 40% 0.36 (Mean)
```
```
SVM Classifier VGG16 (Pretrained) 85% 0.89 (Mean)
```
```
SVM (K-Fold CV) VGG16 (Pretrained) 87% 0.
```
```
Note: The high performance of the VGG16-SVM pipeline (87% accuracy) suggests
that pretrained CNN features are highly effective for medical image classification
tasks.
```
## 🛠 Tech Stack

```
● Deep Learning: TensorFlow, Keras (VGG16)
● Machine Learning: Scikit-learn (SVM, KMeans, K-Fold)
● Image Processing: OpenCV, Scikit-image (HOG)
● Data Analysis: NumPy, Matplotlib
```
## 🚀 Quick Start

To replicate the results, explore the Jupyter Notebooks included in this repository:

1. **CV_final.ipynb** : The main execution pipeline including clustering and classification.
2. **firstoutput.ipynb** : Contains initial visual analysis and cluster visualizations.


