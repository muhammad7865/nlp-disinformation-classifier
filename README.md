<div align="center">

# NLP Disinformation & Social Engineering Text Classifier

**An empirical natural language processing pipeline benchmarking feature representations (TF-IDF vs. PCA) across Naive Bayes, Support Vector Machines (SVM), and Multi-Layer Perceptrons (MLP).**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLTK-NLP-388E3C?style=flat-square)](https://www.nltk.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

</div>

---

## 📌 Executive Summary

Digital social engineering, deceptive narratives, and coordinated disinformation campaigns exploit subtle linguistic cues and lexical distribution shifts to bypass standard heuristic filters. 

This repository presents an end-to-end Machine Learning pipeline that extracts high-dimensional textual representations from raw unstructured corpora, performs dimensionality reduction via Principal Component Analysis (PCA), and benchmarks multiple supervised architectures under rigorous **10-fold Stratified Cross-Validation** to combat class imbalance.

---

## 🔬 Experimental Pipeline

```text
Raw Text Corpora (Truthful vs. Deceptive News)
                        │
                        ▼
            [ Preprocessing Pipeline ]
  • Regex tokenization & lowercasing
  • Stop-word filtration (NLTK corpus)
  • Morphological lemmatization (WordNet)
                        │
                        ▼
         [ Feature Engineering Stage ]
  • Sub-linear TF-IDF n-gram vectorization (10,000+ features)
  • Dimensionality Reduction via PCA (2D latent projection)
                        │
                        ▼
       [ Supervised Model Benchmarking ]
  ┌─────────────────────┼─────────────────────┐
  ▼                     ▼                     ▼
[ Naive Bayes ]       [ Support Vector ]   [ Neural Network ]
• Gaussian NB         • Linear SVM         • MLP Classifier
• Multinomial NB      • RBF Kernel SVM       (100 hidden units)
• Bernoulli NB
                        │
                        ▼
   [ Evaluation & Validation Protocol ]
  • Random 70/30 Split vs. Stratified 70/30 Split
  • 10-Fold Stratified Cross-Validation
  • Macro Precision, Recall, F1-Score & Confusion Matrix Analysis
```

---

## 📊 Empirical Benchmarks & Comparison

Evaluated on the widely recognized **ISOT Fake News Dataset** (containing labeled articles across political and global domains):

| Classifier Family | Specific Model Architecture | Feature Representation | Validation Strategy | Primary Metric (Accuracy) |
| :--- | :--- | :--- | :--- | :---: |
| **Probabilistic** | Multinomial Naive Bayes | TF-IDF (Sparse Matrix) | 10-Fold Stratified CV | **High Precision** |
| **Probabilistic** | Bernoulli Naive Bayes | Binary Word Occurrence | 10-Fold Stratified CV | Strong Baseline |
| **Probabilistic** | Gaussian Naive Bayes | PCA Latent Projection | 10-Fold Stratified CV | Baseline (Low Dimension) |
| **Margin Maximizing** | Linear SVM (`C=1.0`) | TF-IDF Vector Space | 70/30 Stratified Split | **Top Tier Performance** |
| **Margin Maximizing** | RBF Kernel SVM | Scaled TF-IDF Features | 70/30 Stratified Split | High Generalization |
| **Neural Network** | Multi-Layer Perceptron (MLP) | Dense Latent Space | 70/30 Stratified Split | Deep Representation |

---

## 📁 Repository Structure

```text
nlp-disinformation-classifier/
├── notebooks/
│   └── nlp_disinformation_classification.ipynb   # Full interactive analysis, EDA & evaluation
├── src/
│   └── train.py                                  # Modular preprocessing & model training script
├── requirements.txt                              # Required environment libraries
├── .gitignore                                    # Excludes heavy CSV datasets and caches
└── README.md                                     # Research documentation & benchmark summary
```

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/muhammad7865/nlp-disinformation-classifier.git
cd nlp-disinformation-classifier
```

### 2. Install Dependencies
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the Research Notebook
Launch Jupyter Notebook to view data distributions, PCA variance graphs, and confusion matrices:
```bash
jupyter notebook notebooks/nlp_disinformation_classification.ipynb
```

---

## 👤 Author

**Muhammad Awais Mohsin**  
*COMSATS University Islamabad — Cybersecurity & Applied AI Systems Engineer*  
* GitHub: [@muhammad7865](https://github.com/muhammad7865)  
* LinkedIn: [Muhammad Awais Mohsin](https://linkedin.com/in/muhammad-awais-mohsin-b1250026a)  
* Portfolio: [miihammadawais.vercel.app](https://miihammadawais.vercel.app)