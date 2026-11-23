# NLP-Based Disinformation and Social Engineering Classifier

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end natural language processing (NLP) and supervised machine learning classification pipeline designed to detect deceptive textual narratives, manipulated digital news, and social engineering language across benchmark corpora.

---

## Technical Pipeline

1. **Text Preprocessing & Linguistic Normalization:**
   - Case folding, punctuation removal, and regex tokenization.
   - NLTK stop-word filtration and WordNet lemmatization.
2. **Feature Extraction:**
   - Sub-linear Term Frequency-Inverse Document Frequency (TF-IDF) n-gram vectorization across 10,000+ features.
   - Principal Component Analysis (PCA) for lower-dimensional latent semantic projection.
3. **Model Benchmarking:**
   - **Support Vector Machines (SVM):** Evaluated both Linear and Radial Basis Function (RBF) kernels.
   - **Naive Bayes:** Gaussian, Multinomial, and Bernoulli classifiers.
   - **Multi-Layer Perceptron (MLP):** Deep feed-forward neural architecture with regularized loss.
4. **Validation:**
   - Benchmarked using both randomized hold-out splits and 10-fold Stratified Cross-Validation to guarantee balance across imbalanced text splits.

---

## Repository Structure

`	ext
nlp-disinformation-classifier/
├── notebooks/
│   └── nlp_disinformation_classification.ipynb   # Complete analysis and benchmarking
├── src/
│   └── train.py                                  # Modular preprocessing and training script
├── requirements.txt
├── .gitignore
└── README.md
`

---

## Quickstart

### 1. Installation
`ash
git clone https://github.com/muhammad7865/nlp-disinformation-classifier.git
cd nlp-disinformation-classifier
pip install -r requirements.txt
`

### 2. Run Notebook
`ash
jupyter notebook notebooks/nlp_disinformation_classification.ipynb
`

---

## Author
**Muhammad Awais Mohsin**  
COMSATS University Islamabad | Cybersecurity & AI Systems Engineer  
- GitHub: [@muhammad7865](https://github.com/muhammad7865)  
- LinkedIn: [muhammad-awais-mohsin-b1250026a](https://linkedin.com/in/muhammad-awais-mohsin-b1250026a)
