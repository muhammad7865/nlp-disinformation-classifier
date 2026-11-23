\"\"\"
NLP-Based Disinformation and Fake News Classifier
Author: Muhammad Awais Mohsin
\"\"\"

import os
import re
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text)).lower()
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)

def main():
    print('=== NLP Disinformation Classifier Training Pipeline ===')
    print('Pipeline initialized: Preprocessing -> TF-IDF -> Model Benchmarking (Naive Bayes, SVM, MLP)')

if __name__ == '__main__':
    main()
