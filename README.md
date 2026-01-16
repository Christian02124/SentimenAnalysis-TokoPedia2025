# SentimenAnalysis-TokoPedia2025

Sentiment analysis project on Indonesian e-commerce (Tokopedia) product reviews using multiple machine learning approaches.

## 📌 Project Overview
This repository explores sentiment classification (positive / negative) on Tokopedia customer reviews written in Bahasa Indonesia.

The project focuses on:
- Cleaning and preprocessing real-world noisy text data
- Handling imbalanced datasets
- Comparing traditional ML and deep learning approaches
- Evaluating and interpreting model results

## 🧠 Approaches Implemented
This project implements **two different modeling approaches**:

1. **TF-IDF + Logistic Regression**
   - Traditional machine learning approach
   - Interpretable features (top positive & negative words)
   - Fast training and strong baseline performance

2. **LSTM (TensorFlow)**
   - Deep learning approach for sequence modeling
   - Captures word order and contextual relationships
   - Used as experimental comparison model

Each approach is organized in its own folder with dedicated notebooks.

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- TensorFlow / Keras
- Sastrawi (Indonesian stopwords)
- Matplotlib

## 📊 Results & Evaluation
- Confusion Matrix
- Precision, Recall, F1-score
- Class distribution before & after balancing
- Top influential words for sentiment prediction

## 🎯 Purpose
This project is created as a **portfolio project** to demonstrate:
- End-to-end ML workflow
- NLP for Indonesian language
- Model comparison and evaluation
- Readable and reproducible experimentation
