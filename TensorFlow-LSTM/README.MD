# Sentiment Analysis on Indonesian Product Reviews

## 📌 Overview
This project builds a sentiment analysis model to classify Indonesian product reviews into **positive** or **negative** sentiments using **TensorFlow LSTM**.

The goal of this project is to demonstrate a complete **deep learning pipeline**, from text preprocessing to model training, evaluation, and inference readiness.

---

## 📊 Dataset
- Source: Indonesian product reviews (Tokopedia)
- Columns used:
  - `review_text`
  - `rating`
- Sentiment labeling:
  - Rating ≥ 4 → Positive
  - Rating ≤ 2 → Negative
  - Rating = 3 removed (neutral)

---

## 🧹 Data Preprocessing
- Removed neutral ratings
- Text cleaning:
  - Lowercasing
  - Removing newline characters and symbols
- Class imbalance handled using **oversampling** to balance positive and negative reviews

---

## ⚙️ Feature Engineering
- Tokenization using Keras `Tokenizer`
- Integer sequences padded to `max_len=100` using `pad_sequences`
- Vocabulary size: 5000 (`max_words`)

---

## 🤖 Model
- Architecture:
  - Embedding Layer (64 dim)
  - LSTM Layer (64 units)
  - Dropout (0.5)
  - Dense Layer (32 units, ReLU)
  - Output Layer (1 unit, Sigmoid)
- Train-test split: 80:20 (stratified)
- Evaluation metrics:
  - Accuracy
  - Optional: Precision, Recall, F1-score

---

## 📈 Results
- Balanced sentiment distribution after oversampling
- Model performs well on both positive and negative classes
- Training & validation accuracy visualized per epoch
![Accuracy](images/Tensor.png)
---

## 🚀 Inference Example
```python
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = joblib.load("models/saved_model/tokenizer.pkl")
model = load_model("models/saved_model/tokopedia_sentiment_model.keras")

text = "Barangnya bagus dan pengiriman cepat"
seq = tokenizer.texts_to_sequences([text])
pad_seq = pad_sequences(seq, maxlen=100, padding='post')
prediction = model.predict(pad_seq)[0][0]

sentiment = "positive" if prediction >= 0.5 else "negative"
print(sentiment)

