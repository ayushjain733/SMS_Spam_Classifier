# 📱 SMS Spam Classifier

[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen.svg)](https://message-classifier.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)

A Machine Learning-based web application that detects whether an SMS or email message is **Spam** or **Not Spam (Ham)** using Natural Language Processing (NLP).

## 🔗 Live Application
Try the live app here: **[SMS Spam Classifier](https://message-classifier.streamlit.app/)**

## 💡 Features
* **Interactive UI**: Built with Streamlit for a clean, user-friendly experience.
* **Text Preprocessing**: Utilizes NLTK for tokenization, lowercasing, special character removal, stop-word removal, and stemming.
* **Feature Extraction**: Uses TF-IDF (Term Frequency-Inverse Document Frequency) to vectorize textual data.
* **High Accuracy Classification**: Powered by a Multinomial Naive Bayes model tailored for text classification.

## 🛠️ Tech Stack
* **Language**: Python
* **Frontend**: Streamlit
* **Machine Learning**: Scikit-Learn (MultinomialNB, TfidfVectorizer)
* **NLP**: NLTK (PorterStemmer, Stopwords)
* **Data Manipulation**: Pandas, NumPy

## 🚀 Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/ayushjain733/SMS_Spam_Classifier.git
cd SMS_Spam_Classifier
```

**2. Install Dependencies**
Make sure you have Python installed. Run the following command to install the required libraries:
```bash
pip install -r requirements.txt
```

**3. Run Streamlit App**
```bash
streamlit run app.py
```

## 🧠 How it Works
1. **Input**: The user enters a message into the web interface.
2. **Pre-Processing**: The text is cleaned (punctuation removed, converted to lowercase) and stemmed to its root words.
3. **Vectorization**: The cleaned text is converted into numerical vectors using a pre-trained TfidfVectorizer (vectorizer.pkl).
4. **Prediction**: The numerical data is passed to the trained Naive Bayes model (model.pkl), which outputs whether the text exhibits spam patterns.